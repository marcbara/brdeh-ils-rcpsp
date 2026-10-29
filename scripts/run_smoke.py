#!/usr/bin/env python3
"""Run small, durable V0/V5/V6 smoke cases from this repository only."""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def rows(main, state):
    matrix = main.ado_rows(state, 1)
    return [(int(matrix[i, 0]), int(matrix[i, 1]), int(matrix[i, 2]))
            for i in range(matrix.shape[0])]


def feasible(instance: str, makespan: int, starts: list[int]) -> bool:
    from experiments.confirm_ls import _feasible
    return _feasible(instance, makespan, " ".join(str(x) for x in starts)) is None


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--budget", type=int, default=750)
    p.add_argument("--instances", nargs="*", default=["j30/j301_1.sm", "j60/j601_1.sm", "j120/j1201_1.sm"])
    p.add_argument("--variants", nargs="*", choices=("V0", "V5", "V6"), default=("V0", "V5", "V6"))
    p.add_argument("--output", type=Path, default=ROOT / "results" / "smoke" / "smoke.json")
    a = p.parse_args()
    if a.budget < 50:
        raise SystemExit("budget must be at least 50 schedules")
    os.environ.setdefault("JULIA_NUM_THREADS", "1")
    os.environ.setdefault("JULIA_NUM_GC_THREADS", "1")
    os.environ.setdefault("PYTHON_JULIACALL_THREADS", "1")
    # Warm-up is intentionally outside solve timing.  It can be skipped only
    # for constrained diagnostic runners; this does not change a trajectory.
    os.environ.setdefault("BRDEH_SKIP_WARMUP", "1")
    from harness.solvers.brdeh import canonical_seeds, jl, open_process
    from experiments import br_ils
    from experiments.bidir import write_reversed

    print("initializing Julia bridge", flush=True)
    seed_index = 1
    seed = canonical_seeds()[seed_index - 1]
    print("Julia bridge initialized", flush=True)
    result = {"schema": 1, "budget_charged": a.budget, "seed_index": seed_index,
              "seed": seed, "platform": platform.platform(), "runs": []}
    # V0 is the historical B2 multistart without terminal path relinking.
    J0 = jl()
    for instance in a.instances if "V0" in a.variants else ():
        print(f"running V0 {instance}", flush=True)
        t0, c0 = time.perf_counter(), time.process_time()
        process = open_process(instance, {"rules": "klein13", "fbi_max_passes": 1,
                                          "policy_keep_prob": 0.8}, seed)
        slice_ = process.advance(a.budget)
        decoded_makespan, raw_starts = J0.ado_state_starts(process._s)
        starts = [int(x) for x in raw_starts]
        makespan = int(process.best)
        if int(decoded_makespan) != makespan:
            raise RuntimeError(f"V0 elite replay drift: {decoded_makespan} != {makespan}")
        result["runs"].append({"variant": "V0", "instance": instance,
                               "trajectory": process.rows, "makespan": makespan,
                               "charged_schedules": a.budget,
                               "executed_schedules": slice_["executed_schedules"],
                               "cpu_s": time.process_time() - c0,
                               "wall_s": time.perf_counter() - t0,
                               "feasible": feasible(instance, makespan, starts)})

    # V5/V6 share the frozen, exact historical BR-DEH+ILS driver.  V5 is B;
    # V6 is the permanent-switch biased window-rebuild arm.
    if {"V5", "V6"} & set(a.variants):
        J = br_ils._setup_ils()
        fns = (J.seval("ado_gen_start"), J.seval("ado_ils_run!"))
        for instance in a.instances:
            reverse = write_reversed(instance)
            for variant, arm in (("V5", "B"), ("V6", "br_r10_l5")):
                if variant not in a.variants:
                    continue
                print(f"running {variant} {instance}", flush=True)
                t0, c0 = time.perf_counter(), time.process_time()
                state, elite, counters = br_ils.run_one(J, fns, instance, reverse, seed_index, arm, a.budget)
                best_mk, starts = J.ado_best(elite)
                makespan = int(best_mk)
                start_vector = [int(x) for x in starts]
                result["runs"].append({"variant": variant, "arm": arm, "instance": instance,
                                       "trajectory": rows(J, state), "makespan": makespan,
                                       "charged_schedules": a.budget,
                                       "executed_schedules": int(state.schedules),
                                       "counters": counters, "cpu_s": time.process_time() - c0,
                                       "wall_s": time.perf_counter() - t0,
                                       "feasible": feasible(instance, makespan, start_vector)})
    a.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = a.output.with_suffix(a.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    temporary.replace(a.output)
    print(a.output)


if __name__ == "__main__":
    main()
