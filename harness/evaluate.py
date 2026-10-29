"""Gap evaluation of run-driver records against PSPLIB optima.

Aggregation = the one used for every archived number of the project: gap per
(instance, seed) in %, mean over seeds per instance, mean over instances.

Both effort axes read the run's GLOBAL trajectory (best-so-far rows with
global ``schedules`` and ``time_s``): ``value_at(record, axis, limit)`` is the
best makespan among rows with counter <= limit (offline ``value_at``
semantics). Without a limit, the run's final result.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from statistics import mean

from .provenance import REPO_ROOT

OPTIMA = {"j30": REPO_ROOT / "solvers" / "br-deh" / "experiments" / "j30_optimal.csv"}
AXES = ("schedules", "time_s")


def load_optima(family: str = "j30") -> dict[str, int]:
    with open(OPTIMA[family], newline="") as f:
        return {f"{family}/{r['instance']}": int(r["optimal_makespan"]) for r in csv.DictReader(f)}


def gap(makespan: float, optimum: int) -> float:
    return 100.0 * (makespan - optimum) / optimum


def value_at(record: dict, axis: str | None = None, limit: float | None = None):
    if axis is None or limit is None:
        return record["result"]["makespan"]
    if axis not in AXES:
        raise ValueError(f"axis must be one of {AXES}")
    vals = [r["makespan"] for r in record["trajectory"] if r["makespan"] is not None and r[axis] <= limit]
    return min(vals) if vals else None


def mean_gap(records, optima: dict[str, int] | None = None,
             axis: str | None = None, limit: float | None = None) -> dict:
    optima = load_optima() if optima is None else optima
    per_inst = defaultdict(list)
    missing = 0
    for r in records:
        mk = value_at(r, axis, limit)
        if mk is None:          # nothing found yet within the limit
            missing += 1
            continue
        opt = optima[r["instance"]]
        if mk < opt:
            raise ValueError(f"makespan {mk} below optimum {opt} on {r['instance']} seed {r['seed']}")
        per_inst[r["instance"]].append(gap(mk, opt))
    seeds_per_inst = {len(v) for v in per_inst.values()}
    out = {
        "avg_gap_pct": mean(mean(v) for v in per_inst.values()),
        "instances": len(per_inst),
        "seeds_per_instance": sorted(seeds_per_inst),
        "runs": sum(len(v) for v in per_inst.values()),
    }
    if axis is not None:
        out.update(axis=axis, limit=limit, runs_without_solution=missing)
    return out
