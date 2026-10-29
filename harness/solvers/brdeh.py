"""BR-DEH adapter: the level-2→3 contract for this solver.

Two entry points, one code path:

* ``open_process(instance, params, seed) -> BrdehProcess`` — a resumable
  search process (the solver's ``startRun``/``advanceRun!``). The harness
  extends it slice by slice with ``process.advance(b)``.
* ``solve(instance, params, budget, seed) -> Trajectory`` — the one-shot
  contract of artifact documentation, = open a process and advance it once by ``budget``.

Arguments: ``instance`` is relative to the solver's ``inputs/``
(``"j30/j301_1.sm"``); ``params`` is this solver's opaque dict (``PARAM_KEYS``;
named configs are ``PRESETS``); budgets are generated schedules (DEH
constructions + every FBI pass, the solver's ``SCHEDULE_COUNT``), the initial
solution included; ``seed`` is the Xoshiro seed.

Budget semantics (hard cap per process, exact):
``advanceRun!`` stops at the first iteration boundary at or past its target,
so a slice can overshoot by part of one iteration. A process keeps a
CUMULATIVE CAP = sum of the budgets given to it. Each ``advance(b)`` raises
the cap by ``b`` and runs only up to the new cap, so an overshoot is absorbed
by the same process's next slice; an improvement row is VISIBLE only once
its schedule count is <= the cap (rows past it wait for the next slice).
Result at cap C = best visible row = offline ``value_at(C)`` exactly. The
only uncharged compute is the last overshoot of a process, never used.

Time: solver-only seconds measured by the solver (``s.elapsed``), excluding
the initial solution (solver convention); nothing between slices counts.
Path relinking (B3) is refused: it runs after the loop, no slice semantics.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

SOLVER_NAME = "brdeh"
BRIDGE = Path(__file__).with_name("brdeh_bridge.jl")
INT_MAX = 2**63 - 1

# Opaque params of this solver. fbi_max_passes None = loop until no improvement.
PARAM_KEYS = {"rules", "fbi_max_passes", "policy_keep_prob", "path_relink"}
RULE_SETS = {"resource7": False, "klein13": True}  # value: uses Klein policies

# Named points of the params space; each must match configParams() of the
# pinned solver commit (checked at Julia init, see _check_presets).
PRESETS = {
    "brdeh-A": {"rules": "resource7", "fbi_max_passes": None, "policy_keep_prob": 0.5},
    "brdeh-B1": {"rules": "resource7", "fbi_max_passes": 1, "policy_keep_prob": 0.8},
    "brdeh-B2": {"rules": "klein13", "fbi_max_passes": 1, "policy_keep_prob": 0.8},
}
_SOLVER_CONFIG_OF_PRESET = {"brdeh-A": "A", "brdeh-B1": "B1", "brdeh-B2": "B2"}

_JL = None

Row = tuple[int, int, int, float]  # iter, schedules (process-local), makespan, solver time_s


@dataclass
class Trajectory:
    """Visible improvement rows + terminal row (iter = -1, schedules = budget,
    makespan = best, time = measured solver seconds of the run)."""

    rows: list[Row]
    budget: int
    seed: int
    params: dict
    meta: dict = field(default_factory=dict)

    @property
    def makespan(self) -> int:
        return self.rows[-1][2]


def params_hash(params: dict) -> str:
    blob = json.dumps(params, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def validate_params(params: dict) -> dict:
    unknown = set(params) - PARAM_KEYS
    if unknown:
        raise ValueError(f"unknown BR-DEH params: {sorted(unknown)}")
    if params.get("rules") not in RULE_SETS:
        raise ValueError(f"rules must be one of {sorted(RULE_SETS)}: {params.get('rules')!r}")
    fbi = params.get("fbi_max_passes")
    if fbi is not None and (not isinstance(fbi, int) or fbi < 1):
        raise ValueError(f"fbi_max_passes must be None or an int >= 1: {fbi!r}")
    keep = params.get("policy_keep_prob")
    if not isinstance(keep, (int, float)) or not 0.0 <= keep <= 1.0:
        raise ValueError(f"policy_keep_prob must be in [0, 1]: {keep!r}")
    if params.get("path_relink"):
        raise NotImplementedError(
            "path_relink (B3) runs after the iteration loop and has no "
            "slice-consistent semantics (see solver experiments/process.jl)."
        )
    return params


def jl():
    """Lazy juliacall init: include the bridge, check presets, JIT warm-up."""
    global _JL
    if _JL is None:
        from juliacall import Main

        Main.include(str(BRIDGE))
        _check_presets(Main)
        if os.environ.get("BRDEH_SKIP_WARMUP") != "1":
            Main.ado_warmup()
        _JL = Main
    return _JL


def _check_presets(Main):
    for name, cfg in _SOLVER_CONFIG_OF_PRESET.items():
        p = Main.ado_config_params(cfg)
        fbi = None if int(p.fbiMaxPasses) >= INT_MAX else int(p.fbiMaxPasses)
        solver_view = {
            "rules": "klein13" if bool(p.klein) else "resource7",
            "fbi_max_passes": fbi,
            "policy_keep_prob": float(p.policyKeepProb),
        }
        if solver_view != PRESETS[name] or bool(p.pathRelink):
            raise RuntimeError(
                f"preset {name} drifted from configParams({cfg!r}) of the pinned "
                f"solver: harness {PRESETS[name]} vs solver {solver_view}"
            )


def environment() -> dict:
    info = jl().ado_runtime_info()
    return {"julia": str(info.julia), "julia_threads": int(info.nthreads),
            "julia_gc_threads": int(info.ngcthreads)}


def canonical_seeds() -> list[int]:
    """The 32 seeds hardcoded in the solver (src/heuristic.jl main())."""
    return [int(s) for s in jl().SEEDS]


class BrdehProcess:
    """One resumable BR-DEH search process (a Julia RunState)."""

    def __init__(self, instance: str, params: dict, seed: int):
        validate_params(params)
        self.instance, self.params, self.seed = instance, dict(params), int(seed)
        J = jl()
        fbi = params["fbi_max_passes"]
        self._J = J
        self._advance = J.seval("ado_advance!")  # Julia bang-name, not a Python attribute
        self._s = J.ado_start(instance, self.seed, RULE_SETS[params["rules"]],
                              INT_MAX if fbi is None else int(fbi),
                              float(params["policy_keep_prob"]))
        self.cap = 0            # cumulative budget given to this process
        self.rows: list[Row] = []      # visible rows (schedules <= cap)
        self._pending: list[Row] = []  # fetched rows past the cap
        self._fetched = 0
        self.schedules, self.solver_time = self._counters()[:2]

    def _counters(self):
        sched, elapsed, n = self._J.ado_counters(self._s)
        return int(sched), float(elapsed), int(n)

    def advance(self, budget: int) -> dict:
        """Raise the cap by ``budget`` and run up to it. Returns the rows that
        became visible plus the measured clocks and real spend of the call.

        Clocks (doctrine, artifact documentation "Eje de tiempo"): ``cpu_s`` = process CPU
        time around the solver call (charged; includes Julia GC threads and
        any work pushed to other threads), ``solver_time_s`` = the solver's
        own wall clock, ``wall_s`` = wall around the call (glue included),
        ``thread_cpu_s`` (diagnostic: process/thread ratio exposes offloaded
        work), ``gc_s`` and ``compile_s`` deltas of the Julia runtime."""
        if budget <= 0:
            raise ValueError(f"budget must be > 0: {budget}")
        sched0, time0 = self.schedules, self.solver_time
        self.cap += budget
        gc0, comp0 = self._J.ado_runtime_counters()
        c0, tc0, t0 = time.process_time(), time.thread_time(), time.perf_counter()
        self._advance(self._s, self.cap - self.schedules)
        wall = time.perf_counter() - t0
        cpu, tcpu = time.process_time() - c0, time.thread_time() - tc0
        gc1, comp1 = self._J.ado_runtime_counters()
        self.schedules, self.solver_time, n = self._counters()
        if n > self._fetched:
            m = self._J.ado_rows(self._s, self._fetched + 1)
            self._pending += [(int(m[k, 0]), int(m[k, 1]), int(m[k, 2]), float(m[k, 3]))
                              for k in range(m.shape[0])]
            self._fetched = n
        new = [r for r in self._pending if r[1] <= self.cap]
        self._pending = [r for r in self._pending if r[1] > self.cap]
        self.rows += new
        return {
            "rows": new,
            "solver_time_s": self.solver_time - time0,
            "cpu_s": cpu,
            "thread_cpu_s": tcpu,
            "gc_s": (int(gc1) - int(gc0)) / 1e9,
            "compile_s": (int(comp1) - int(comp0)) / 1e9,
            "executed_schedules": self.schedules - sched0,
            "wall_s": wall,
        }

    @property
    def best(self) -> int | None:
        return self.rows[-1][2] if self.rows else None


def open_process(instance: str, params: dict, seed: int) -> BrdehProcess:
    return BrdehProcess(instance, params, seed)


def solve(instance: str, params: dict, budget: int, seed: int) -> Trajectory:
    p = open_process(instance, params, seed)
    sl = p.advance(budget)
    if not p.rows:
        raise ValueError(f"budget {budget} smaller than the initial solution")
    return Trajectory(
        rows=p.rows + [(-1, budget, p.rows[-1][2], p.solver_time)],
        budget=budget,
        seed=int(seed),
        params=dict(params),
        meta={
            "solver": SOLVER_NAME,
            "params_hash": params_hash(params),
            "executed_schedules": p.schedules,
            "solver_time_s": p.solver_time,
            "cpu_s": sl["cpu_s"],
            "wall_s": sl["wall_s"],
        },
    )
