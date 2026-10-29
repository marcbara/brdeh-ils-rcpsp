"""PSPLIB instance data for the harness: parser, Kolisch parameters, bounds.

Self-contained on purpose (level-3 common data, artifact documentation): the harness
needs instance features as policy CONTEXT and a lower bound for rewards that
must not peek at the optima. Same fixed-offset .sm layout as the community
parsers (and as the solver's readInputs).

Features (Kolisch, Sprecher & Drexl 1995 definitions, computed from the file):

* ``nc``: network complexity = precedence arcs / activities, dummies included
  (the generator's definition: j30 sets 1.5 / 1.8 / 2.1).
* ``rf``: resource factor = mean over non-dummy jobs of the fraction of
  resources they use (j30: 0.25 / 0.5 / 0.75 / 1.0).
* ``rs``: resource strength = mean over resources of
  (K - rmin) / (rmax - rmin), rmin = largest single-job demand, rmax = peak
  demand of the earliest-start (precedence-only) schedule; the generator
  targets 0.2 / 0.5 / 0.7 / 1.0 (realized values differ slightly).
* ``lb_cpm``: critical-path length (precedence-only lower bound).
* ``n_jobs``, ``n_resources``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path

from .provenance import REPO_ROOT

INPUTS = REPO_ROOT / "solvers" / "br-deh" / "inputs"


@dataclass(frozen=True)
class SmInstance:
    name: str
    durations: tuple[int, ...]
    successors: tuple[tuple[int, ...], ...]   # 0-based
    demands: tuple[tuple[int, ...], ...]
    capacities: tuple[int, ...]


def parse_sm(path: Path, name: str | None = None) -> SmInstance:
    lines = Path(path).read_text().splitlines()
    n = int(lines[5].split()[-1])
    k = int(lines[8].split()[-2])
    succ = []
    for i in range(n):
        parts = [int(x) for x in lines[18 + i].split()]
        succ.append(tuple(s - 1 for s in parts[3:]))
    base = 18 + n + 4
    dur, dem = [], []
    for i in range(n):
        parts = [int(x) for x in lines[base + i].split()]
        dur.append(parts[2])
        dem.append(tuple(parts[3:3 + k]))
    caps = tuple(int(x) for x in lines[base + n + 3].split()[:k])
    return SmInstance(name or Path(path).name, tuple(dur), tuple(succ), tuple(dem), caps)


def earliest_starts(inst: SmInstance) -> list[int]:
    """Forward pass; PSPLIB numbers jobs topologically (checked)."""
    es = [0] * len(inst.durations)
    for i, ss in enumerate(inst.successors):
        for j in ss:
            if j <= i:
                raise ValueError(f"{inst.name}: not topologically numbered ({i}->{j})")
            es[j] = max(es[j], es[i] + inst.durations[i])
    return es


@dataclass(frozen=True)
class Features:
    nc: float
    rf: float
    rs: float
    lb_cpm: int
    n_jobs: int
    n_resources: int

    def as_dict(self) -> dict:
        return asdict(self)


def features(inst: SmInstance) -> Features:
    n, k = len(inst.durations), len(inst.capacities)
    arcs = sum(len(s) for s in inst.successors)
    real = [i for i in range(n) if inst.durations[i] > 0]
    rf = sum(sum(1 for r in range(k) if inst.demands[i][r] > 0) / k for i in real) / len(real)

    es = earliest_starts(inst)
    lb = max(es[i] + inst.durations[i] for i in range(n))
    rs_vals = []
    for r in range(k):
        rmin = max(inst.demands[i][r] for i in range(n))
        profile = [0] * (lb + 1)
        for i in real:
            for t in range(es[i], es[i] + inst.durations[i]):
                profile[t] += inst.demands[i][r]
        rmax = max(profile)
        # resource unused by every job: strength undefined -> fully slack
        rs_vals.append(1.0 if rmax == rmin else (inst.capacities[r] - rmin) / (rmax - rmin))
    return Features(nc=arcs / n, rf=rf, rs=sum(rs_vals) / k, lb_cpm=lb, n_jobs=n, n_resources=k)


@lru_cache(maxsize=None)
def instance_features(instance: str) -> dict:
    """Features of a solver instance name (``"j30/j301_1.sm"``), cached."""
    return features(parse_sm(INPUTS / instance, instance)).as_dict()


def psplib_group(instance: str) -> int:
    """Parameter-combination index of a PSPLIB name (j30<g>_<i>.sm -> g)."""
    stem = Path(instance).stem          # j3017_4
    family_digits = len(Path(instance).parent.name) - 1   # "j30" -> 2
    return int(stem.split("_")[0][1 + family_digits:])
