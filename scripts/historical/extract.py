#!/usr/bin/env python3
"""Read-only extraction of every number used in the manuscript.

Reads the unpacked campaign artifacts under experiments/runs/ (and the archived
BR-DEH 2024 trajectories); writes only into HISTORICAL_OUT.
No solver is executed.

    python scripts/reproduce_historical.py

Conventions (identical to the campaign analyzers):
  value_at(traj, B) = best makespan among trajectory rows with schedules <= B
  gap_i = 100 * (mean_s value_at - ref_i) / ref_i, ref = optimum (J30) or CPM bound (J60/J120)
  set mean = mean over instances of the per-instance gap (seeds averaged first)
  best-of-10 = per instance min over seeds of value_at(5000), then gap, then mean
  paired comparison = per-instance difference of gaps (x - y), bootstrap CI95 (10,000 resamples,
  numpy seed 20261028), two-sided Wilcoxon signed-rank (scipy, zero differences dropped),
  better/tie/worse counted on the per-instance mean makespan.
"""

import csv
import gzip
import hashlib
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, median

import numpy as np
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from harness.evaluate import load_optima  # noqa: E402
from harness.instances import instance_features  # noqa: E402

HIST = Path(os.environ["HISTORICAL_RAW_ROOT"])
RUNS = HIST / "experiments" / "runs"
OUT = Path(os.environ["HISTORICAL_OUT"])
OUT.mkdir(exist_ok=True)
SEEDS = list(range(1, 11))
BUDGETS = (1000, 2500, 5000)
BENCHES = ("j30", "j60", "j120")
ARCH = {"j30": HIST / "solvers/br-deh/experiments/results/full_B2.csv.gz",
        "j60": HIST / "experiments/archive/j60_B2_50000.csv.gz",
        "j120": HIST / "experiments/archive/j120_B2_50000.csv.gz"}
HASHES = {}


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def read_traj(paths, variants, seeds=SEEDS, maxsched=5000):
    tr = {v: defaultdict(list) for v in variants}
    for p in paths:
        HASHES[str(Path(p).relative_to(HIST))] = sha(p)
        with open(p) as f:
            for r in csv.DictReader(f):
                v = r["variant"]
                s = int(r["seed_idx"])
                if v in tr and s in seeds and int(r["schedules"]) <= maxsched:
                    tr[v][(r["instance"], s)].append((int(r["iter"]), int(r["schedules"]), int(r["makespan"])))
    return tr


def read_archive(bench, seeds=SEEDS, maxsched=50000):
    p = ARCH[bench]
    HASHES[str(p.relative_to(HIST if p.is_relative_to(HIST) else ROOT))] = sha(p)
    out = defaultdict(list)
    with gzip.open(p, "rt") as f:
        for r in csv.DictReader(f):
            s = int(r["seed_idx"])
            if s in seeds and int(r["schedules"]) <= maxsched:
                out[(r["instance"], s)].append((int(r["iter"]), int(r["schedules"]), int(float(r["makespan"]))))
    return out


def value_at(rows, B):
    return min(mk for _, s, mk in rows if s <= B)


OPT = load_optima()


def refs(insts):
    return {i: (OPT[i] if i.startswith("j30") else instance_features(i)["lb_cpm"]) for i in insts}


def inst_gap(t, i, B, ref, seeds=SEEDS):
    return 100 * (mean(value_at(t[(i, s)], B) for s in seeds) - ref[i]) / ref[i]


def set_gap(t, insts, B, ref, seeds=SEEDS):
    return mean(inst_gap(t, i, B, ref, seeds) for i in insts)


def best_gap(t, insts, B, ref, seeds=SEEDS):
    return mean(100 * (min(value_at(t[(i, s)], B) for s in seeds) - ref[i]) / ref[i] for i in insts)


RNG = np.random.default_rng(20261028)


def paired(tx, ty, insts, B, ref, seeds=SEEDS):
    d = np.array([inst_gap(tx, i, B, ref, seeds) - inst_gap(ty, i, B, ref, seeds) for i in insts])
    mx = [mean(value_at(tx[(i, s)], B) for s in seeds) for i in insts]
    my = [mean(value_at(ty[(i, s)], B) for s in seeds) for i in insts]
    btw = [sum(a < b - 1e-12 for a, b in zip(mx, my)), sum(abs(a - b) <= 1e-12 for a, b in zip(mx, my)),
           sum(a > b + 1e-12 for a, b in zip(mx, my))]
    bs = RNG.choice(d, size=(10000, len(d)), replace=True).mean(axis=1)
    p = float(wilcoxon(d).pvalue) if np.any(d != 0) else 1.0
    return {"mean": float(d.mean()), "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
            "wilcoxon_p": p, "better_tie_worse": btw, "n": len(insts)}


def holm(ps):
    order = sorted(range(len(ps)), key=lambda k: ps[k])
    adj = [0.0] * len(ps)
    run = 0.0
    for r, k in enumerate(order):
        run = max(run, min(1.0, (len(ps) - r) * ps[k]))
        adj[k] = run
    return adj


def at_bound(t, insts, B, ref, seeds=SEEDS):
    return sum(value_at(t[(i, s)], B) == ref[i] for i in insts for s in seeds)


def stagnation(t, insts, ref, B=5000, seeds=SEEDS):
    fin = {(i, s): value_at(t[(i, s)], B) for i in insts for s in seeds}
    open_runs = [k for k in fin if fin[k] != ref[k[0]]]
    # the first row reaching the final value is the last improvement
    last = []
    for k in open_runs:
        rows = sorted(t[k], key=lambda r: r[1])
        first_hit = min(s for _, s, mk in rows if s <= B and mk == fin[k])
        last.append(first_hit / B)
    return {"runs": len(fin), "share_at_proven_bound": 1 - len(open_runs) / len(fin),
            "median_last_improvement": median(last) if last else None,
            "budget_without_gain": 1 - median(last) if last else None}


def main():
    res = {"conventions": __doc__, "benches": {}}
    profile_budgets = [100, 200, 300, 500, 750, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000]
    for b in BENCHES:
        cb = read_traj(sorted((RUNS / "confirm_bidir").glob(f"traj_{b}_shard*of*.csv")),
                       ["base", "base_ls", "winner", "winner_ls", "fwd78", "back78", "loser", "mix"])
        esc = read_traj(sorted((RUNS / "escoto_ideas").glob(f"traj_{b}v5_shard*of*.csv")), ["I_hyb"])
        ils = read_traj(sorted((RUNS / "br_ils").glob(f"traj_{b}full_shard*of*.csv")), ["br_r10_l5"])
        arch = read_archive(b)
        V = {"V0": arch, "V1": cb["base"], "V2": cb["base_ls"], "V3": cb["winner"], "V4": cb["winner_ls"],
             "V5": esc["I_hyb"], "V6": ils["br_r10_l5"]}
        insts = sorted({k[0] for k in V["V6"]})
        for name, t in V.items():
            miss = [(i, s) for i in insts for s in SEEDS if (i, s) not in t]
            assert not miss, (b, name, miss[:3], len(miss))
        ref = refs(insts)
        r = {"n_instances": len(insts), "reference": "optimum" if b == "j30" else "lb_cpm", "versions": {},
             "steps": {}, "profile": {}}
        for name, t in V.items():
            r["versions"][name] = {
                "mean": {B: set_gap(t, insts, B, ref) for B in BUDGETS},
                "best10_5000": best_gap(t, insts, 5000, ref),
                "runs_at_reference_5000": at_bound(t, insts, 5000, ref),
                "stagnation": stagnation(t, insts, ref)}
            r["profile"][name] = {B: set_gap(t, insts, B, ref) for B in profile_budgets}
        # V0 at 50,000 schedules (archive runs to 50k schedules or more)
        r["V0_50000"] = {"mean": set_gap(arch, insts, 50000, ref), "best10": best_gap(arch, insts, 50000, ref)}
        steps = [("V1", "V0"), ("V2", "V1"), ("V3", "V1"), ("V4", "V1"), ("V5", "V4"), ("V6", "V5"), ("V6", "V0"),
                 ("V5", "V0")]
        for B in BUDGETS:
            c = {f"{x} vs {y}": paired(V[x], V[y], insts, B, ref) for x, y in steps}
            for (k, v), ph in zip(c.items(), holm([v["wilcoxon_p"] for v in c.values()])):
                v["p_holm_family"] = ph
            r["steps"][B] = c
        res["benches"][b] = r
        print(b, {k: round(v["mean"][5000], 3) for k, v in r["versions"].items()}, flush=True)
    (OUT / "main.json").write_text(json.dumps(res, indent=1))
    # profile csv for pgfplots
    for b in BENCHES:
        prof = res["benches"][b]["profile"]
        with open(OUT / f"profile_{b}.csv", "w", newline="") as f:
            w = csv.writer(f, delimiter=" ")
            w.writerow(["B"] + list(prof))
            for B in profile_budgets:
                w.writerow([B] + [round(prof[v][B], 4) for v in prof])
    (OUT / "source_hashes_main.json").write_text(json.dumps(HASHES, indent=1, sort_keys=True))


if __name__ == "__main__":
    main()
