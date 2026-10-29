
import argparse
import csv
import json
from statistics import mean

from experiments.a1_uniform import JL as JL_A1, _read_traj
from experiments.fixed_rule import JL as JL_FIXED
from experiments.init13 import BUDGET, archive_rows, instances, value_at
from experiments.variants import JL as JL_VAR
from harness.evaluate import load_optima
from harness.instances import instance_features
from harness.provenance import REPO_ROOT
from harness.solvers.brdeh import canonical_seeds, jl

OUT = REPO_ROOT / "experiments" / "runs" / "confirm"
RUNS = REPO_ROOT / "experiments" / "runs"
SEEDS = list(range(1, 11))


def cmd_run(a):
    J = jl()
    J.seval(JL_A1)
    J.seval(JL_VAR)
    J.seval(JL_FIXED)
    start = J.seval("ado_init13_start_mk")
    run_v = J.seval("ado_run_v!")
    run_f = J.seval("ado_run_fixed!")
    tovec = J.seval("x -> Int[x...]")
    seeds = canonical_seeds()
    mine = instances(a.bench)[a.shard[0]::a.shard[1]]
    OUT.mkdir(parents=True, exist_ok=True)
    tag = f"{a.bench}_s{'-'.join(map(str, a.seeds))}_shard{a.shard[0]}of{a.shard[1]}"
    with open(OUT / f"traj_{tag}.csv", "w", newline="") as ft:
        wt = csv.writer(ft)
        wt.writerow(["instance", "seed_idx", "variant", "iter", "schedules", "makespan"])
        for c, inst in enumerate(mine, 1):
            for si in a.seeds:
                for v in ("brdeh2026", "F1"):
                    s, w, nt, mk = start(inst, seeds[si - 1], 1_000_003 * si + 7)
                    if v == "brdeh2026":
                        run_v(s, BUDGET, True)
                    else:
                        run_f(s, BUDGET, 1, int(w), tovec([int(w)]))
                    m = J.ado_rows(s, 1)
                    for k in range(m.shape[0]):
                        if int(m[k, 1]) <= BUDGET:
                            wt.writerow([inst, si, v, int(m[k, 0]), int(m[k, 1]), int(m[k, 2])])
            if c % 20 == 0:
                print(f"{tag}: {c}/{len(mine)}", flush=True)
    print("done", tag)


def _load(bench):
    tr = {}
    paths = sorted(OUT.glob(f"traj_{bench}_*.csv"))
    tr["brdeh2026"] = _read_traj(paths, "brdeh2026")
    tr["F1"] = _read_traj(paths, "F1")
    if bench == "j120":  # seeds 1-3 from the exploratory runs (same code path)
        tr["brdeh2026"].update(_read_traj(sorted((RUNS / "variants").glob("traj_j120_s1-2-3_shard*.csv")), "init13_nobeta"))
        tr["F1"].update(_read_traj(sorted((RUNS / "fixed_rule").glob("traj_j120_s1-2-3_shard*.csv")), "F1"))
    insts = sorted({k[0] for k in tr["F1"]})
    tr["b2"] = archive_rows(bench, set(SEEDS), set(insts))
    return tr, insts


def cmd_analyze(a):
    import random
    from scipy.stats import wilcoxon
    res = {"seeds": SEEDS, "budget": BUDGET, "benches": {}}
    for bench in ("j30", "j60", "j120"):
        tr, insts = _load(bench)
        missing = [(v, i, s) for v in tr for i in insts for s in SEEDS if (i, s) not in tr[v]]
        assert not missing, f"{bench}: missing runs {missing[:5]} (+{len(missing)})"
        ref = load_optima() if bench == "j30" else {i: instance_features(i)["lb_cpm"] for i in insts}
        val = lambda t, i, B: mean(value_at(t[(i, s)], B) for s in SEEDS)
        r = {"n_instances": len(insts), "reference": "optimum" if bench == "j30" else "lb_cpm", "gap": {}, "cmp": {}}
        for B in (1000, 2500, 5000):
            r["gap"][B] = {v: mean(100 * (val(t, i, B) - ref[i]) / ref[i] for i in insts) for v, t in tr.items()}
            r["cmp"][B] = {}
            for x, y in (("brdeh2026", "b2"), ("F1", "b2"), ("F1", "brdeh2026")):
                d = [100 * (val(tr[x], i, B) - val(tr[y], i, B)) / ref[i] for i in insts]
                nz = [t for t in d if t != 0]
                rng = random.Random(12345)
                boots = sorted(mean(rng.choice(d) for _ in d) for _ in range(2000))
                r["cmp"][B][f"{x} vs {y}"] = {
                    "mean_diff_points": mean(d), "ci95": [boots[49], boots[1949]],
                    "x_better": sum(t < 0 for t in d), "tie": sum(t == 0 for t in d), "x_worse": sum(t > 0 for t in d),
                    "wilcoxon_p": float(wilcoxon(nz).pvalue) if len(nz) > 10 else None}
        res["benches"][bench] = r
    (OUT / "summary.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "analyze"])
    ap.add_argument("bench", nargs="?")
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    ap.add_argument("--shard", type=int, nargs=2, default=[0, 1])
    a = ap.parse_args()
    {"run": cmd_run, "analyze": cmd_analyze}[a.cmd](a)
