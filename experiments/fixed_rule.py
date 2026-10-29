
import argparse
import csv
import json
from collections import defaultdict
from statistics import mean

from experiments.a1_uniform import JL as JL_A1, _read_traj
from experiments.init13 import BUDGET, instances, value_at
from harness.instances import instance_features
from harness.provenance import REPO_ROOT
from harness.solvers.brdeh import canonical_seeds, jl

VARIANTS = ["F1", "F2", "F3"]
OUT = REPO_ROOT / "experiments" / "runs" / "fixed_rule"
BASE = REPO_ROOT / "experiments" / "runs" / "variants"

JL = r"""
function ado_run_fixed!(s, target::Int, mode::Int, w::Int, tied::Vector{Int})
    pols = s.sortPolicies
    while s.schedules < target
        c0 = SCHEDULE_COUNT[]
        beta = rand(s.rng, Float64)
        pol = if mode == 1
            pols[w]
        elseif mode == 2
            rand(s.rng, Float64) < 0.8 ? pols[w] : rand(s.rng, pols)
        else
            pols[tied[rand(s.rng, 1:length(tied))]]
        end
        sol = MPJ_Heuristic(s.inputs, s.availability_times, beta, pol, true, s.rng)
        sol = localSearch(sol, s.inputs, s.availability_units; maxPasses = s.fbiMaxPasses)
        s.schedules += SCHEDULE_COUNT[] - c0
        s.iter += 1
        if sol.makespan < s.bestSolution.makespan
            s.bestBeta = beta
            s.bestSolution = sol
            push!(s.traj, (iter = s.iter, schedules = s.schedules,
                           makespan = sol.makespan, time = 0.0))
        end
    end
    return nothing
end
"""


def cmd_run(a):
    J = jl()
    J.seval(JL_A1)
    J.seval(JL)
    start = J.seval("ado_init13_start_mk")
    run = J.seval("ado_run_fixed!")
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
                for mode, v in enumerate(VARIANTS, 1):
                    s, w, nt, mk = start(inst, seeds[si - 1], 1_000_003 * si + 7)
                    best = min(mk)
                    tied = tovec([k + 1 for k, x in enumerate(mk) if x == best])
                    run(s, BUDGET, mode, int(w), tied)
                    m = J.ado_rows(s, 1)
                    for k in range(m.shape[0]):
                        if int(m[k, 1]) <= BUDGET:
                            wt.writerow([inst, si, v, int(m[k, 0]), int(m[k, 1]), int(m[k, 2])])
            if c % 25 == 0:
                print(f"{tag}: {c}/{len(mine)}", flush=True)
    print("done", tag)


def cmd_analyze(a):
    from scipy.stats import wilcoxon
    tag = f"{a.bench}_s{'-'.join(map(str, a.seeds))}"
    paths = sorted(OUT.glob(f"traj_{tag}_shard*.csv"))
    tr = {v: _read_traj(paths, v) for v in VARIANTS}
    bpaths = sorted(BASE.glob(f"traj_{tag}_shard*.csv"))
    tr["base"] = _read_traj(bpaths, "init13_nobeta")
    tr["b2"] = _read_traj(bpaths, "b2")
    insts = sorted({k[0] for k in tr["F1"]})
    assert all(tr[v][(i, s)][0] == tr["base"][(i, s)][0] for v in VARIANTS for i in insts for s in a.seeds)
    ref = {i: instance_features(i)["lb_cpm"] for i in insts}
    val = lambda t, i, B: mean(value_at(t[(i, s)], B) for s in a.seeds)
    res = {"bench": a.bench, "seeds": a.seeds, "n_instances": len(insts), "gap": {}, "cmp": {}}
    for B in (250, 1000, 2500, 5000):
        res["gap"][B] = {n: mean(100 * (val(t, i, B) - ref[i]) / ref[i] for i in insts) for n, t in tr.items()}

    def cmp(x, y, B):
        d = [val(tr[x], i, B) - val(tr[y], i, B) for i in insts]
        nz = [t for t in d if t != 0]
        return [sum(t < 0 for t in d), sum(t == 0 for t in d), sum(t > 0 for t in d),
                float(wilcoxon(nz).pvalue) if len(nz) > 10 else None]
    pairs = [("F1", "base"), ("F2", "base"), ("F3", "base"), ("F1", "F2"), ("F3", "F1"), ("F3", "F2")]
    for B in (1000, 2500, 5000):
        res["cmp"][B] = {f"{x} vs {y} (x better/tie/y better, p)": cmp(x, y, B) for x, y in pairs}
    (OUT / f"summary_{tag}.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "analyze"])
    ap.add_argument("bench")
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--shard", type=int, nargs=2, default=[0, 1])
    a = ap.parse_args()
    {"run": cmd_run, "analyze": cmd_analyze}[a.cmd](a)
