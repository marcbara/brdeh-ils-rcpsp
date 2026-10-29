
import argparse
import csv
import json
from collections import defaultdict
from statistics import mean

from experiments.init13 import BUDGET, JL as JL_INIT13, archive_rows, instances, value_at
from harness.evaluate import load_optima
from harness.instances import instance_features
from harness.provenance import REPO_ROOT
from harness.solvers.brdeh import canonical_seeds, jl

VARIANTS = ["b2", "nobeta", "init13", "init13_nobeta"]
OUT = REPO_ROOT / "experiments" / "runs" / "variants"

JL = r"""
function _ado_hash(sol)
    hash(sort([(t.id, t.start_time) for t in sol.tasks]))
end
function ado_iter_v!(s, uniformBeta::Bool, seenB, seenF, dup)
    c0 = SCHEDULE_COUNT[]
    beta = uniformBeta ? rand(s.rng, Float64) : rand(s.rng, TriangularDist(0.0, 1.0, s.bestBeta))
    sortPolicy = rand(s.rng, Float64) < s.policyKeepProb ? s.bestPolicy : rand(s.rng, s.sortPolicies)
    solution = MPJ_Heuristic(s.inputs, s.availability_times, beta, sortPolicy, true, s.rng)
    hb = _ado_hash(solution); dup[1] += (hb in seenB); push!(seenB, hb)
    solution = localSearch(solution, s.inputs, s.availability_units; maxPasses = s.fbiMaxPasses)
    hf = _ado_hash(solution); dup[2] += (hf in seenF); push!(seenF, hf)
    s.schedules += SCHEDULE_COUNT[] - c0
    improved = solution.makespan < s.bestSolution.makespan
    if improved
        s.bestBeta = beta
        s.bestPolicy = sortPolicy
        s.bestSolution = solution
    end
    s.iter += 1
    return improved
end
function ado_run_v!(s, target::Int, uniformBeta::Bool)
    seenB = Set{UInt64}(); seenF = Set{UInt64}(); dup = [0, 0]; n = 0
    while s.schedules < target
        n += 1
        if ado_iter_v!(s, uniformBeta, seenB, seenF, dup)
            push!(s.traj, (iter = s.iter, schedules = s.schedules,
                           makespan = s.bestSolution.makespan, time = 0.0))
        end
    end
    return (n, dup[1], dup[2])
end
"""


def rows_of(J, s):
    m = J.ado_rows(s, 1)
    return [(int(m[k, 0]), int(m[k, 1]), int(m[k, 2])) for k in range(m.shape[0])]


def cmd_run(a):
    J = jl()
    J.seval(JL_INIT13)
    J.seval(JL)
    init13 = J.seval("ado_init13_start")
    run = J.seval("ado_run_v!")
    seeds = canonical_seeds()
    mine = instances(a.bench)[a.shard[0]::a.shard[1]]
    OUT.mkdir(parents=True, exist_ok=True)
    tag = f"{a.bench}_s{'-'.join(map(str, a.seeds))}_shard{a.shard[0]}of{a.shard[1]}"
    with open(OUT / f"traj_{tag}.csv", "w", newline="") as ft, open(OUT / f"runs_{tag}.csv", "w", newline="") as fr:
        wt, wr = csv.writer(ft), csv.writer(fr)
        wt.writerow(["instance", "seed_idx", "variant", "iter", "schedules", "makespan"])
        wr.writerow(["instance", "seed_idx", "variant", "tries", "dup_built", "dup_final", "init_rule", "n_tied"])
        for k, inst in enumerate(mine, 1):
            for si in a.seeds:
                for v in VARIANTS:
                    rule, nt = "SAD", 0
                    if v.startswith("init13"):
                        s, w, nt = init13(inst, seeds[si - 1], 1_000_003 * si + 7)
                        rule = int(w)
                    else:
                        s = J.ado_start(inst, seeds[si - 1], True, 1, 0.8)
                    n, db, df = run(s, BUDGET, v.endswith("nobeta"))
                    for it, sch, mk in rows_of(J, s):
                        if sch <= BUDGET:
                            wt.writerow([inst, si, v, it, sch, mk])
                    wr.writerow([inst, si, v, int(n), int(db), int(df), rule, int(nt)])
            if k % 25 == 0:
                print(f"{tag}: {k}/{len(mine)}", flush=True)
    print("done", tag)


def cmd_analyze(a):
    from scipy.stats import wilcoxon
    tag = f"{a.bench}_s{'-'.join(map(str, a.seeds))}"
    tr, runs = defaultdict(list), []
    for p in sorted(OUT.glob(f"traj_{tag}_shard*.csv")):
        with open(p) as f:
            for r in csv.DictReader(f):
                tr[(r["instance"], int(r["seed_idx"]), r["variant"])].append(
                    (int(r["iter"]), int(r["schedules"]), int(r["makespan"])))
    for p in sorted(OUT.glob(f"runs_{tag}_shard*.csv")):
        with open(p) as f:
            runs += list(csv.DictReader(f))
    insts = sorted({k[0] for k in tr})
    arch = archive_rows(a.bench, set(a.seeds), set(insts))
    # replica check: b2 must equal the archive row by row
    bad = [(i, s) for i in insts for s in a.seeds if tr[(i, s, "b2")] != arch[(i, s)]]
    assert not bad, f"b2 replica differs from archive: {bad[:5]}"
    # init13 here must equal the stage-1 run (same start, same loop)
    s1 = REPO_ROOT / "experiments" / "runs" / "init13"
    st1 = defaultdict(list)
    for p in sorted(s1.glob(f"init13_{tag}_shard*.csv")):
        with open(p) as f:
            for r in csv.DictReader(f):
                st1[(r["instance"], int(r["seed_idx"]))].append((int(r["iter"]), int(r["schedules"]), int(r["makespan"])))
    if st1:
        bad1 = [(i, s) for i in insts for s in a.seeds if tr[(i, s, "init13")] != st1[(i, s)]]
        assert not bad1, f"init13 differs from stage 1: {bad1[:5]}"
    ref = load_optima() if a.bench == "j30" else {i: instance_features(i)["lb_cpm"] for i in insts}
    gap = lambda mk, i: 100.0 * (mk - ref[i]) / ref[i]
    val = lambda v, i, B: mean(value_at(tr[(i, s, v)], B) for s in a.seeds)
    res = {"bench": a.bench, "seeds": a.seeds, "n_instances": len(insts),
           "checks": {"b2_replica_equals_archive": True, "init13_equals_stage1": bool(st1)},
           "gap": {}, "vs_b2": {}, "pairs": {}}
    for B in (39, 250, 1000, 2500, 5000):
        res["gap"][B] = {v: mean(gap(val(v, i, B), i) for i in insts) for v in VARIANTS}

    def cmp(x, y, B):
        d = [val(x, i, B) - val(y, i, B) for i in insts]
        nz = [t for t in d if t != 0]
        return {"x_better": sum(t < 0 for t in d), "tie": sum(t == 0 for t in d), "y_better": sum(t > 0 for t in d),
                "wilcoxon_p": float(wilcoxon(nz).pvalue) if len(nz) > 10 else None}
    for B in (1000, 5000):
        res["vs_b2"][B] = {v: cmp(v, "b2", B) for v in VARIANTS[1:]}
        res["pairs"][B] = {"init13_nobeta vs init13": cmp("init13_nobeta", "init13", B)}
    dup = {}
    for v in VARIANTS:
        rr = [r for r in runs if r["variant"] == v]
        t = sum(int(r["tries"]) for r in rr)
        dup[v] = {"tries": t, "dup_built_pct": 100 * sum(int(r["dup_built"]) for r in rr) / t,
                  "dup_final_pct": 100 * sum(int(r["dup_final"]) for r in rr) / t}
    res["duplicates"] = dup
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
