
import argparse
import csv
import json
from statistics import mean

from experiments.a1_uniform import _read_traj
from experiments.e1_learned import JL as JL_E1
from experiments.init13 import BUDGET, instances, value_at
from harness.instances import instance_features
from harness.provenance import REPO_ROOT
from harness.solvers.brdeh import canonical_seeds, jl

OUT = REPO_ROOT / "experiments" / "runs" / "pr_stagnation"
RUNS = REPO_ROOT / "experiments" / "runs"
SEEDS = [1, 2, 3]
STRIDE = 6
S_STALL, CAP, N_ORIG, K_ELITE = 500, 300, 4, 20

JL = r"""
_order(v::Vector{Int}) = sortperm(v)      # ids by start time (stable: ties by id)
function ado_prs_start(name::String, seed::Int, tieseed::Int, k::Int)
    mk = Int[]; n = 0
    e = Elite(k, Vector{Int}[], Int[], UInt64[])
    cd(ADO_SOLVER_ROOT) do
        for j in 1:13
            inp = readInputs(name)
            n = maximum(t.id for t in inp.tasks)
            r = Xoshiro(seed)
            pols, _ = buildPolicies(inp, r, "B2")
            s = startRun(inp, pols, r; fbiMaxPasses = 1, policyKeepProb = 0.8, initPolicy = pols[j])
            push!(mk, Int(s.bestSolution.makespan))
            _offer!(e, _snap(s.bestSolution, n), Int(s.bestSolution.makespan))
        end
    end
    m = minimum(mk)
    winners = findall(==(m), mk)
    w = winners[rand(Xoshiro(tieseed), 1:length(winners))]
    s = cd(ADO_SOLVER_ROOT) do
        inputs = readInputs(name)
        rng = Xoshiro(seed)
        pols, _ = buildPolicies(inputs, rng, "B2")
        startRun(inputs, pols, rng; fbiMaxPasses = 1, policyKeepProb = 0.8, initPolicy = pols[w])
    end
    @assert s.bestSolution.makespan == m
    s.schedules = 39
    s.traj[1] = (iter = 0, schedules = 39, makespan = m, time = 0.0)
    return s, w, e, n
end
function _relink!(s, e::Elite, n::Int, gbest::Ref{Int}, cap::Int, norig::Int, target::Int)
    idx = sortperm(e.mk)
    length(idx) < 2 && return (0, 0)
    guide = _order(e.st[idx[1]])
    ft = Vector{Int64}(undef, s.inputs.nTasks)
    used = 0; found = 0
    for o in idx[2:min(end, norig + 1)]
        order = _order(e.st[o])
        for pos in 1:length(order)
            (used >= cap || s.schedules >= target) && return (used, found)
            order[pos] == guide[pos] && continue
            src = findfirst(==(guide[pos]), order)
            order[pos], order[src] = order[src], order[pos]
            repairPrecedence!(order, s.inputs)
            c0 = SCHEDULE_COUNT[]
            mk = decodeOrder!(s.inputs, s.availability_units, order, ft)
            s.schedules += SCHEDULE_COUNT[] - c0
            used += 1
            st = zeros(Int, n)
            for t in s.inputs.tasks
                st[t.id] = ft[t.id] - t.duration
            end
            _offer!(e, st, mk)
            if mk < gbest[]
                gbest[] = mk; found += 1
                push!(s.traj, (iter = s.iter, schedules = s.schedules, makespan = Float64(mk), time = 0.0))
            end
        end
    end
    return (used, found)
end
function ado_prs_run!(s, target::Int, mode::Int, w::Int, e::Elite, n::Int,
                      S::Int, cap::Int, norig::Int, seed2::Int)
    gbest = Ref(Int(s.bestSolution.makespan))
    runbest = gbest[]; m0 = gbest[]
    since = 0; episodes = 0; prused = 0; prfound = 0; restarts = 0
    pols = s.sortPolicies
    while s.schedules < target
        c0 = SCHEDULE_COUNT[]
        beta = rand(s.rng, Float64)
        pol = rand(s.rng, Float64) < s.policyKeepProb ? s.bestPolicy : rand(s.rng, pols)
        sol = MPJ_Heuristic(s.inputs, s.availability_times, beta, pol, true, s.rng)
        sol = localSearch(sol, s.inputs, s.availability_units; maxPasses = s.fbiMaxPasses)
        spent = SCHEDULE_COUNT[] - c0
        s.schedules += spent
        s.iter += 1
        mk = Int(sol.makespan)
        mode == 1 && _offer!(e, _snap(sol, n), mk)
        if mk < runbest
            runbest = mk
            s.bestBeta = beta
            s.bestPolicy = pol
        end
        if mk < gbest[]
            gbest[] = mk; since = 0
            s.bestSolution = sol
            push!(s.traj, (iter = s.iter, schedules = s.schedules, makespan = Float64(mk), time = 0.0))
        else
            since += spent
        end
        if mode != 0 && since >= S && s.schedules < target
            episodes += 1
            if mode == 1
                u, f = _relink!(s, e, n, gbest, cap, norig, target)
                prused += u; prfound += f
                runbest = min(runbest, gbest[])
            else
                restarts += 1
                s.rng = Xoshiro(seed2 + 7919 * restarts)
                s.bestPolicy = pols[w]
                runbest = m0
            end
            since = 0
        end
    end
    return (s.iter, episodes, prused, prfound)
end
"""


def cmd_run(a):
    J = jl()
    J.seval(JL_E1)
    J.seval(JL)
    start, run = J.seval("ado_prs_start"), J.seval("ado_prs_run!")
    seeds = canonical_seeds()
    sub = instances("j120")[::STRIDE]
    if a.limit:
        sub = sub[:a.limit]
    mine = sub[a.shard[0]::a.shard[1]]
    mode = {"none": 0, "pr": 1, "restart": 2}[a.mode]
    OUT.mkdir(parents=True, exist_ok=True)
    tag = f"{a.mode}_shard{a.shard[0]}of{a.shard[1]}" + (f"_lim{a.limit}" if a.limit else "")
    with open(OUT / f"traj_{tag}.csv", "w", newline="") as ft, open(OUT / f"runs_{tag}.csv", "w", newline="") as fr:
        wt, wr = csv.writer(ft), csv.writer(fr)
        wt.writerow(["instance", "seed_idx", "variant", "iter", "schedules", "makespan"])
        wr.writerow(["instance", "seed_idx", "tries", "episodes", "pr_schedules", "pr_improvements"])
        for inst in mine:
            for si in SEEDS:
                s, w, e, n = start(inst, seeds[si - 1], 1_000_003 * si + 7, K_ELITE)
                st = run(s, BUDGET, mode, int(w), e, n, S_STALL, CAP, N_ORIG, 424_242 + 1009 * si)
                wr.writerow([inst, si] + [int(x) for x in st])
                m = J.ado_rows(s, 1)
                for k in range(m.shape[0]):
                    if int(m[k, 1]) <= BUDGET:
                        wt.writerow([inst, si, a.mode, int(m[k, 0]), int(m[k, 1]), int(m[k, 2])])
    print("done", tag)


def cmd_analyze(a):
    from scipy.stats import wilcoxon
    tr = {m: _read_traj(sorted(OUT.glob(f"traj_{m}_shard*of7.csv")), m) for m in ("pr", "restart")}
    tr["brdeh2026"] = _read_traj(sorted((RUNS / "variants").glob("traj_j120_s1-2-3_shard*.csv")), "init13_nobeta")
    insts = sorted({k[0] for k in tr["pr"]})
    ref = {i: instance_features(i)["lb_cpm"] for i in insts}
    val = lambda t, i, B: mean(value_at(t[(i, s)], B) for s in SEEDS)
    res = {"n_instances": len(insts), "seeds": SEEDS, "params": {"S": S_STALL, "cap": CAP, "origins": N_ORIG, "elite": K_ELITE},
           "gap": {}, "cmp": {}}
    for B in (1000, 2500, 5000):
        res["gap"][B] = {v: mean(100 * (val(t, i, B) - ref[i]) / ref[i] for i in insts) for v, t in tr.items()}
        res["cmp"][B] = {}
        for x, y in (("pr", "brdeh2026"), ("restart", "brdeh2026"), ("pr", "restart")):
            d = [100 * (val(tr[x], i, B) - val(tr[y], i, B)) / ref[i] for i in insts]
            nz = [t for t in d if t != 0]
            res["cmp"][B][f"{x} vs {y}"] = [round(mean(d), 3), sum(t < 0 for t in d), sum(t == 0 for t in d),
                                            sum(t > 0 for t in d), float(wilcoxon(nz).pvalue) if len(nz) > 10 else None]
    rows = []
    for p in sorted(OUT.glob("runs_pr_shard*of7.csv")):
        with open(p) as f:
            rows += list(csv.DictReader(f))
    tot = lambda k: sum(int(r[k]) for r in rows)
    res["pr_stats"] = {"runs": len(rows), "episodes_per_run": tot("episodes") / len(rows),
                       "pr_schedules_share": tot("pr_schedules") / (len(rows) * BUDGET),
                       "pr_improvements_per_run": tot("pr_improvements") / len(rows),
                       "runs_with_pr_improvement": mean(int(r["pr_improvements"]) > 0 for r in rows)}
    (OUT / "summary.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "analyze"])
    ap.add_argument("--mode", choices=["none", "pr", "restart"], default="pr")
    ap.add_argument("--shard", type=int, nargs=2, default=[0, 1])
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    {"run": cmd_run, "analyze": cmd_analyze}[a.cmd](a)
