
import argparse
import csv
import json
from statistics import mean

from experiments.a1_uniform import _read_traj
from experiments.e1_learned import JL as JL_E1
from experiments.init13 import BUDGET, instances, value_at
from experiments.pr_stagnation import CAP, JL as JL_PRS, K_ELITE, N_ORIG, S_STALL, SEEDS, STRIDE
from harness.instances import instance_features
from harness.provenance import REPO_ROOT
from harness.solvers.brdeh import canonical_seeds, jl

OUT = REPO_ROOT / "experiments" / "runs" / "pr2_stagnation"
RUNS = REPO_ROOT / "experiments" / "runs"

JL = r"""
function _sol_from_ft(inputs, ft, mk)
    for t in inputs.tasks
        t.finish_time = ft[t.id]
        t.start_time = ft[t.id] - t.duration
    end
    sol = Solution()
    sol.tasks = sort(inputs.tasks, by = t -> (t.start_time, t.id))
    sol.makespan = Float64(mk)
    return sol
end
# FBI on a decoded order; returns (makespan, start-time snapshot) of the kept schedule
function _fbi_ft!(s, ft, mk, n)
    st0 = zeros(Int, n)
    for t in s.inputs.tasks
        st0[t.id] = ft[t.id] - t.duration
    end
    sol = _sol_from_ft(s.inputs, ft, mk)
    c0 = SCHEDULE_COUNT[]
    sol2 = localSearch(sol, s.inputs, s.availability_units; maxPasses = s.fbiMaxPasses)
    s.schedules += SCHEDULE_COUNT[] - c0
    if sol2.makespan < mk
        return Int(sol2.makespan), _snap(sol2, n)
    end
    return Int(mk), st0
end
function _improve!(s, gbest, e, st, mk)
    _offer!(e, st, mk)
    if mk < gbest[]
        gbest[] = mk
        push!(s.traj, (iter = s.iter, schedules = s.schedules, makespan = Float64(mk), time = 0.0))
        return 1
    end
    return 0
end
function _relink_fbi!(s, e::Elite, n::Int, gbest::Ref{Int}, cap::Int, norig::Int, target::Int)
    idx = sortperm(e.mk)
    length(idx) < 2 && return (0, 0)
    guide = _order(e.st[idx[1]])
    ft = Vector{Int64}(undef, s.inputs.nTasks)
    c_start = s.schedules; found = 0
    for o in idx[2:min(end, norig + 1)]
        order = _order(e.st[o])
        bestmk = typemax(Int); bestft = copy(ft)
        for pos in 1:length(order)
            (s.schedules - c_start >= cap || s.schedules >= target) && break
            order[pos] == guide[pos] && continue
            src = findfirst(==(guide[pos]), order)
            order[pos], order[src] = order[src], order[pos]
            repairPrecedence!(order, s.inputs)
            c0 = SCHEDULE_COUNT[]
            mk = decodeOrder!(s.inputs, s.availability_units, order, ft)
            s.schedules += SCHEDULE_COUNT[] - c0
            if mk < bestmk
                bestmk = mk; bestft .= ft
            end
        end
        bestmk == typemax(Int) && continue
        @assert _order_ok(order, s.inputs)
        s.schedules + 2 > target && break
        mk2, st2 = _fbi_ft!(s, bestft, bestmk, n)
        found += _improve!(s, gbest, e, st2, mk2)
        (s.schedules - c_start >= cap || s.schedules >= target) && break
    end
    return (s.schedules - c_start, found)
end
function _order_ok(order, inputs)
    where = zeros(Int, length(order))
    for (p, id) in enumerate(order)
        where[id] = p
    end
    for t in inputs.tasks, (pid, _) in t.predecessors
        where[pid] < where[t.id] || return false
    end
    return true
end
function _window(order, pos, inputs, where)
    t = inputs.tasks[order[pos]]
    lo = 1; hi = length(order)
    for (pid, _) in t.predecessors
        lo = max(lo, where[pid] + 1)
    end
    for (sid, _) in t.successors
        hi = min(hi, where[sid] - 1)
    end
    return lo, hi
end
function _local_search!(s, e::Elite, n::Int, gbest::Ref{Int}, cap::Int, target::Int)
    idx = sortperm(e.mk)
    cur = _order(e.st[idx[1]]); curmk = e.mk[idx[1]]
    ft = Vector{Int64}(undef, s.inputs.nTasks)
    c_start = s.schedules; found = 0
    L = length(cur)
    @assert decodeOrder!(s.inputs, s.availability_units, cur, ft) <= curmk   # known answer, not counted
    while s.schedules - c_start < cap && s.schedules + 3 <= target
        where = zeros(Int, L)
        for (p, id) in enumerate(cur)
            where[id] = p
        end
        pos = rand(s.rng, 2:L-1)
        lo, hi = _window(cur, pos, s.inputs, where)
        lo == hi && continue
        newpos = rand(s.rng, lo:hi)
        newpos == pos && continue
        cand = copy(cur)
        id = cand[pos]; deleteat!(cand, pos); insert!(cand, newpos, id)
        c0 = SCHEDULE_COUNT[]
        mk = decodeOrder!(s.inputs, s.availability_units, cand, ft)
        s.schedules += SCHEDULE_COUNT[] - c0
        mk <= curmk || continue
        @assert _order_ok(cand, s.inputs)
        mk2, st2 = _fbi_ft!(s, ft, mk, n)
        if mk2 <= curmk
            cur = _order(st2); curmk = mk2
            found += _improve!(s, gbest, e, st2, mk2)
        end
    end
    return (s.schedules - c_start, found)
end
function ado_prs2_run!(s, target::Int, mode::Int, e::Elite, n::Int, S::Int, cap::Int, norig::Int)
    gbest = Ref(Int(s.bestSolution.makespan))
    runbest = gbest[]
    since = 0; tsince = 0; episodes = 0; used = 0; found = 0
    pols = s.sortPolicies
    while s.schedules < target
        c0 = SCHEDULE_COUNT[]
        beta = rand(s.rng, Float64)
        pol = rand(s.rng, Float64) < s.policyKeepProb ? s.bestPolicy : rand(s.rng, pols)
        sol = MPJ_Heuristic(s.inputs, s.availability_times, beta, pol, true, s.rng)
        st0 = _snap(sol, n); mk0 = Int(sol.makespan)
        sol = localSearch(sol, s.inputs, s.availability_units; maxPasses = s.fbiMaxPasses)
        spent = SCHEDULE_COUNT[] - c0
        s.schedules += spent
        tsince += spent
        s.iter += 1
        mk = Int(sol.makespan)
        _offer!(e, mk < mk0 ? _snap(sol, n) : st0, mk)
        if mk < runbest
            runbest = mk; s.bestBeta = beta; s.bestPolicy = pol
        end
        if mk < gbest[]
            gbest[] = mk; since = 0
            s.bestSolution = sol
            push!(s.traj, (iter = s.iter, schedules = s.schedules, makespan = Float64(mk), time = 0.0))
        else
            since += spent
        end
        # mode 5: local search every S schedules of tries, whatever happens (attribution control)
        if (mode == 5 ? tsince >= S : since >= S) && s.schedules < target
            episodes += 1
            u, f = mode == 3 ? _relink_fbi!(s, e, n, gbest, cap, norig, target) :
                               _local_search!(s, e, n, gbest, cap, target)
            used += u; found += f
            runbest = min(runbest, gbest[])
            since = 0; tsince = 0
        end
    end
    @assert s.schedules <= target + 2
    @assert minimum(e.mk) == gbest[]
    return (s.iter, episodes, used, found)
end
"""


def cmd_run(a):
    J = jl()
    J.seval(JL_E1)
    J.seval(JL_PRS)
    J.seval(JL)
    start, run = J.seval("ado_prs_start"), J.seval("ado_prs2_run!")
    seeds = canonical_seeds()
    sub = instances("j120")[::STRIDE]
    if a.limit:
        sub = sub[:a.limit]
    mine = sub[a.shard[0]::a.shard[1]]
    mode = {"prfbi": 3, "ls": 4}[a.mode]
    OUT.mkdir(parents=True, exist_ok=True)
    tag = f"{a.mode}_shard{a.shard[0]}of{a.shard[1]}" + (f"_lim{a.limit}" if a.limit else "")
    with open(OUT / f"traj_{tag}.csv", "w", newline="") as ft, open(OUT / f"runs_{tag}.csv", "w", newline="") as fr:
        wt, wr = csv.writer(ft), csv.writer(fr)
        wt.writerow(["instance", "seed_idx", "variant", "iter", "schedules", "makespan"])
        wr.writerow(["instance", "seed_idx", "tries", "episodes", "shake_schedules", "shake_improvements"])
        for inst in mine:
            for si in SEEDS:
                s, w, e, n = start(inst, seeds[si - 1], 1_000_003 * si + 7, K_ELITE)
                st = run(s, BUDGET, mode, e, n, S_STALL, CAP, N_ORIG)
                wr.writerow([inst, si] + [int(x) for x in st])
                m = J.ado_rows(s, 1)
                prev = None
                for k in range(m.shape[0]):
                    if int(m[k, 1]) <= BUDGET:
                        wt.writerow([inst, si, a.mode, int(m[k, 0]), int(m[k, 1]), int(m[k, 2])])
    print("done", tag)


def cmd_analyze(a):
    from scipy.stats import wilcoxon
    tr = {m: _read_traj(sorted(OUT.glob(f"traj_{m}_shard*of7.csv")), m) for m in ("prfbi", "ls")}
    prs = RUNS / "pr_stagnation"
    tr["pr"] = _read_traj(sorted(prs.glob("traj_pr_shard*of7.csv")), "pr")
    tr["restart"] = _read_traj(sorted(prs.glob("traj_restart_shard*of7.csv")), "restart")
    tr["brdeh2026"] = _read_traj(sorted((RUNS / "variants").glob("traj_j120_s1-2-3_shard*.csv")), "init13_nobeta")
    insts = sorted({k[0] for k in tr["ls"]})
    ref = {i: instance_features(i)["lb_cpm"] for i in insts}
    val = lambda t, i, B: mean(value_at(t[(i, s)], B) for s in SEEDS)
    res = {"n_instances": len(insts), "seeds": SEEDS, "gap": {}, "cmp": {}, "shake": {}}
    for B in (1000, 2500, 5000):
        res["gap"][B] = {v: mean(100 * (val(t, i, B) - ref[i]) / ref[i] for i in insts) for v, t in tr.items()}
        res["cmp"][B] = {}
        for x, y in (("prfbi", "brdeh2026"), ("ls", "brdeh2026"), ("prfbi", "restart"), ("ls", "restart"), ("prfbi", "pr")):
            d = [100 * (val(tr[x], i, B) - val(tr[y], i, B)) / ref[i] for i in insts]
            nz = [t for t in d if t != 0]
            res["cmp"][B][f"{x} vs {y}"] = [round(mean(d), 3), sum(t < 0 for t in d), sum(t == 0 for t in d),
                                            sum(t > 0 for t in d), float(wilcoxon(nz).pvalue) if len(nz) > 10 else None]
    for m in ("prfbi", "ls"):
        rows = []
        for p in sorted(OUT.glob(f"runs_{m}_shard*of7.csv")):
            with open(p) as f:
                rows += list(csv.DictReader(f))
        tot = lambda k: sum(int(r[k]) for r in rows)
        res["shake"][m] = {"episodes_per_run": tot("episodes") / len(rows),
                           "budget_share": tot("shake_schedules") / (len(rows) * BUDGET),
                           "improvements_per_run": tot("shake_improvements") / len(rows),
                           "runs_with_improvement": mean(int(r["shake_improvements"]) > 0 for r in rows)}
    (OUT / "summary.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "analyze"])
    ap.add_argument("--mode", choices=["prfbi", "ls"], default="prfbi")
    ap.add_argument("--shard", type=int, nargs=2, default=[0, 1])
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    {"run": cmd_run, "analyze": cmd_analyze}[a.cmd](a)
