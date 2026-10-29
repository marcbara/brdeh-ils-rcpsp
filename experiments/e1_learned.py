
import argparse
import csv
import json
from collections import defaultdict
from statistics import mean

from experiments.a1_uniform import _read_traj
from experiments.init13 import BUDGET, RULES, instances, value_at
from harness.instances import instance_features
from harness.provenance import REPO_ROOT
from harness.solvers.brdeh import canonical_seeds, jl

K_ELITE = 20
SHARE = 0.5   # share of tries that use rule 14 (see docstring: smoke test showed 1.5 % inside the 80/20)
OUT = REPO_ROOT / "experiments" / "runs" / "e1_learned"
BASE = REPO_ROOT / "experiments" / "runs" / "variants"
RULES14 = RULES + ["LEARN"]

JL = r"""
struct LearnedPolicy
    pr :: Vector{Float64}          # indexed by task id; updated in place
end
@inline function (p::LearnedPolicy)(a, b)
    if a.start_time < b.start_time
        return true
    elseif a.start_time > b.start_time
        return false
    end
    return p.pr[a.id] < p.pr[b.id]
end
mutable struct Elite
    k :: Int
    st :: Vector{Vector{Int}}      # start times by task id (snapshots: Tasks are aliased)
    mk :: Vector{Int}
    hs :: Vector{UInt64}
end
_snap(sol, n) = (v = zeros(Int, n); for t in sol.tasks; v[t.id] = t.start_time; end; v)
function _refresh!(p::LearnedPolicy, e::Elite)
    fill!(p.pr, 0.0)
    for v in e.st
        p.pr .+= v
    end
    p.pr ./= length(e.st)
    return nothing
end
function _offer!(e::Elite, v::Vector{Int}, mk::Int)
    h = hash(v)
    h in e.hs && return false
    if length(e.mk) < e.k
        push!(e.st, v); push!(e.mk, mk); push!(e.hs, h)
        return true
    end
    w = argmax(e.mk)
    mk < e.mk[w] || return false
    e.st[w] = v; e.mk[w] = mk; e.hs[w] = h
    return true
end
function ado_e1_start(name::String, seed::Int, tieseed::Int, k::Int)
    mk = Int[]
    n = 0
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
    learned = LearnedPolicy(zeros(n))
    _refresh!(learned, e)
    pr14 = copy(learned.pr)                 # priorities used by the rule-14 init build
    cd(ADO_SOLVER_ROOT) do
        inp = readInputs(name)
        r = Xoshiro(seed)
        pols, _ = buildPolicies(inp, r, "B2")
        p14 = Any[pols...; learned]
        s = startRun(inp, p14, r; fbiMaxPasses = 1, policyKeepProb = 0.8, initPolicy = learned)
        @assert s.schedules == 3
        push!(mk, Int(s.bestSolution.makespan))
        _offer!(e, _snap(s.bestSolution, n), Int(s.bestSolution.makespan)) && _refresh!(learned, e)
    end
    m = minimum(mk)
    winners = findall(==(m), mk)
    w = winners[rand(Xoshiro(tieseed), 1:length(winners))]
    s = cd(ADO_SOLVER_ROOT) do
        inputs = readInputs(name)
        rng = Xoshiro(seed)
        pols, _ = buildPolicies(inputs, rng, "B2")
        p14 = Any[pols...; learned]
        # rebuild the winner's init schedule; rule 14 must use the priorities it had then
        ip = w == 14 ? LearnedPolicy(pr14) : p14[w]
        st = startRun(inputs, p14, rng; fbiMaxPasses = 1, policyKeepProb = 0.8, initPolicy = ip)
        w == 14 && (st.bestPolicy = learned)
        st
    end
    @assert s.bestSolution.makespan == m
    s.schedules = 42
    s.traj[1] = (iter = 0, schedules = 42, makespan = m, time = 0.0)
    return s, w, length(winners), mk, learned, e, n
end
function ado_e1_run!(s, target::Int, learned::LearnedPolicy, e::Elite, n::Int, share::Float64)
    tries = 0; tries14 = 0; ref14 = 0; rec14 = 0; entered = 0
    klein = s.sortPolicies[1:13]
    while s.schedules < target
        c0 = SCHEDULE_COUNT[]
        beta = rand(s.rng, Float64)
        pol = if rand(s.rng, Float64) < share
            learned
        else
            rand(s.rng, Float64) < s.policyKeepProb ? s.bestPolicy : rand(s.rng, klein)
        end
        is14 = pol === learned
        sol = MPJ_Heuristic(s.inputs, s.availability_times, beta, pol, true, s.rng)
        sol = localSearch(sol, s.inputs, s.availability_units; maxPasses = s.fbiMaxPasses)
        s.schedules += SCHEDULE_COUNT[] - c0
        s.iter += 1
        tries += 1; tries14 += is14; ref14 += (s.bestPolicy === learned)
        if _offer!(e, _snap(sol, n), Int(sol.makespan))
            _refresh!(learned, e); entered += 1
        end
        if sol.makespan < s.bestSolution.makespan
            s.bestBeta = beta
            is14 || (s.bestPolicy = pol)   # the Klein channel keeps its own reference rule
            s.bestSolution = sol
            rec14 += is14
            push!(s.traj, (iter = s.iter, schedules = s.schedules,
                           makespan = sol.makespan, time = 0.0))
        end
    end
    return (tries, tries14, ref14, rec14, entered, s.bestPolicy === learned)
end
"""


def cmd_run(a):
    J = jl()
    J.seval(JL)
    start = J.seval("ado_e1_start")
    run = J.seval("ado_e1_run!")
    seeds = canonical_seeds()
    mine = instances(a.bench)[a.shard[0]::a.shard[1]]
    OUT.mkdir(parents=True, exist_ok=True)
    tag = f"{a.bench}_s{'-'.join(map(str, a.seeds))}_shard{a.shard[0]}of{a.shard[1]}"
    with open(OUT / f"traj_{tag}.csv", "w", newline="") as ft, open(OUT / f"runs_{tag}.csv", "w", newline="") as fr:
        wt, wr = csv.writer(ft), csv.writer(fr)
        wt.writerow(["instance", "seed_idx", "iter", "schedules", "makespan"])
        wr.writerow(["instance", "seed_idx", "init_winner", "n_tied", "init_mk_best13", "init_mk_learn",
                     "tries", "tries_learn", "tries_ref_learn", "records_learn", "elite_entries", "final_ref_learn"])
        for c, inst in enumerate(mine, 1):
            for si in a.seeds:
                s, w, nt, mk, learned, e, n = start(inst, seeds[si - 1], 1_000_003 * si + 7, K_ELITE)
                mk = [int(x) for x in mk]
                st = run(s, BUDGET, learned, e, n, SHARE)
                wr.writerow([inst, si, RULES14[int(w) - 1], int(nt), min(mk[:13]), mk[13]] +
                            [int(x) for x in st[:5]] + [int(bool(st[5]))])
                m = J.ado_rows(s, 1)
                for k in range(m.shape[0]):
                    if int(m[k, 1]) <= BUDGET:
                        wt.writerow([inst, si, int(m[k, 0]), int(m[k, 1]), int(m[k, 2])])
            if c % 25 == 0:
                print(f"{tag}: {c}/{len(mine)}", flush=True)
    print("done", tag)


def cmd_analyze(a):
    from scipy.stats import wilcoxon
    tag = f"{a.bench}_s{'-'.join(map(str, a.seeds))}"
    e1 = _read_traj(sorted(OUT.glob(f"traj_{tag}_shard*.csv")))
    bp = sorted(BASE.glob(f"traj_{tag}_shard*.csv"))
    base, b2 = _read_traj(bp, "init13_nobeta"), _read_traj(bp, "b2")
    insts = sorted({k[0] for k in e1})
    ref = {i: instance_features(i)["lb_cpm"] for i in insts}
    val = lambda t, i, B: mean(value_at(t[(i, s)], B) for s in a.seeds)
    res = {"bench": a.bench, "seeds": a.seeds, "n_instances": len(insts), "k_elite": K_ELITE, "gap": {}, "vs_base": {}, "vs_b2": {}}
    for B in (42, 250, 1000, 2500, 5000):
        res["gap"][B] = {n: mean(100 * (val(t, i, B) - ref[i]) / ref[i] for i in insts)
                         for n, t in (("b2", b2), ("brdeh2026", base), ("e1", e1))}

    def cmp(x, y, B):
        d = [val(x, i, B) - val(y, i, B) for i in insts]
        nz = [t for t in d if t != 0]
        return {"e1_better": sum(t < 0 for t in d), "tie": sum(t == 0 for t in d), "e1_worse": sum(t > 0 for t in d),
                "wilcoxon_p": float(wilcoxon(nz).pvalue) if len(nz) > 10 else None,
                "mean_diff": mean(d)}
    for B in (1000, 2500, 5000):
        res["vs_base"][B] = cmp(e1, base, B)
        res["vs_b2"][B] = cmp(e1, b2, B)
    runs = []
    for p in sorted(OUT.glob(f"runs_{tag}_shard*.csv")):
        with open(p) as f:
            runs += list(csv.DictReader(f))
    tot = lambda k: sum(int(r[k]) for r in runs)
    win = defaultdict(int)
    for r in runs:
        win[r["init_winner"]] += 1
    res["learned_rule"] = {
        "runs": len(runs),
        "init_learn_better_than_best13_share": mean(int(r["init_mk_learn"]) < int(r["init_mk_best13"]) for r in runs),
        "init_learn_equal_best13_share": mean(int(r["init_mk_learn"]) == int(r["init_mk_best13"]) for r in runs),
        "init_winner_counts": dict(sorted(win.items(), key=lambda kv: -kv[1])),
        "share_tries_with_learn": tot("tries_learn") / tot("tries"),
        "share_tries_learn_is_reference": tot("tries_ref_learn") / tot("tries"),
        "share_records_by_learn": tot("records_learn") / max(1, sum(len(e1[k]) - 1 for k in e1)),
        "final_reference_learn_share": mean(int(r["final_ref_learn"]) for r in runs),
        "elite_entries_per_run": tot("elite_entries") / len(runs),
    }
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
