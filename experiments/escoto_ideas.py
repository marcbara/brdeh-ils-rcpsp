
import argparse
import csv
import gzip
import hashlib
import json
import time
from statistics import mean

from experiments.a1_uniform import _read_traj
from experiments.bidir import JL as JL_BI, write_reversed
from experiments.bidir_ls import JL as JL_BILS
from experiments.confirm_bidir import JL as JL_DIR, OUT as OUT_DIR
from experiments.confirm_ls import _cmp, _feasible, _holm
from experiments.e1_learned import JL as JL_E1
from experiments.init13 import BUDGET, RULES, instances, value_at
from experiments.pr2_stagnation import JL as JL_PR2
from experiments.pr_stagnation import CAP, JL as JL_PRS, S_STALL, STRIDE
from harness.evaluate import load_optima
from harness.instances import instance_features
from harness.provenance import REPO_ROOT
from harness.solvers.brdeh import canonical_seeds, jl

OUT = REPO_ROOT / "experiments" / "runs" / "escoto_ideas"
HYB_FILE = REPO_ROOT / "experiments" / "escoto_hybrid_policies_90f8ab1.jl"
HYB_SHA = "cbc6f3a2a3aaa565fab42b149fa0090d8151525ef72561e89894fa000d5a1e87"
assert RULES[8] == "LFT" and RULES[12] == "HWRU"
H = 0.3
BIG = 10 ** 9
# spec: (pool, beta mode, fbi mode); beta 0 uniform, 1 triangular inherited, 2 narrow inherited,
# 3 narrow fixed 0.5; fbi 0 one pass, p > 0 dynamic with patience p, -1 unlimited every try
SPECS = {
    "": ("k13", 0, 0), "_dyn1": ("k13", 0, 1), "_dyn5": ("k13", 0, 5), "_fbiall": ("k13", 0, -1),
    "_tri": ("k13", 1, 0), "_narrow": ("k13", 2, 0), "_narrowfix": ("k13", 3, 0),
    "_hyb": ("hyb", 0, 0), "_hyb14": ("k14", 0, 0), "_lft": ("lft", 0, 0), "_hwru": ("hwru", 0, 0),
}
BASES = {"I": (3, True), "N": (1, False)}          # dirs, local search
ARMS = {b + k: (b,) + v for b in BASES for k, v in SPECS.items()}

JL = r"""
function ado_pool(inp, rng, kind::String)
    pols, _ = buildPolicies(inp, rng, "B2")
    kind == "k13" && return pols
    kind == "lft" && return [pols[9]]
    kind == "hwru" && return [pols[13]]
    hyb = makeSingleHybridPolicy(inp, rng)
    kind == "hyb" && return hyb
    kind == "k14" && return vcat(pols, hyb)
    error("pool $kind")
end
function ado_gen_start(name::String, revname::String, seed::Int, tieseed::Int, dirs::Int, kind::String)
    mk = Int[]; snaps = Vector{Vector{Int}}(); isback = Bool[]; rule = Int[]; n = 0
    e = Elite(20, Vector{Int}[], Int[], UInt64[])
    side = nothing; L = 0
    c0 = SCHEDULE_COUNT[]
    cd(ADO_SOLVER_ROOT) do
        for (d, fname) in ((1, name), (2, revname))
            dirs & d == 0 && continue
            L = length(ado_pool(readInputs(fname), Xoshiro(seed), kind))
            for j in 1:L
                inp = readInputs(fname); n = maximum(t.id for t in inp.tasks)
                r = Xoshiro(seed); pool = ado_pool(inp, r, kind)
                s = startRun(inp, pool, r; fbiMaxPasses = 1, policyKeepProb = 0.8, initPolicy = pool[j])
                push!(mk, Int(s.bestSolution.makespan)); push!(rule, j); push!(isback, d == 2)
                push!(snaps, d == 2 ? _flip(s.bestSolution, n) : _snap(s.bestSolution, n))
            end
        end
        inp = readInputs(revname)
        side = RevSide(inp, ado_pool(inp, Xoshiro(seed), kind), Dict(x.id => zeros(x.units) for x in inp.resources),
                       zeros(Int64, inp.nResources, inp.timeHorizon + 1))
    end
    init_spent = SCHEDULE_COUNT[] - c0
    @assert init_spent == 3 * L * count_ones(dirs)
    for k in eachindex(mk)
        _offer!(e, snaps[k], mk[k])
    end
    m = minimum(mk)
    wf = [k for k in eachindex(mk) if mk[k] == m && !isback[k]]
    wb = [k for k in eachindex(mk) if mk[k] == m && isback[k]]
    pool = isempty(wf) ? wb : wf
    w = pool[rand(Xoshiro(tieseed), 1:length(pool))]
    tie = !isempty(wf) && !isempty(wb)
    ridx = rule[w]; back = isback[w]
    s = cd(ADO_SOLVER_ROOT) do
        inputs = readInputs(name)
        rng = Xoshiro(seed)
        pols = ado_pool(inputs, rng, kind)
        startRun(inputs, pols, rng; fbiMaxPasses = 1, policyKeepProb = 0.8, initPolicy = pols[ridx])
    end
    if back
        s.bestSolution = Solution(); s.bestSolution.makespan = Float64(m)
    end
    @assert Int(s.bestSolution.makespan) == m
    s.schedules = init_spent
    s.traj[1] = (iter = 0, schedules = s.schedules, makespan = m, time = 0.0)
    return s, side, e, n, ridx, back, tie
end
function _ado_beta(rng, bmode::Int, bb::Float64, h::Float64)
    bmode == 0 && return rand(rng, Float64)
    bmode == 1 && return rand(rng, TriangularDist(0.0, 1.0, bb))
    c = bmode == 2 ? bb : 0.5
    lo = max(0.0, c - h); hi = min(1.0, c + h)
    return rand(rng, TriangularDist(lo, hi, clamp(c, lo, hi)))
end
# returns (tries, tries_backward, episodes, ls_schedules, ls_records, deep_tries, deep_extra_schedules,
#          deep_records, beta_records, max_overshoot)
function ado_gen_run!(s, side, target::Int, e::Elite, n::Int, ridx::Int, back::Bool, ls::Bool, S::Int, cap::Int,
                      bmode::Int, h::Float64, fbimode::Int)
    gbest = Ref(Int(s.bestSolution.makespan))
    pols = s.sortPolicies
    bb = 0.5
    since = 0; sincet = 0; episodes = 0; used = 0; found = 0
    deep = 0; deepx = 0; deeprec = 0; brec = 0; nb = 0
    while s.schedules < target
        c0 = SCHEDULE_COUNT[]
        beta = _ado_beta(s.rng, bmode, bb, h)
        pol = rand(s.rng, Float64) < s.policyKeepProb ? pols[ridx] : rand(s.rng, pols)
        idx = findfirst(p -> p === pol, pols)
        isdeep = fbimode < 0 || (fbimode > 0 && sincet >= fbimode)
        passes = isdeep ? typemax(Int64) : 1
        if back
            nb += 1
            sol = MPJ_Heuristic(side.inputs, side.at, beta, side.pols[idx], true, s.rng)
            st0 = _flip(sol, n); mk0 = Int(sol.makespan)
            sol = localSearch(sol, side.inputs, side.au; maxPasses = passes)
            mk = Int(sol.makespan)
            st = mk < mk0 ? _flip(sol, n) : st0
        else
            sol = MPJ_Heuristic(s.inputs, s.availability_times, beta, pol, true, s.rng)
            st0 = _snap(sol, n); mk0 = Int(sol.makespan)
            sol = localSearch(sol, s.inputs, s.availability_units; maxPasses = passes)
            mk = Int(sol.makespan)
            st = mk < mk0 ? _snap(sol, n) : st0
        end
        spent = SCHEDULE_COUNT[] - c0
        if isdeep
            deep += 1; deepx += spent - 3
        end
        s.schedules += spent
        s.iter += 1
        if mk < gbest[]
            gbest[] = mk; ridx = idx; since = 0; sincet = 0
            if bmode in (1, 2)
                bb = beta; brec += 1
            end
            deeprec += isdeep
            _offer!(e, st, mk)
            push!(s.traj, (iter = s.iter, schedules = s.schedules, makespan = Float64(mk), time = 0.0))
        else
            since += spent
            sincet = (isdeep && fbimode > 0) ? 0 : sincet + 1
        end
        if ls && since >= S && s.schedules < target
            episodes += 1
            u, f = _local_search!(s, e, n, gbest, cap, target)
            used += u; found += f
            since = 0
            f > 0 && (sincet = 0)
        end
    end
    @assert minimum(e.mk) == gbest[]
    return (s.iter, nb, episodes, used, found, deep, deepx, deeprec, brec, s.schedules - target)
end
"""


def _setup():
    assert hashlib.sha256(HYB_FILE.read_bytes()).hexdigest() == HYB_SHA, "composite-rule file changed"
    J = jl()
    for code in (JL_E1, JL_PRS, JL_PR2, JL_BI, JL_BILS, JL_DIR):
        J.seval(code)
    J.include(str(HYB_FILE))
    J.seval(JL)
    return J


VIDEO5 = ["I_hyb", "I_hyb14", "I_lft", "I_dyn1", "I_narrow"]
VIDEO_N = ["N_hyb", "N_hyb14", "N_lft", "N_dyn1", "N_narrow"]   # '<bench>v5n': the same ideas on BR-DEH 2026


def _scope(bench):
    """(benchmark, instances, seeds); 'j120pilot' = the usual 100-instance J120 pilot, seeds 1-3;
    '<bench>v5' = complete set, seeds 1-10, the five arms confirmed for the video (VIDEO5)."""
    if bench == "j120pilot":
        return "j120", instances("j120")[::STRIDE], [1, 2, 3]
    if bench.endswith("v5"):
        return bench[:-2], instances(bench[:-2]), list(range(1, 11))
    if bench.endswith("v5n"):
        return bench[:-3], instances(bench[:-3]), list(range(1, 11))
    return bench, instances(bench), list(range(1, 11))


def run_one(J, fns, inst, rev, si, arm, budget=BUDGET):
    base, pool, bmode, fbimode = ARMS[arm]
    dirs, ls = BASES[base]
    seeds = canonical_seeds()
    start, run = fns
    s, side, e, n, ridx, back, tie = start(inst, rev, seeds[si - 1], 1_000_003 * si + 7, dirs, pool)
    st = run(s, side, budget, e, n, int(ridx), bool(back), ls, S_STALL, CAP, bmode, H, fbimode)
    return s, e, bool(back), bool(tie), int(ridx), [int(x) for x in st]


def cmd_run(a):
    J = _setup()
    fns = (J.seval("ado_gen_start"), J.seval("ado_gen_run!"))
    best = J.seval("ado_best")
    bench, insts, seeds = _scope(a.bench)
    arms = a.arms.split(",") if a.arms else (VIDEO5 if a.bench.endswith("v5") else VIDEO_N if a.bench.endswith("v5n") else list(ARMS))
    mine = insts[a.shard[0]::a.shard[1]]
    OUT.mkdir(parents=True, exist_ok=True)
    tag = f"{a.bench}_shard{a.shard[0]}of{a.shard[1]}"
    with open(OUT / f"traj_{tag}.csv", "w", newline="") as ft, open(OUT / f"runs_{tag}.csv", "w", newline="") as fr, \
            gzip.open(OUT / f"bests_{tag}.csv.gz", "wt", newline="") as fb:
        wt, wr, wb = csv.writer(ft), csv.writer(fr), csv.writer(fb)
        wt.writerow(["instance", "seed_idx", "variant", "iter", "schedules", "makespan"])
        wr.writerow(["instance", "seed_idx", "variant", "init_back_wins", "init_tie", "init_rule", "tries",
                     "tries_backward", "episodes", "ls_schedules", "ls_records", "deep_tries", "deep_extra_schedules",
                     "deep_records", "beta_records", "overshoot", "cpu_s"])
        wb.writerow(["instance", "seed_idx", "variant", "makespan", "starts"])
        for inst in mine:
            rev = write_reversed(inst)
            for si in seeds:
                for arm in arms:
                    t0 = time.process_time()
                    s, e, back, tie, ridx, st = run_one(J, fns, inst, rev, si, arm)
                    cpu = time.process_time() - t0
                    wr.writerow([inst, si, arm, int(back), int(tie), ridx] + st + [round(cpu, 3)])
                    mk, v = best(e)
                    wb.writerow([inst, si, arm, int(mk), " ".join(str(int(x)) for x in v)])
                    m = J.ado_rows(s, 1)
                    for k in range(m.shape[0]):
                        if int(m[k, 1]) <= BUDGET:
                            wt.writerow([inst, si, arm, int(m[k, 0]), int(m[k, 1]), int(m[k, 2])])
            ft.flush()
    print("done", tag)


def _rows(J, s):
    m = J.ado_rows(s, 1)
    return [(int(m[k, 0]), int(m[k, 1]), int(m[k, 2])) for k in range(m.shape[0]) if int(m[k, 1]) <= BUDGET]


def cmd_check(a):
    """Known answers on a sample: the bases reproduce the confirmed campaign; switches at their
    neutral value reproduce the base; the inherited-triangular arm reproduces the 2024-style
    beta of the 'variants' campaign."""
    J = _setup()
    fns = (J.seval("ado_gen_start"), J.seval("ado_gen_run!"))
    res = {}
    for bench, k in (("j30", 4), ("j60", 3), ("j120", 3)):
        insts = instances(bench)[::max(1, len(instances(bench)) // k)][:k]
        paths = sorted(OUT_DIR.glob(f"traj_{bench}_shard*of*.csv"))
        ref = {m: _read_traj(paths, m) for m in ("base", "winner_ls")}
        ok = tot = 0
        for inst in insts:
            rev = write_reversed(inst)
            for si in (1, 2):
                for arm, refm in (("N", "base"), ("I", "winner_ls")):
                    s, *_ = run_one(J, fns, inst, rev, si, arm)
                    tot += 1; ok += _rows(J, s) == [tuple(x) for x in ref[refm][(inst, si)]]
        res[f"bases_equal_campaign_{bench}"] = [ok, tot]
    # neutral switches: dynamic FBI with infinite patience == base
    inst = instances("j120")[0]; rev = write_reversed(inst)
    ARMS["N_dynoff"] = ("N", "k13", 0, BIG); ARMS["I_dynoff"] = ("I", "k13", 0, BIG)
    ok = tot = 0
    for si in (1, 2, 3):
        for x, y in (("N_dynoff", "N"), ("I_dynoff", "I")):
            a1, *_ = run_one(J, fns, inst, rev, si, x)
            r1 = _rows(J, a1)
            a2, *_ = run_one(J, fns, inst, rev, si, y)
            tot += 1; ok += r1 == _rows(J, a2)
    res["dyn_infinite_patience_equals_base"] = [ok, tot]
    # 2024-style inherited beta on the 13-rule start == 'variants' init13 (J120, seeds 1-3)
    vp = sorted((REPO_ROOT / "experiments" / "runs" / "variants").glob("traj_j120_s1-2-3_shard*.csv"))
    vref = _read_traj(vp, "init13")
    sample = instances("j120")[::STRIDE][:4]
    ok = tot = 0
    for inst in sample:
        rev = write_reversed(inst)
        for si in (1, 2, 3):
            s, *_ = run_one(J, fns, inst, rev, si, "N_tri")
            tot += 1; ok += _rows(J, s) == [tuple(x) for x in vref[(inst, si)]]
    res["N_tri_equals_variants_init13"] = [ok, tot]
    # budget and feasibility on every new arm, one instance
    bad = []
    for arm in [x for x in ARMS if x not in ("N_dynoff", "I_dynoff")]:
        s, e, *_ , st = run_one(J, fns, inst, rev, 1, arm)
        mk, v = J.ado_best(e)
        err = _feasible(inst, int(mk), " ".join(str(int(x)) for x in v))
        if err:
            bad.append([arm, err])
    res["feasibility_all_arms_violations"] = bad
    (OUT / "check.json").parent.mkdir(parents=True, exist_ok=True)
    (OUT / "check.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


def _paths(bench, kind):
    return sorted(OUT.glob(f"{kind}_{bench}_shard*of*.csv*"))


def cmd_verify(a):
    bench, insts, seeds = _scope(a.bench)
    tr = {m: _read_traj(_paths(a.bench, "traj"), m) for m in ARMS}
    bad, n = [], 0
    for p in _paths(a.bench, "bests"):
        with gzip.open(p, "rt") as f:
            for r in csv.DictReader(f):
                key = (r["instance"], int(r["seed_idx"]))
                err = _feasible(r["instance"], int(r["makespan"]), r["starts"])
                n += 1
                if err is None and tr[r["variant"]][key][-1][2] != int(r["makespan"]) \
                        and tr[r["variant"]][key][-1][1] <= BUDGET:
                    err = "traj final != best"
                if err:
                    bad.append([*key, r["variant"], err])
    res = {"checked": n, "violations": len(bad), "first": bad[:20]}
    (OUT / f"verify_{a.bench}.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


def cmd_analyze(a):
    bench, insts, seeds = _scope(a.bench)
    opt = load_optima()
    ref = {i: (opt[i] if bench == "j30" else instance_features(i)["lb_cpm"]) for i in insts}
    tr = {m: _read_traj(_paths(a.bench, "traj"), m) for m in ARMS}
    tr = {m: t for m, t in tr.items() if t}
    if a.bench.endswith(("v5", "v5n")):     # bases from the confirmed campaign (reproduced exactly: check.json)
        cb = sorted(OUT_DIR.glob(f"traj_{bench}_shard*of*.csv"))
        tr["I"], tr["N"] = _read_traj(cb, "winner_ls"), _read_traj(cb, "base")
    rows = []
    for p in _paths(a.bench, "runs"):
        with open(p) as f:
            rows += list(csv.DictReader(f))
    res = {"bench": a.bench, "n_instances": len(insts), "seeds": seeds, "gap": {}, "cmp": {}, "mech": {}}
    for B in (1000, 2500, 5000):
        res["gap"][B] = {m: round(mean(100 * (mean(value_at(t[(i, s)], B) for s in seeds) - ref[i]) / ref[i]
                                       for i in insts), 3) for m, t in tr.items()}
        pairs = [(m, m[0]) for m in tr if m not in ("I", "N")] + [("I", "N")]
        c = {f"{x} vs {y}": _cmp(tr, insts, seeds, ref, B, x, y) for x, y in pairs if y in tr}
        holm = _holm([v["p"] for v in c.values()])
        for (k, v), ph in zip(c.items(), holm):
            v["p_holm"] = ph
        res["cmp"][B] = c
    for m in tr:
        rr = [r for r in rows if r["variant"] == m]
        if not rr:
            continue
        f = lambda k: round(mean(float(r[k]) for r in rr), 3)
        res["mech"][m] = {"tries": f("tries"), "deep_tries": f("deep_tries"),
                          "deep_extra_share": round(sum(int(r["deep_extra_schedules"]) for r in rr) / (len(rr) * BUDGET), 3),
                          "deep_records": f("deep_records"), "beta_records": f("beta_records"),
                          "ls_share": round(sum(int(r["ls_schedules"]) for r in rr) / (len(rr) * BUDGET), 3),
                          "init_rule_14_share": round(mean(r["init_rule"] == "14" for r in rr), 3),
                          "max_overshoot": max(int(r["overshoot"]) for r in rr), "cpu_s": f("cpu_s")}
    (OUT / f"summary_{a.bench}.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "check", "verify", "analyze"])
    ap.add_argument("bench", nargs="?", default="j120pilot")
    ap.add_argument("--shard", type=int, nargs=2, default=[0, 1])
    ap.add_argument("--arms", default="")
    a = ap.parse_args()
    {"run": cmd_run, "check": cmd_check, "verify": cmd_verify, "analyze": cmd_analyze}[a.cmd](a)
