
import argparse
import csv
import gzip
import json
from statistics import mean

from experiments.a1_uniform import _read_traj
from experiments.confirm_ls import _cmp, _feasible
from experiments.e1_learned import JL as JL_E1
from experiments.init13 import BUDGET, instances, value_at
from experiments.pr_stagnation import SEEDS, STRIDE
from harness.instances import INPUTS, instance_features
from harness.provenance import REPO_ROOT
from harness.solvers.brdeh import canonical_seeds, jl

OUT = REPO_ROOT / "experiments" / "runs" / "bidir"
RUNS = REPO_ROOT / "experiments" / "runs"
REV = OUT / "rev"
MODES = ("bi_init", "bi_dir", "bi_mix")
CHECK_N = 10                     # instances with the fwd13 known-answer run


def write_reversed(inst: str) -> str:
    """Write the reversed .sm (same line layout readInputs expects); return the solver-relative name."""
    lines = (INPUTS / inst).read_text().splitlines()
    n = int(lines[5].split()[-1])
    succ = {i: [int(x) for x in lines[17 + i].split()[3:]] for i in range(1, n + 1)}
    pred = {i: [] for i in range(1, n + 1)}
    for i, ss in succ.items():
        for j in ss:
            pred[j].append(i)
    out = list(lines)
    for new in range(1, n + 1):
        old = n + 1 - new
        ns = sorted(n + 1 - p for p in pred[old])
        out[17 + new] = f"{new} 1 {len(ns)} " + " ".join(map(str, ns))
        parts = lines[17 + n + 4 + old].split()
        out[17 + n + 4 + new] = " ".join([str(new)] + parts[1:])
    path = REV / inst
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(out) + "\n")
    return "../../../experiments/runs/bidir/rev/" + inst          # relative to solvers/br-deh/inputs/


JL = r"""
mutable struct RevSide
    inputs; pols; at; au
end
# start times in ORIGINAL labels of a schedule built on the reversed instance
function _flip(sol, n)
    C = Int(sol.makespan); v = zeros(Int, n)
    for t in sol.tasks
        v[n + 1 - t.id] = C - (t.start_time + t.duration)
    end
    return v
end
function ado_bidir_start(name::String, revname::String, seed::Int, tieseed::Int, both::Bool)
    mk = Int[]; snaps = Vector{Vector{Int}}(); n = 0
    e = Elite(20, Vector{Int}[], Int[], UInt64[])
    side = nothing
    cd(ADO_SOLVER_ROOT) do
        for j in 1:13
            inp = readInputs(name); n = maximum(t.id for t in inp.tasks)
            r = Xoshiro(seed); pols, _ = buildPolicies(inp, r, "B2")
            s = startRun(inp, pols, r; fbiMaxPasses = 1, policyKeepProb = 0.8, initPolicy = pols[j])
            push!(mk, Int(s.bestSolution.makespan)); push!(snaps, _snap(s.bestSolution, n))
        end
        if both
            for j in 1:13
                inp = readInputs(revname)
                r = Xoshiro(seed); pols, _ = buildPolicies(inp, r, "B2")
                s = startRun(inp, pols, r; fbiMaxPasses = 1, policyKeepProb = 0.8, initPolicy = pols[j])
                push!(mk, Int(s.bestSolution.makespan)); push!(snaps, _flip(s.bestSolution, n))
            end
        end
        inp = readInputs(revname)
        pols, _ = buildPolicies(inp, Xoshiro(seed), "B2")
        side = RevSide(inp, pols, Dict(x.id => zeros(x.units) for x in inp.resources),
                       zeros(Int64, inp.nResources, inp.timeHorizon + 1))
    end
    for k in eachindex(mk)
        _offer!(e, snaps[k], mk[k])
    end
    m = minimum(mk)
    winners = findall(==(m), mk)
    w = winners[rand(Xoshiro(tieseed), 1:length(winners))]
    ridx = (w - 1) % 13 + 1; back = w > 13
    s = cd(ADO_SOLVER_ROOT) do
        inputs = readInputs(name)
        rng = Xoshiro(seed)
        pols, _ = buildPolicies(inputs, rng, "B2")
        startRun(inputs, pols, rng; fbiMaxPasses = 1, policyKeepProb = 0.8, initPolicy = pols[ridx])
    end
    if back
        s.bestSolution = Solution(); s.bestSolution.makespan = Float64(m)
    end
    @assert Int(s.bestSolution.makespan) == m
    s.schedules = both ? 78 : 39
    s.traj[1] = (iter = 0, schedules = s.schedules, makespan = m, time = 0.0)
    return s, side, e, n, ridx, back
end
# pback: probability that a try is built backward (0 and 1 consume no extra random number)
function ado_bidir_run!(s, side, target::Int, e::Elite, n::Int, ridx::Int, pback::Float64)
    gbest = Int(s.bestSolution.makespan)
    pols = s.sortPolicies
    nback = 0
    while s.schedules < target
        c0 = SCHEDULE_COUNT[]
        beta = rand(s.rng, Float64)
        pol = rand(s.rng, Float64) < s.policyKeepProb ? pols[ridx] : rand(s.rng, pols)
        idx = findfirst(p -> p === pol, pols)
        back = pback >= 1.0 ? true : pback <= 0.0 ? false : rand(s.rng, Float64) < pback
        if back
            nback += 1
            sol = MPJ_Heuristic(side.inputs, side.at, beta, side.pols[idx], true, s.rng)
            st0 = _flip(sol, n); mk0 = Int(sol.makespan)
            sol = localSearch(sol, side.inputs, side.au; maxPasses = s.fbiMaxPasses)
            mk = Int(sol.makespan)
            st = mk < mk0 ? _flip(sol, n) : st0
        else
            sol = MPJ_Heuristic(s.inputs, s.availability_times, beta, pol, true, s.rng)
            st0 = _snap(sol, n); mk0 = Int(sol.makespan)
            sol = localSearch(sol, s.inputs, s.availability_units; maxPasses = s.fbiMaxPasses)
            mk = Int(sol.makespan)
            st = mk < mk0 ? _snap(sol, n) : st0
        end
        s.schedules += SCHEDULE_COUNT[] - c0
        s.iter += 1
        if mk < gbest
            gbest = mk; ridx = idx
            _offer!(e, st, mk)
            push!(s.traj, (iter = s.iter, schedules = s.schedules, makespan = Float64(mk), time = 0.0))
        end
    end
    @assert s.schedules <= target + 2
    @assert minimum(e.mk) == gbest
    return (s.iter, nback)
end
ado_best(e) = (i = argmin(e.mk); (e.mk[i], e.st[i]))
"""


def cmd_run(a):
    J = jl()
    J.seval(JL_E1)
    J.seval(JL)
    start, run, best = J.seval("ado_bidir_start"), J.seval("ado_bidir_run!"), J.seval("ado_best")
    seeds = canonical_seeds()
    sub = instances("j120")[::STRIDE]
    check = set(sub[::len(sub) // CHECK_N][:CHECK_N])
    mine = sub[a.shard[0]::a.shard[1]]
    OUT.mkdir(parents=True, exist_ok=True)
    tag = f"shard{a.shard[0]}of{a.shard[1]}"
    with open(OUT / f"traj_{tag}.csv", "w", newline="") as ft, open(OUT / f"runs_{tag}.csv", "w", newline="") as fr, \
            gzip.open(OUT / f"bests_{tag}.csv.gz", "wt", newline="") as fb:
        wt, wr, wb = csv.writer(ft), csv.writer(fr), csv.writer(fb)
        wt.writerow(["instance", "seed_idx", "variant", "iter", "schedules", "makespan"])
        wr.writerow(["instance", "seed_idx", "variant", "init_back_wins", "tries", "back_tries"])
        wb.writerow(["instance", "seed_idx", "variant", "makespan", "starts"])
        for inst in mine:
            rev = write_reversed(inst)
            for si in SEEDS:
                for mode in (("fwd13",) if inst in check else ()) + MODES:
                    s, side, e, n, ridx, back = start(inst, rev, seeds[si - 1], 1_000_003 * si + 7, mode != "fwd13")
                    pback = {"fwd13": 0.0, "bi_init": 0.0, "bi_dir": 1.0 if back else 0.0, "bi_mix": 0.5}[mode]
                    it, nb = run(s, side, BUDGET, e, n, int(ridx), pback)
                    wr.writerow([inst, si, mode, int(bool(back)), int(it), int(nb)])
                    mk, v = best(e)
                    wb.writerow([inst, si, mode, int(mk), " ".join(str(int(x)) for x in v)])
                    m = J.ado_rows(s, 1)
                    for k in range(m.shape[0]):
                        if int(m[k, 1]) <= BUDGET:
                            wt.writerow([inst, si, mode, int(m[k, 0]), int(m[k, 1]), int(m[k, 2])])
            ft.flush()
    print("done", tag)


def cmd_analyze(a):
    paths = sorted(OUT.glob("traj_shard*.csv"))
    tr = {m: _read_traj(paths, m) for m in MODES + ("fwd13",)}
    tr["brdeh2026"] = _read_traj(sorted((RUNS / "variants").glob("traj_j120_s1-2-3_shard*.csv")), "init13_nobeta")
    tr["ls"] = _read_traj(sorted((RUNS / "pr2_stagnation").glob("traj_ls_shard*of7.csv")), "ls")
    insts = sorted({k[0] for k in tr["bi_init"]})
    ref = {i: instance_features(i)["lb_cpm"] for i in insts}
    res = {"n_instances": len(insts), "seeds": SEEDS, "checks": {}, "gap": {}, "cmp": {}}
    res["checks"]["fwd13_equals_brdeh2026"] = [sum(tr["brdeh2026"][k] == v for k, v in tr["fwd13"].items()),
                                               len(tr["fwd13"])]
    bad = n = 0
    for p in sorted(OUT.glob("bests_shard*.csv.gz")):
        with gzip.open(p, "rt") as f:
            for r in csv.DictReader(f):
                n += 1
                err = _feasible(r["instance"], int(r["makespan"]), r["starts"])
                fin = tr[r["variant"]][(r["instance"], int(r["seed_idx"]))][-1][2]
                bad += bool(err) or fin != int(r["makespan"])
    res["checks"]["best_schedules_checked"] = n
    res["checks"]["violations"] = bad
    init = {}
    for B in (39, 78, 1000, 2500, 5000):
        res["gap"][B] = {v: round(mean(100 * (mean(value_at(t[(i, s)], B) for s in SEEDS) - ref[i]) / ref[i]
                                       for i in insts), 3) for v, t in tr.items()
                         if v != "fwd13" and t[(insts[0], 1)][0][1] <= B}     # bi_* start at 78
    for B in (78, 1000, 2500, 5000):
        res["cmp"][B] = {f"{x} vs {y}": _cmp(tr, insts, SEEDS, ref, B, x, y)
                         for x, y in (("bi_init", "brdeh2026"), ("bi_dir", "brdeh2026"), ("bi_mix", "brdeh2026"),
                                      ("bi_dir", "bi_init"), ("bi_mix", "bi_init"))}
    rows = []
    for p in sorted(OUT.glob("runs_shard*.csv")):
        with open(p) as f:
            rows += list(csv.DictReader(f))
    rb = [r for r in rows if r["variant"] == "bi_init"]
    res["init_backward_wins_share"] = round(mean(int(r["init_back_wins"]) for r in rb), 3)
    (OUT / "summary.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "analyze"])
    ap.add_argument("--shard", type=int, nargs=2, default=[0, 1])
    a = ap.parse_args()
    {"run": cmd_run, "analyze": cmd_analyze}[a.cmd](a)
