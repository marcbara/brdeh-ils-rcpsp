
import argparse
import csv
import gzip
import json
import time
from statistics import mean

from experiments.a1_uniform import _read_traj
from experiments.bidir import JL as JL_BI, write_reversed
from experiments.bidir_ls import JL as JL_BILS
from experiments.confirm import _load
from experiments.confirm_ls import OUT as OUT_LS, _ci, _cmp, _feasible, _holm
from experiments.e1_learned import JL as JL_E1
from experiments.init13 import BUDGET, instances, value_at
from experiments.pr2_stagnation import JL as JL_PR2
from experiments.pr_stagnation import CAP, JL as JL_PRS, S_STALL, STRIDE
from harness.evaluate import load_optima
from harness.instances import instance_features
from harness.provenance import REPO_ROOT
from harness.solvers.brdeh import canonical_seeds, jl

OUT = REPO_ROOT / "experiments" / "runs" / "confirm_bidir"
SEEDS = list(range(1, 11))
HOLD = list(range(11, 21))
# mode -> (init directions: 1 fwd, 3 both; try rule; local search; seeds)
ARMS = {
    "base": (1, "fwd", False, SEEDS),
    "base_ls": (1, "fwd", True, SEEDS),
    "fwd78": (3, "fwd", False, SEEDS + HOLD),
    "back78": (3, "back", False, SEEDS + HOLD),
    "winner": (3, "winner", False, SEEDS),
    "loser": (3, "loser", False, SEEDS),
    "mix": (3, "mix", False, SEEDS),
    "winner_ls": (3, "winner", True, SEEDS),
}

JL = r"""
# dirs: 1 = forward rules only, 3 = forward + backward. Ties: forward first.
function ado_dir_start(name::String, revname::String, seed::Int, tieseed::Int, dirs::Int)
    mk = Int[]; snaps = Vector{Vector{Int}}(); isback = Bool[]; rule = Int[]; n = 0
    e = Elite(20, Vector{Int}[], Int[], UInt64[])
    side = nothing
    cd(ADO_SOLVER_ROOT) do
        for (d, fname) in ((1, name), (2, revname))
            dirs & d == 0 && continue
            for j in 1:13
                inp = readInputs(fname); n = maximum(t.id for t in inp.tasks)
                r = Xoshiro(seed); pols, _ = buildPolicies(inp, r, "B2")
                s = startRun(inp, pols, r; fbiMaxPasses = 1, policyKeepProb = 0.8, initPolicy = pols[j])
                push!(mk, Int(s.bestSolution.makespan)); push!(rule, j); push!(isback, d == 2)
                push!(snaps, d == 2 ? _flip(s.bestSolution, n) : _snap(s.bestSolution, n))
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
    wf = [k for k in eachindex(mk) if mk[k] == m && !isback[k]]
    wb = [k for k in eachindex(mk) if mk[k] == m && isback[k]]
    pool = isempty(wf) ? wb : wf
    w = pool[rand(Xoshiro(tieseed), 1:length(pool))]
    tie = !isempty(wf) && !isempty(wb)
    ridx = rule[w]; back = isback[w]
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
    s.schedules = 39 * count_ones(dirs)
    s.traj[1] = (iter = 0, schedules = s.schedules, makespan = m, time = 0.0)
    return s, side, e, n, ridx, back, tie
end
"""


def cmd_run(a):
    J = jl()
    for code in (JL_E1, JL_PRS, JL_PR2, JL_BI, JL_BILS, JL):
        J.seval(code)
    start = J.seval("ado_dir_start")
    run, run_ls, best = J.seval("ado_bidir_run!"), J.seval("ado_bidir_ls_run!"), J.seval("ado_best")
    seeds = canonical_seeds()
    mine = instances(a.bench)[a.shard[0]::a.shard[1]]
    OUT.mkdir(parents=True, exist_ok=True)
    tag = f"{a.bench}_shard{a.shard[0]}of{a.shard[1]}"
    with open(OUT / f"traj_{tag}.csv", "w", newline="") as ft, open(OUT / f"runs_{tag}.csv", "w", newline="") as fr, \
            gzip.open(OUT / f"bests_{tag}.csv.gz", "wt", newline="") as fb:
        wt, wr, wb = csv.writer(ft), csv.writer(fr), csv.writer(fb)
        wt.writerow(["instance", "seed_idx", "variant", "iter", "schedules", "makespan"])
        wr.writerow(["instance", "seed_idx", "variant", "init_back_wins", "init_tie", "tries_backward",
                     "tries", "ls_schedules", "ls_improvements", "cpu_s"])
        wb.writerow(["instance", "seed_idx", "variant", "makespan", "starts"])
        for inst in mine:
            rev = write_reversed(inst)
            for si in SEEDS + HOLD:
                for mode, (dirs, rule, ls, sd) in ARMS.items():
                    if si not in sd:
                        continue
                    t0 = time.process_time()
                    s, side, e, n, ridx, back, tie = start(inst, rev, seeds[si - 1], 1_000_003 * si + 7, dirs)
                    back = bool(back)
                    tb = {"fwd": False, "back": True, "winner": back, "loser": not back}.get(rule)
                    if ls:
                        it, ep, used, found = run_ls(s, side, BUDGET, e, n, int(ridx), tb, S_STALL, CAP)
                        nb = it if tb else 0
                    else:
                        it, nb = run(s, side, BUDGET, e, n, int(ridx), 0.5 if rule == "mix" else float(tb))
                        used = found = 0
                    cpu = time.process_time() - t0
                    wr.writerow([inst, si, mode, int(back), int(bool(tie)), int(nb), int(it), int(used), int(found),
                                 round(cpu, 3)])
                    mk, v = best(e)
                    wb.writerow([inst, si, mode, int(mk), " ".join(str(int(x)) for x in v)])
                    m = J.ado_rows(s, 1)
                    for k in range(m.shape[0]):
                        if int(m[k, 1]) <= BUDGET:
                            wt.writerow([inst, si, mode, int(m[k, 0]), int(m[k, 1]), int(m[k, 2])])
            ft.flush()
    print("done", tag)


def _paths(bench, kind):
    return sorted(OUT.glob(f"{kind}_{bench}_shard*of*.csv*"))


def cmd_verify(a):
    bad, n = [], {}
    for bench in ("j30", "j60", "j120"):
        if not _paths(bench, "traj"):
            continue
        trs = {m: _read_traj(_paths(bench, "traj"), m) for m in ARMS}
        n[bench] = 0
        for p in _paths(bench, "bests"):
            with gzip.open(p, "rt") as f:
                for r in csv.DictReader(f):
                    key = (r["instance"], int(r["seed_idx"]))
                    err = _feasible(r["instance"], int(r["makespan"]), r["starts"])
                    final = trs[r["variant"]][key][-1][2]
                    if err is None and final != int(r["makespan"]):
                        err = f"traj final {final} != best {r['makespan']}"
                    n[bench] += 1
                    if err:
                        bad.append([bench, *key, r["variant"], err])
    res = {"checked": n, "violations": len(bad), "first": bad[:20]}
    (OUT / "verify.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


PAIRS = (("winner", "base"), ("winner", "fwd78"), ("winner", "back78"), ("winner", "mix"),
         ("loser", "fwd78"), ("loser", "back78"), ("back78", "base"), ("fwd78", "base"),
         ("base_ls", "base"), ("winner_ls", "base"), ("winner_ls", "base_ls"), ("winner_ls", "winner"))


def _oracle(tr, insts, B, pick_seeds):
    """Per instance, the better of fwd78 / back78 (mean over pick_seeds at budget B)."""
    v = lambda t, i: mean(value_at(t[(i, s)], B) for s in pick_seeds)
    return {i: ("fwd78" if v(tr["fwd78"], i) <= v(tr["back78"], i) else "back78") for i in insts}


def cmd_analyze(a):
    opt = load_optima()
    res = {"seeds": SEEDS, "holdout_seeds": HOLD, "budget": BUDGET, "benches": {}, "checks": {}}
    tests = []
    for bench in ("j30", "j60", "j120"):
        base_vps, insts = _load(bench)
        tr = {m: _read_traj(_paths(bench, "traj"), m) for m in ARMS}
        missing = {m: sum((i, s) not in tr[m] for i in insts for s in ARMS[m][3]) for m in ARMS}
        if any(missing.values()):
            res["benches"][bench] = {"missing": missing}
            continue
        ls_vps = _read_traj(sorted(OUT_LS.glob(f"traj_{bench}_shard*of7.csv")), "ls")
        ks = [(i, s) for i in insts for s in SEEDS]
        res["checks"][f"{bench}_base_modal_equals_vps"] = [sum(tr["base"][k] == base_vps["brdeh2026"][k] for k in ks), len(ks)]
        res["checks"][f"{bench}_base_ls_modal_equals_vps_ls"] = [sum(tr["base_ls"][k] == ls_vps[k] for k in ks), len(ks)]
        lb = {i: instance_features(i)["lb_cpm"] for i in insts}
        res["checks"][f"{bench}_below_cpm_bound"] = sum(value_at(t[k], BUDGET) < lb[k[0]] for t in tr.values() for k in t)
        if bench == "j30":
            res["checks"]["j30_below_optimum"] = sum(value_at(t[k], BUDGET) < opt[k[0]] for t in tr.values() for k in t)
        ref = {i: (opt[i] if bench == "j30" else lb[i]) for i in insts}
        g = lambda t, i, B, sd=SEEDS: 100 * (mean(value_at(t[(i, s)], B) for s in sd) - ref[i]) / ref[i]
        r = {"instances": len(insts), "gap": {}, "cmp": {}, "selection": {}}
        rows = []
        for p in _paths(bench, "runs"):
            with open(p) as f:
                rows += list(csv.DictReader(f))
        rw = [x for x in rows if x["variant"] == "winner"]
        r["init_backward_wins_share"] = round(mean(int(x["init_back_wins"]) for x in rw), 3)
        r["init_tie_share"] = round(mean(int(x["init_tie"]) for x in rw), 3)
        winback = {(x["instance"], int(x["seed_idx"])): int(x["init_back_wins"]) for x in rw}
        r["cpu_s_per_run_modal"] = {m: round(mean(float(x["cpu_s"]) for x in rows if x["variant"] == m), 3) for m in ARMS}
        for B in (1000, 2500, 5000):
            honest, insample = _oracle(tr, insts, B, HOLD), _oracle(tr, insts, B, SEEDS)
            tr["oracle_honest"] = {(i, s): tr[honest[i]][(i, s)] for i in insts for s in SEEDS}
            tr["oracle_insample"] = {(i, s): tr[insample[i]][(i, s)] for i in insts for s in SEEDS}
            r["gap"][B] = {m: round(mean(g(t, i, B) for i in insts), 3) for m, t in tr.items()}
            r["cmp"][B] = {}
            for x, y in PAIRS + (("winner", "oracle_honest"),):
                r["cmp"][B][f"{x} vs {y}"] = _cmp(tr, insts, SEEDS, ref, B, x, y)
                if not y.startswith("oracle"):
                    tests.append((bench, B, f"{x} vs {y}"))
            fixed = min(r["gap"][B]["fwd78"], r["gap"][B]["back78"])
            den = fixed - r["gap"][B]["oracle_honest"]
            inter = [(g(tr["winner_ls"], i, B) - g(tr["winner"], i, B)) - (g(tr["base_ls"], i, B) - g(tr["base"], i, B))
                     for i in insts]
            r["selection"][B] = {
                "best_fixed_direction": "fwd78" if r["gap"][B]["fwd78"] <= r["gap"][B]["back78"] else "back78",
                "init_hit_rate_vs_honest_oracle": round(mean(winback[k] == (honest[k[0]] == "back78") for k in ks), 3),
                "oracle_honest_backward_share": round(mean(honest[i] == "back78" for i in insts), 3),
                "capture_of_honest_oracle": round((fixed - r["gap"][B]["winner"]) / den, 3) if den > 0 else None,
                "interaction_2x2": {"mean": round(mean(inter), 3), "ci95": _ci(inter)},
            }
        if bench == "j30":
            r["optima_hit_5000"] = {m: sum(value_at(tr[m][(i, s)], BUDGET) == opt[i] for i in insts for s in SEEDS)
                                    for m in ARMS}
        if bench == "j120":
            pilot = set(instances("j120")[::STRIDE])
            clean = [i for i in insts if i not in pilot]
            r["without_pilot"] = {"instances": len(clean), **{f"{x} vs {y} @5000": _cmp(tr, clean, SEEDS, ref, BUDGET, x, y)
                                  for x, y in (("winner", "base"), ("winner", "back78"), ("winner", "fwd78"),
                                               ("loser", "back78"), ("winner_ls", "base"))}}
        for k in ("oracle_honest", "oracle_insample"):
            tr.pop(k, None)
        res["benches"][bench] = r
    ps = [res["benches"][b]["cmp"][B][k]["p"] for b, B, k in tests]
    for (b, B, k), q in zip(tests, _holm(ps)):
        res["benches"][b]["cmp"][B][k]["p_holm"] = q
    (OUT / "summary.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "verify", "analyze"])
    ap.add_argument("bench", nargs="?", choices=["j30", "j60", "j120"])
    ap.add_argument("--shard", type=int, nargs=2, default=[0, 1])
    a = ap.parse_args()
    {"run": cmd_run, "verify": cmd_verify, "analyze": cmd_analyze}[a.cmd](a)
