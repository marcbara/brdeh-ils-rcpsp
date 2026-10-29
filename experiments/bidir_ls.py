
import argparse
import csv
import gzip
import json
from statistics import mean

from experiments.a1_uniform import _read_traj
from experiments.bidir import CHECK_N, JL as JL_BI, write_reversed
from experiments.confirm_ls import _cmp, _feasible
from experiments.e1_learned import JL as JL_E1
from experiments.init13 import BUDGET, instances, value_at
from experiments.pr2_stagnation import JL as JL_PR2
from experiments.pr_stagnation import CAP, JL as JL_PRS, S_STALL, SEEDS, STRIDE
from harness.instances import instance_features
from harness.provenance import REPO_ROOT
from harness.solvers.brdeh import canonical_seeds, jl

OUT = REPO_ROOT / "experiments" / "runs" / "bidir_ls"
RUNS = REPO_ROOT / "experiments" / "runs"

JL = r"""
function ado_bidir_ls_run!(s, side, target::Int, e::Elite, n::Int, ridx::Int, back::Bool, S::Int, cap::Int)
    gbest = Ref(Int(s.bestSolution.makespan))
    pols = s.sortPolicies
    since = 0; episodes = 0; used = 0; found = 0
    while s.schedules < target
        c0 = SCHEDULE_COUNT[]
        beta = rand(s.rng, Float64)
        pol = rand(s.rng, Float64) < s.policyKeepProb ? pols[ridx] : rand(s.rng, pols)
        idx = findfirst(p -> p === pol, pols)
        if back
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
        spent = SCHEDULE_COUNT[] - c0
        s.schedules += spent
        s.iter += 1
        if mk < gbest[]
            gbest[] = mk; ridx = idx; since = 0
            _offer!(e, st, mk)
            push!(s.traj, (iter = s.iter, schedules = s.schedules, makespan = Float64(mk), time = 0.0))
        else
            since += spent
        end
        if since >= S && s.schedules < target
            episodes += 1
            u, f = _local_search!(s, e, n, gbest, cap, target)
            used += u; found += f
            since = 0
        end
    end
    @assert s.schedules <= target + 2
    @assert minimum(e.mk) == gbest[]
    return (s.iter, episodes, used, found)
end
"""


def cmd_run(a):
    J = jl()
    for code in (JL_E1, JL_PRS, JL_PR2, JL_BI, JL):
        J.seval(code)
    start, run, best = J.seval("ado_bidir_start"), J.seval("ado_bidir_ls_run!"), J.seval("ado_best")
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
        wr.writerow(["instance", "seed_idx", "variant", "init_back_wins", "tries", "episodes", "ls_schedules", "ls_improvements"])
        wb.writerow(["instance", "seed_idx", "variant", "makespan", "starts"])
        for inst in mine:
            rev = write_reversed(inst)
            for si in SEEDS:
                for mode in ("bi_dir_ls",) + (("check",) if inst in check else ()):
                    s, side, e, n, ridx, back = start(inst, rev, seeds[si - 1], 1_000_003 * si + 7, True)
                    st = run(s, side, BUDGET, e, n, int(ridx), bool(back), S_STALL if mode == "bi_dir_ls" else 10 ** 9, CAP)
                    wr.writerow([inst, si, mode, int(bool(back))] + [int(x) for x in st])
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
    tr = {m: _read_traj(paths, m) for m in ("bi_dir_ls", "check")}
    tr["bi_dir"] = _read_traj(sorted((RUNS / "bidir").glob("traj_shard*.csv")), "bi_dir")
    tr["ls"] = _read_traj(sorted((RUNS / "pr2_stagnation").glob("traj_ls_shard*of7.csv")), "ls")
    tr["brdeh2026"] = _read_traj(sorted((RUNS / "variants").glob("traj_j120_s1-2-3_shard*.csv")), "init13_nobeta")
    insts = sorted({k[0] for k in tr["bi_dir_ls"]})
    ref = {i: instance_features(i)["lb_cpm"] for i in insts}
    res = {"n_instances": len(insts), "seeds": SEEDS, "checks": {}, "gap": {}, "cmp": {}}
    res["checks"]["trigger_off_equals_bi_dir"] = [sum(tr["bi_dir"][k] == v for k, v in tr["check"].items()), len(tr["check"])]
    bad = n = 0
    for p in sorted(OUT.glob("bests_shard*.csv.gz")):
        with gzip.open(p, "rt") as f:
            for r in csv.DictReader(f):
                n += 1
                err = _feasible(r["instance"], int(r["makespan"]), r["starts"])
                fin = tr[r["variant"]][(r["instance"], int(r["seed_idx"]))][-1][2]
                bad += bool(err) or fin != int(r["makespan"])
    res["checks"]["best_schedules_checked"], res["checks"]["violations"] = n, bad
    for B in (1000, 2500, 5000):
        res["gap"][B] = {v: round(mean(100 * (mean(value_at(t[(i, s)], B) for s in SEEDS) - ref[i]) / ref[i]
                                       for i in insts), 3) for v, t in tr.items() if v != "check"}
        res["cmp"][B] = {f"{x} vs {y}": _cmp(tr, insts, SEEDS, ref, B, x, y)
                         for x, y in (("bi_dir_ls", "brdeh2026"), ("bi_dir_ls", "bi_dir"), ("bi_dir_ls", "ls"))}
    rows = []
    for p in sorted(OUT.glob("runs_shard*.csv")):
        with open(p) as f:
            rows += [r for r in csv.DictReader(f) if r["variant"] == "bi_dir_ls"]
    res["ls_share"] = round(sum(int(r["ls_schedules"]) for r in rows) / (len(rows) * BUDGET), 3)
    res["runs_with_ls_improvement"] = round(mean(int(r["ls_improvements"]) > 0 for r in rows), 3)
    (OUT / "summary.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "analyze"])
    ap.add_argument("--shard", type=int, nargs=2, default=[0, 1])
    a = ap.parse_args()
    {"run": cmd_run, "analyze": cmd_analyze}[a.cmd](a)
