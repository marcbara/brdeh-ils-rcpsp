
import argparse
import csv
import gzip
import json
import time
from statistics import mean

from experiments.a1_uniform import _read_traj
from experiments.bidir import write_reversed
from experiments.confirm_ls import _cmp, _feasible, _holm
from experiments.escoto_ideas import BUDGET, CAP, S_STALL, _setup
from experiments.escoto_ideas import OUT as ESC_OUT
from experiments.init13 import instances, value_at
from harness.evaluate import load_optima
from harness.instances import instance_features
from harness.provenance import REPO_ROOT
from harness.solvers.brdeh import canonical_seeds

OUT = REPO_ROOT / "experiments" / "runs" / "br_ils"
PERT = {"none": 0, "br": 1, "uni": 2, "grd": 3, "rm": 4, "ls": 5, "axbr": 6, "axuni": 7}
# arm -> (perturbation, rho, lam)
ARMS = {"B": ("none", 0.0, 0.0)}
for p in ("br", "uni", "grd", "rm"):
    for rho in (0.1, 0.2, 0.3):
        for lam in (0.5, 1.0):
            ARMS[f"{p}_r{int(rho * 100)}_l{int(lam * 10)}"] = (p, rho, lam)
for lam in (0.5, 1.0):
    ARMS[f"ls_l{int(lam * 10)}"] = ("ls", 0.0, lam)
# as in Arnau, Barrena, Panadero, de la Torre & Juan (EJOR 2022, BR-ILS, Algorithm 2): undo the LAST p % of the
# decisions (suffix of the order) and rebuild them with the biased heuristic; p adaptive (pmin 10 %, +10 % per
# non-improving iteration up to pmax 50 %, back to pmin after an accepted move); demon acceptance (a worse
# solution is accepted if it is worse by less than the last improvement, then demon = 0). Their paper has no
# local search inside the ILS ("ax_*"); "_ls" adds ours; "_hils" = their two-stage HILS: the 5 best distinct
# schedules found by the tries (the elite) are the base solutions, visited in turn.
AX = {"ax_br": ("axbr", 0.0, 0.0, 1), "ax_uni": ("axuni", 0.0, 0.0, 1), "ax_br_ls": ("axbr", 0.0, 1.0, 1),
      "ax_br_hils": ("axbr", 0.0, 0.0, 5), "ax_br_ls_hils": ("axbr", 0.0, 1.0, 5)}
ARMS = {k: v + (1,) for k, v in ARMS.items()}
ARMS.update(AX)
# Frozen after the dev campaigns (summary_dev.json, summary_devax.json), 2026-10-20, before any test run:
# the best dev arm on J120 at 5,000 (window 10 %, biased rebuild, L = n/2) and its same-cost controls.
# The suffix / adaptive-p / demon arms (as in Arnau et al. 2022) were clearly worse on dev and are not tested.
FROZEN = ["B", "br_r10_l5", "uni_r10_l5", "grd_r10_l5", "rm_r10_l5", "ls_l5"]
DEV_SEEDS = list(range(21, 31))      # disjoint from the test seeds 1-10 (canonical seeds 21-30)
TEST_SEEDS = list(range(1, 11))

JL = r"""
function _ils_ls!(s, e, n, gbest, cur, curmk, L::Int, target::Int, ft, rng)
    Lc = length(cur); fails = 0; found = 0; idle = 0
    while fails < L && s.schedules + 3 <= target && idle < 50 * L
        where = zeros(Int, Lc)
        for (p, id) in enumerate(cur)
            where[id] = p
        end
        pos = rand(rng, 2:Lc-1)
        lo, hi = _window(cur, pos, s.inputs, where)
        newpos = lo < hi ? rand(rng, lo:hi) : pos
        if newpos == pos
            idle += 1; continue
        end
        idle = 0
        cand = copy(cur)
        id = cand[pos]; deleteat!(cand, pos); insert!(cand, newpos, id)
        c0 = SCHEDULE_COUNT[]
        mk = decodeOrder!(s.inputs, s.availability_units, cand, ft)
        s.schedules += SCHEDULE_COUNT[] - c0
        if mk > curmk
            fails += 1; continue
        end
        mk2, st2 = _fbi_ft!(s, ft, mk, n)
        if mk2 <= curmk
            better = mk2 < curmk
            cur = _order(st2); curmk = mk2
            found += _improve!(s, gbest, e, st2, mk2)
            fails = better ? 0 : fails + 1
        else
            fails += 1
        end
    end
    return cur, curmk, found
end
function _ils_perturb(s, base, pert::Int, k::Int, rng, pol)
    Lc = length(base)
    if pert == 4                                   # rm: k random feasible insertion moves
        cur = copy(base); done = 0; tries = 0
        while done < k && tries < 50 * k
            tries += 1
            where = zeros(Int, Lc)
            for (p, id) in enumerate(cur)
                where[id] = p
            end
            pos = rand(rng, 2:Lc-1)
            lo, hi = _window(cur, pos, s.inputs, where)
            lo < hi || continue
            newpos = rand(rng, lo:hi)
            newpos == pos && continue
            id = cur[pos]; deleteat!(cur, pos); insert!(cur, newpos, id); done += 1
        end
        return cur
    end
    kk = min(k, Lc - 2)
    p0 = rand(rng, 2:(Lc - kk))                   # window base[p0 : p0+kk-1], dummies stay put
    seg = base[p0:p0+kk-1]
    out = base[1:p0-1]
    placed = falses(Lc)
    for id in out
        placed[id] = true
    end
    rest = copy(seg)
    beta = rand(rng, Float64)
    while !isempty(rest)
        el = [id for id in rest if all(placed[pid] for (pid, _) in s.inputs.tasks[id].predecessors)]
        @assert !isempty(el)
        if pert == 2
            pick = el[rand(rng, 1:length(el))]
        else
            sort!(el, by = id -> (hybridScore(pol, s.inputs.tasks[id]), id))
            pick = pert == 3 ? el[1] : el[getRandomPosition(beta, length(el), rng)]
        end
        push!(out, pick); placed[pick] = true
        deleteat!(rest, findfirst(==(pick), rest))
    end
    append!(out, base[p0+kk:end])
    return out
end
function _ils_suffix(s, base, kk::Int, rng, pol, uni::Bool)
    Lc = length(base); kk = clamp(kk, 2, Lc - 2)
    p0 = Lc - kk                                    # rebuild base[p0 : Lc-1], the last kk real activities
    out = base[1:p0-1]
    placed = falses(Lc)
    for id in out
        placed[id] = true
    end
    rest = base[p0:Lc-1]
    beta = rand(rng, Float64)
    while !isempty(rest)
        el = [id for id in rest if all(placed[pid] for (pid, _) in s.inputs.tasks[id].predecessors)]
        @assert !isempty(el)
        if uni
            pick = el[rand(rng, 1:length(el))]
        else
            sort!(el, by = id -> (hybridScore(pol, s.inputs.tasks[id]), id))
            pick = el[getRandomPosition(beta, length(el), rng)]
        end
        push!(out, pick); placed[pick] = true
        deleteat!(rest, findfirst(==(pick), rest))
    end
    push!(out, base[Lc])
    return out
end
function _ils_ax!(s, e, n, gbest, target::Int, L::Int, npool::Int, uni::Bool, rng, pol)
    ft = Vector{Int64}(undef, s.inputs.nTasks)
    nreal = length(e.st[1]) - 2
    ord = sortperm(e.mk)[1:min(npool, length(e.mk))]
    bases = [_order(e.st[i]) for i in ord]; bmk = [e.mk[i] for i in ord]
    demon = zeros(Int, length(ord)); p = fill(0.1, length(ord))
    its = 0; acc = 0; rec = 0; j = 0
    if L > 0
        for q in eachindex(bases)
            bases[q], bmk[q], f = _ils_ls!(s, e, n, gbest, bases[q], bmk[q], L, target, ft, rng); rec += f
        end
    end
    while s.schedules + 3 <= target
        its += 1; j = j % length(bases) + 1
        cand = _ils_suffix(s, bases[j], round(Int, p[j] * nreal), rng, pol, uni)
        @assert _order_ok(cand, s.inputs)
        c1 = SCHEDULE_COUNT[]
        mk = decodeOrder!(s.inputs, s.availability_units, cand, ft)
        s.schedules += SCHEDULE_COUNT[] - c1
        s.schedules + 2 <= target || break
        mk2, st2 = _fbi_ft!(s, ft, mk, n)
        rec += _improve!(s, gbest, e, st2, mk2)
        cur, curmk = _order(st2), mk2
        if L > 0
            cur, curmk, f = _ils_ls!(s, e, n, gbest, cur, curmk, L, target, ft, rng); rec += f
        end
        d = curmk - bmk[j]
        if d < 0
            bases[j], bmk[j] = cur, curmk; demon[j] = -d; p[j] = 0.0; acc += 1
        elseif d < demon[j]
            bases[j], bmk[j] = cur, curmk; demon[j] = 0; p[j] = 0.0; acc += 1
        end
        p[j] = min(p[j] + 0.1, 0.5)
    end
    return its, acc, rec
end
function ado_ils_run!(s, side, target::Int, e::Elite, n::Int, ridx::Int, back::Bool, S::Int, cap::Int,
                      pert::Int, k::Int, L::Int, ilsseed::Int, npool::Int)
    gbest = Ref(Int(s.bestSolution.makespan))
    pols = s.sortPolicies
    since = 0; episodes = 0; used = 0; found = 0
    its = 0; acc = 0; ilsrec = 0; ils_start = -1; ils_used = 0
    while s.schedules < target
        c0 = SCHEDULE_COUNT[]
        beta = rand(s.rng, Float64)
        pol = rand(s.rng, Float64) < s.policyKeepProb ? pols[ridx] : rand(s.rng, pols)
        idx = findfirst(p -> p === pol, pols)
        if back
            sol = MPJ_Heuristic(side.inputs, side.at, beta, side.pols[idx], true, s.rng)
            st0 = _flip(sol, n); mk0 = Int(sol.makespan)
            sol = localSearch(sol, side.inputs, side.au; maxPasses = 1)
            mk = Int(sol.makespan)
            st = mk < mk0 ? _flip(sol, n) : st0
        else
            sol = MPJ_Heuristic(s.inputs, s.availability_times, beta, pol, true, s.rng)
            st0 = _snap(sol, n); mk0 = Int(sol.makespan)
            sol = localSearch(sol, s.inputs, s.availability_units; maxPasses = 1)
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
            if pert == 0
                episodes += 1
                u, f = _local_search!(s, e, n, gbest, cap, target)
                used += u; found += f
                since = 0
            else
                # ILS from here to the end of the budget, own random stream
                rng = Xoshiro(ilsseed); pol1 = pols[1]
                ils_start = s.schedules
                if pert >= 6
                    its, acc, ilsrec = _ils_ax!(s, e, n, gbest, target, L, npool, pert == 7, rng, pol1)
                    ils_used = s.schedules - ils_start
                    break
                end
                ft = Vector{Int64}(undef, s.inputs.nTasks)
                i0 = sortperm(e.mk)[1]
                base = _order(e.st[i0]); basemk = e.mk[i0]
                base, basemk, f = _ils_ls!(s, e, n, gbest, base, basemk, L, target, ft, rng); ilsrec += f
                while s.schedules + 3 <= target
                    its += 1
                    if pert == 5
                        cur, curmk = base, basemk
                    else
                        cand = _ils_perturb(s, base, pert, k, rng, pol1)
                        @assert _order_ok(cand, s.inputs)
                        c1 = SCHEDULE_COUNT[]
                        mk = decodeOrder!(s.inputs, s.availability_units, cand, ft)
                        s.schedules += SCHEDULE_COUNT[] - c1
                        s.schedules + 2 <= target || break
                        mk2, st2 = _fbi_ft!(s, ft, mk, n)
                        ilsrec += _improve!(s, gbest, e, st2, mk2)
                        cur, curmk = _order(st2), mk2
                    end
                    cur, curmk, f = _ils_ls!(s, e, n, gbest, cur, curmk, L, target, ft, rng); ilsrec += f
                    if curmk <= basemk
                        acc += 1; base, basemk = cur, curmk
                    end
                end
                ils_used = s.schedules - ils_start
                break
            end
        end
    end
    @assert s.schedules <= target + 2
    @assert minimum(e.mk) == gbest[]
    return (s.iter, episodes, used, found, its, acc, ilsrec, ils_start, ils_used, s.schedules - target)
end
"""


def _setup_ils():
    J = _setup()
    J.seval(JL)
    return J


def _y(inst):
    return int(inst.rsplit("_", 1)[1].split(".")[0])


def _scope(bench):
    """dev: J60 + J120, instances _1 and _2 of every parameter class, seeds 21-30 (tuning only).
    '<bench>test': J60/J120 instances _3.._10 (J30: all), seeds 1-10. 'smoke': 4 J120 dev instances, seeds 21-22."""
    if bench in ("dev", "devax"):
        return [i for b in ("j60", "j120") for i in instances(b) if _y(i) <= 2], DEV_SEEDS
    if bench == "smoke":
        return [i for i in instances("j120") if _y(i) <= 2][::30][:4], DEV_SEEDS[:2]
    if bench.endswith("full"):                   # video table: every project, the frozen ILS only (B = I_hyb of <j>v5)
        return instances(bench[:-4]), TEST_SEEDS
    if bench.endswith("test"):
        b = bench[:-4]
        return [i for i in instances(b) if b == "j30" or _y(i) >= 3], TEST_SEEDS
    raise SystemExit(f"bench {bench}")


def run_one(J, fns, inst, rev, si, arm, budget=BUDGET):
    pert, rho, lam, npool = ARMS[arm]
    seeds = canonical_seeds()
    start, run = fns
    s, side, e, n, ridx, back, tie = start(inst, rev, seeds[si - 1], 1_000_003 * si + 7, 3, "hyb")
    nreal = int(n) - 2
    k = max(2, round(rho * nreal)) if pert in ("br", "uni", "grd", "rm") else 0
    L = max(1, round(lam * int(n))) if lam > 0 else 0
    st = run(s, side, budget, e, n, int(ridx), bool(back), S_STALL, CAP, PERT[pert], k, L, 7919 * seeds[si - 1] + 13, npool)
    return s, e, [int(x) for x in st]


def _arms(a):
    if a.arms:
        return a.arms.split(",")
    if a.bench == "devax":
        return list(AX)
    if a.bench.endswith("test"):
        return list(FROZEN)
    if a.bench.endswith("full"):
        return ["br_r10_l5"]
    return [m for m in ARMS if m not in AX]


def cmd_run(a):
    J = _setup_ils()
    fns = (J.seval("ado_gen_start"), J.seval("ado_ils_run!"))
    insts, seeds = _scope(a.bench)
    arms = _arms(a)
    mine = insts[a.shard[0]::a.shard[1]]
    OUT.mkdir(parents=True, exist_ok=True)
    tag = f"{a.bench}_shard{a.shard[0]}of{a.shard[1]}"
    with open(OUT / f"traj_{tag}.csv", "w", newline="") as ft, open(OUT / f"runs_{tag}.csv", "w", newline="") as fr, \
            gzip.open(OUT / f"bests_{tag}.csv.gz", "wt", newline="") as fb:
        wt, wr, wb = csv.writer(ft), csv.writer(fr), csv.writer(fb)
        wt.writerow(["instance", "seed_idx", "variant", "iter", "schedules", "makespan"])
        wr.writerow(["instance", "seed_idx", "variant", "tries", "episodes", "ls_schedules", "ls_records", "ils_iters",
                     "ils_accepted", "ils_records", "ils_start", "ils_schedules", "overshoot", "cpu_s"])
        wb.writerow(["instance", "seed_idx", "variant", "makespan", "starts"])
        for inst in mine:
            rev = write_reversed(inst)
            for si in seeds:
                for arm in arms:
                    t0 = time.process_time()
                    s, e, st = run_one(J, fns, inst, rev, si, arm)
                    wr.writerow([inst, si, arm] + st + [round(time.process_time() - t0, 3)])
                    mk, v = J.ado_best(e)
                    wb.writerow([inst, si, arm, int(mk), " ".join(str(int(x)) for x in v)])
                    m = J.ado_rows(s, 1)
                    for r in range(m.shape[0]):
                        if int(m[r, 1]) <= BUDGET:
                            wt.writerow([inst, si, arm, int(m[r, 0]), int(m[r, 1]), int(m[r, 2])])
            ft.flush()
    print("done", tag)


def _rows(J, s):
    m = J.ado_rows(s, 1)
    return [(int(m[k, 0]), int(m[k, 1]), int(m[k, 2])) for k in range(m.shape[0]) if int(m[k, 1]) <= BUDGET]


def cmd_check(a):
    """Known answers: B == I_hyb of the confirmed escoto_ideas campaign (seeds 1-2, J30/J60/J120 samples);
    every ILS arm == B up to its first stall; feasibility of every arm's best; budget never exceeded by > 2."""
    J = _setup_ils()
    fns = (J.seval("ado_gen_start"), J.seval("ado_ils_run!"))
    res = {}
    ok = tot = 0
    for bench, k in (("j30", 3), ("j60", 3), ("j120", 3)):
        insts = instances(bench)[::max(1, len(instances(bench)) // k)][:k]
        ref = _read_traj(sorted(ESC_OUT.glob(f"traj_{bench}v5_shard*of*.csv")), "I_hyb")
        for inst in insts:
            rev = write_reversed(inst)
            for si in (1, 2):
                s, *_ = run_one(J, fns, inst, rev, si, "B")
                tot += 1; ok += _rows(J, s) == [tuple(x) for x in ref[(inst, si)]]
    res["B_equals_I_hyb"] = [ok, tot]
    inst = instances("j120")[7]; rev = write_reversed(inst)
    sb, _, stb = run_one(J, fns, inst, rev, 1, "B")
    rb = _rows(J, sb)
    same, bad, over = [], [], []
    for arm in ARMS:
        if arm == "B":
            continue
        s, e, st = run_one(J, fns, inst, rev, 1, arm)
        r = _rows(J, s)
        cut = st[7]                                      # ils_start
        same.append(int([x for x in r if x[1] <= cut] == [x for x in rb if x[1] <= cut]))
        mk, v = J.ado_best(e)
        err = _feasible(inst, int(mk), " ".join(str(int(x)) for x in v))
        if err:
            bad.append([arm, err])
        over.append(st[-1])
        if st[4] == 0:
            bad.append([arm, "no ILS iteration"])
    res["arms_equal_B_before_first_stall"] = [sum(same), len(same)]
    res["feasibility_or_inert_violations"] = bad
    res["max_overshoot"] = max(over)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "check.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


def _paths(bench, kind):
    return sorted(OUT.glob(f"{kind}_{bench}_shard*of*.csv*"))


def cmd_verify(a):
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
    insts, seeds = _scope(a.bench)
    opt = load_optima()
    ref = {i: (opt[i] if i.startswith("j30") else instance_features(i)["lb_cpm"]) for i in insts}
    tr = {m: _read_traj(_paths(a.bench, "traj"), m) for m in ARMS}
    tr = {m: t for m, t in tr.items() if t}
    if a.bench == "devax":                       # the reference B comes from the dev campaign (same instances, seeds)
        tr["B"] = _read_traj(_paths("dev", "traj"), "B")
    rows = []
    for p in _paths(a.bench, "runs"):
        with open(p) as f:
            rows += list(csv.DictReader(f))
    res = {"bench": a.bench, "n_instances": len(insts), "seeds": seeds, "by_set": {}, "mech": {}}
    for b in sorted({i.split("/")[0] for i in insts}):
        ib = [i for i in insts if i.startswith(b)]
        d = {"n": len(ib), "gap": {}, "cmp": {}}
        for B in (1000, 2500, 5000):
            d["gap"][B] = {m: round(mean(100 * (mean(value_at(t[(i, s)], B) for s in seeds) - ref[i]) / ref[i]
                                         for i in ib), 3) for m, t in tr.items()}
            c = {f"{m} vs B": _cmp(tr, ib, seeds, ref, B, m, "B") for m in tr if m != "B"}
            for (k, v), ph in zip(c.items(), _holm([v["p"] for v in c.values()])):
                v["p_holm"] = ph
            d["cmp"][B] = c
        res["by_set"][b] = d
    for m in tr:
        rr = [r for r in rows if r["variant"] == m]
        if not rr:
            continue
        f = lambda k: round(mean(float(r[k]) for r in rr), 3)
        res["mech"][m] = {k: f(k) for k in ("tries", "ls_records", "ils_iters", "ils_accepted", "ils_records",
                                              "ils_schedules", "cpu_s")}
        res["mech"][m]["max_overshoot"] = max(int(r["overshoot"]) for r in rr)
    (OUT / f"summary_{a.bench}.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({b: {"5000": d["gap"][5000]} for b, d in res["by_set"].items()}, indent=1)[:4000])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "check", "verify", "analyze"])
    ap.add_argument("bench", nargs="?", default="smoke")
    ap.add_argument("--shard", type=int, nargs=2, default=[0, 1])
    ap.add_argument("--arms", default="")
    a = ap.parse_args()
    {"run": cmd_run, "check": cmd_check, "verify": cmd_verify, "analyze": cmd_analyze}[a.cmd](a)
