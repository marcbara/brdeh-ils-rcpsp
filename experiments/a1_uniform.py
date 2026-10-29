
import argparse
import csv
import gzip
import json
from collections import defaultdict
from statistics import mean

from experiments.init13 import BUDGET, RULES, instances, value_at
from harness.instances import instance_features
from harness.provenance import REPO_ROOT
from harness.solvers.brdeh import canonical_seeds, jl

OUT = REPO_ROOT / "experiments" / "runs" / "a1_uniform"
BASE = REPO_ROOT / "experiments" / "runs" / "variants"

JL = r"""
function ado_init13_start_mk(name::String, seed::Int, tieseed::Int)
    mk = Float64[]
    cd(ADO_SOLVER_ROOT) do
        for k in 1:13
            inp = readInputs(name)
            r = Xoshiro(seed)
            pols, _ = buildPolicies(inp, r, "B2")
            s = startRun(inp, pols, r; fbiMaxPasses = 1, policyKeepProb = 0.8, initPolicy = pols[k])
            @assert s.schedules == 3
            push!(mk, s.bestSolution.makespan)
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
    return s, w, length(winners), mk
end
function ado_run_a1!(s, target::Int)
    rows = NTuple{6,Float64}[]
    n = 0
    while s.schedules < target
        n += 1
        c0 = SCHEDULE_COUNT[]
        beta = rand(s.rng, Float64)
        k = rand(s.rng, 1:length(s.sortPolicies))
        sol = MPJ_Heuristic(s.inputs, s.availability_times, beta, s.sortPolicies[k], true, s.rng)
        mb = sol.makespan
        sol = localSearch(sol, s.inputs, s.availability_units; maxPasses = s.fbiMaxPasses)
        s.schedules += SCHEDULE_COUNT[] - c0
        s.iter += 1
        if sol.makespan < s.bestSolution.makespan
            s.bestBeta = beta
            s.bestPolicy = s.sortPolicies[k]
            s.bestSolution = sol
            push!(s.traj, (iter = s.iter, schedules = s.schedules,
                           makespan = sol.makespan, time = 0.0))
        end
        push!(rows, (n, s.schedules, k, beta, mb, sol.makespan))
    end
    M = Matrix{Float64}(undef, length(rows), 6)
    for (i, r) in enumerate(rows), j in 1:6
        M[i, j] = r[j]
    end
    return M
end
"""


def cmd_run(a):
    J = jl()
    J.seval(JL)
    start = J.seval("ado_init13_start_mk")
    run = J.seval("ado_run_a1!")
    seeds = canonical_seeds()
    mine = instances(a.bench)[a.shard[0]::a.shard[1]]
    OUT.mkdir(parents=True, exist_ok=True)
    tag = f"{a.bench}_s{'-'.join(map(str, a.seeds))}_shard{a.shard[0]}of{a.shard[1]}"
    with open(OUT / f"traj_{tag}.csv", "w", newline="") as ft, \
            gzip.open(OUT / f"tries_{tag}.csv.gz", "wt", newline="") as fl, \
            open(OUT / f"init_{tag}.csv", "w", newline="") as fi:
        wt, wl, wi = csv.writer(ft), csv.writer(fl), csv.writer(fi)
        wt.writerow(["instance", "seed_idx", "iter", "schedules", "makespan"])
        wl.writerow(["instance", "seed_idx", "try", "schedules", "rule", "beta", "mk_built", "mk_final"])
        wi.writerow(["instance", "seed_idx", "winner", "n_tied"] + RULES)
        for c, inst in enumerate(mine, 1):
            for si in a.seeds:
                s, w, nt, mk = start(inst, seeds[si - 1], 1_000_003 * si + 7)
                wi.writerow([inst, si, RULES[int(w) - 1], int(nt)] + [int(x) for x in mk])
                M = run(s, BUDGET)
                for r in range(M.shape[0]):
                    wl.writerow([inst, si, int(M[r, 0]), int(M[r, 1]), RULES[int(M[r, 2]) - 1],
                                 round(float(M[r, 3]), 4), int(M[r, 4]), int(M[r, 5])])
                m = J.ado_rows(s, 1)
                for k in range(m.shape[0]):
                    if int(m[k, 1]) <= BUDGET:
                        wt.writerow([inst, si, int(m[k, 0]), int(m[k, 1]), int(m[k, 2])])
            if c % 25 == 0:
                print(f"{tag}: {c}/{len(mine)}", flush=True)
    print("done", tag)


def _read_traj(paths, variant=None):
    tr = defaultdict(list)
    for p in paths:
        with open(p) as f:
            for r in csv.DictReader(f):
                if variant and r["variant"] != variant:
                    continue
                tr[(r["instance"], int(r["seed_idx"]))].append(
                    (int(r["iter"]), int(r["schedules"]), int(r["makespan"])))
    return tr


def _spearman(x, y):
    from scipy.stats import spearmanr
    if len(set(x)) < 2 or len(set(y)) < 2:
        return None
    return float(spearmanr(x, y).statistic)


def _rates(tries, frac=0.10):
    """Per rule: share of its tries whose final makespan is in the run's top `frac`."""
    if not tries:
        return {}
    cut = sorted(t[1] for t in tries)[max(0, int(len(tries) * frac) - 1)]
    n, top = defaultdict(int), defaultdict(int)
    for rule, mk in tries:
        n[rule] += 1
        top[rule] += mk <= cut
    return {r: top[r] / n[r] for r in n}


def cmd_analyze(a):
    from scipy.stats import wilcoxon
    tag = f"{a.bench}_s{'-'.join(map(str, a.seeds))}"
    a1 = _read_traj(sorted(OUT.glob(f"traj_{tag}_shard*.csv")))
    base = _read_traj(sorted(BASE.glob(f"traj_{tag}_shard*.csv")), "init13_nobeta")
    b2 = _read_traj(sorted(BASE.glob(f"traj_{tag}_shard*.csv")), "b2")
    insts = sorted({k[0] for k in a1})
    assert all(a1[(i, s)][0] == base[(i, s)][0] for i in insts for s in a.seeds), "init differs from base"
    ref = {i: instance_features(i)["lb_cpm"] for i in insts}
    val = lambda tr, i, B: mean(value_at(tr[(i, s)], B) for s in a.seeds)
    res = {"bench": a.bench, "seeds": a.seeds, "n_instances": len(insts),
           "checks": {"init_equals_base": True}, "gap": {}, "vs_base": {}, "vs_b2": {}}
    for B in (39, 250, 1000, 2500, 5000):
        res["gap"][B] = {n: mean(100 * (val(t, i, B) - ref[i]) / ref[i] for i in insts)
                         for n, t in (("b2", b2), ("base", base), ("a1", a1))}

    def cmp(x, y, B):
        d = [val(x, i, B) - val(y, i, B) for i in insts]
        nz = [t for t in d if t != 0]
        return {"a1_better": sum(t < 0 for t in d), "tie": sum(t == 0 for t in d), "a1_worse": sum(t > 0 for t in d),
                "wilcoxon_p": float(wilcoxon(nz).pvalue)}
    for B in (1000, 2500, 5000):
        res["vs_base"][B] = cmp(a1, base, B)
        res["vs_b2"][B] = cmp(a1, b2, B)

    # ---- try log ----
    init = {}
    for p in sorted(OUT.glob(f"init_{tag}_shard*.csv")):
        with open(p) as f:
            for r in csv.DictReader(f):
                init[(r["instance"], int(r["seed_idx"]))] = {k: int(r[k]) for k in RULES}
    runs = defaultdict(list)
    for p in sorted(OUT.glob(f"tries_{tag}_shard*.csv.gz")):
        with gzip.open(p, "rt") as f:
            for r in csv.DictReader(f):
                runs[(r["instance"], int(r["seed_idx"]))].append(
                    (r["rule"], float(r["beta"]), int(r["mk_built"]), int(r["mk_final"])))
    rho_init, rho_halves, rho_oddeven, winner_hit = [], [], [], []
    for key, tr in runs.items():
        rates = _rates([(t[0], t[3]) for t in tr])
        rules = sorted(rates)
        # (a) deterministic init makespan (lower = better) vs randomized top-10% rate
        rho = _spearman([-init[key][r] for r in rules], [rates[r] for r in rules])
        if rho is not None:
            rho_init.append(rho)
        best_init = min(init[key].values())
        top_rate = max(rates.values())
        winner_hit.append(any(init[key][r] == best_init and rates[r] == top_rate for r in rules))
        # (b) stationarity: first vs second half, against odd vs even tries as noise reference
        h = len(tr) // 2
        r1, r2 = _rates([(t[0], t[3]) for t in tr[:h]]), _rates([(t[0], t[3]) for t in tr[h:]])
        ro, re_ = _rates([(t[0], t[3]) for t in tr[0::2]]), _rates([(t[0], t[3]) for t in tr[1::2]])
        com = sorted(set(r1) & set(r2))
        x = _spearman([r1[r] for r in com], [r2[r] for r in com])
        com2 = sorted(set(ro) & set(re_))
        y = _spearman([ro[r] for r in com2], [re_[r] for r in com2])
        if x is not None and y is not None:
            rho_halves.append(x)
            rho_oddeven.append(y)
    # (c) share of built-makespan variance explained by the rule, per beta bin (within run)
    bins = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.01)]
    eta = {f"{lo:.1f}-{min(hi, 1):.1f}": [] for lo, hi in bins}
    for tr in runs.values():
        for lo, hi in bins:
            sel = [(t[0], t[2]) for t in tr if lo <= t[1] < hi]
            if len(sel) < 30:
                continue
            m = mean(v for _, v in sel)
            sst = sum((v - m) ** 2 for _, v in sel)
            if sst == 0:
                continue
            g = defaultdict(list)
            for rule, v in sel:
                g[rule].append(v)
            ssb = sum(len(v) * (mean(v) - m) ** 2 for v in g.values())
            eta[f"{lo:.1f}-{min(hi, 1):.1f}"].append(ssb / sst)
    res["tries"] = {
        "runs": len(runs), "mean_tries_per_run": mean(len(t) for t in runs.values()),
        "a_spearman_init_vs_random_rate": mean(rho_init),
        "a_init_winner_is_top_rate_rule_share": mean(winner_hit),
        "b_spearman_first_vs_second_half": mean(rho_halves),
        "b_spearman_odd_vs_even_noise_ref": mean(rho_oddeven),
        "c_eta2_rule_by_beta_bin": {k: (mean(v) if v else None) for k, v in eta.items()},
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
