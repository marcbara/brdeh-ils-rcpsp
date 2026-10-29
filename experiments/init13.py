
import argparse
import csv
import gzip
import json
from statistics import mean

from harness.evaluate import load_optima
from harness.instances import instance_features
from harness.provenance import REPO_ROOT
from harness.solvers.brdeh import canonical_seeds, jl

BUDGET = 5000
OUT = REPO_ROOT / "experiments" / "runs" / "init13"
ARCH = {
    "j30": REPO_ROOT / "solvers" / "br-deh" / "experiments" / "results" / "full_B2.csv.gz",
    "j60": REPO_ROOT / "experiments" / "archive" / "j60_B2_50000.csv.gz",
    "j120": REPO_ROOT / "experiments" / "archive" / "j120_B2_50000.csv.gz",
}

JL = r"""
function ado_init13_start(name::String, seed::Int, tieseed::Int)
    mk = Float64[]
    cd(ADO_SOLVER_ROOT) do
        for k in 1:13
            inp = readInputs(name)          # fresh tasks per rule (Tasks are mutated)
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
        rng = Xoshiro(seed)                 # same stream position as a plain B2 start
        pols, _ = buildPolicies(inputs, rng, "B2")
        startRun(inputs, pols, rng; fbiMaxPasses = 1, policyKeepProb = 0.8, initPolicy = pols[w])
    end
    @assert s.bestSolution.makespan == m
    s.schedules = 39                        # 13 deterministic builds x 3 schedules
    s.traj[1] = (iter = 0, schedules = 39, makespan = m, time = 0.0)
    return s, w, length(winners)
end
"""
RULES = ["SAD", "MIS", "MTS", "HRPW", "HRPW*", "EFT", "EFTD", "LST", "LFT", "STFS", "STFD", "HRD", "HWRU"]


INSTANCE_LISTS = REPO_ROOT / "experiments" / "instance_lists.json"   # archive order, for machines without archives


def instances(bench):
    if not ARCH[bench].exists() and INSTANCE_LISTS.exists():
        import json
        return json.loads(INSTANCE_LISTS.read_text())[bench]
    seen = []
    with gzip.open(ARCH[bench], "rt") as f:
        for r in csv.DictReader(f):
            if not seen or seen[-1] != r["instance"]:
                if r["instance"] not in seen:
                    seen.append(r["instance"])
    return seen


def archive_rows(bench, seeds, insts=None):
    out = {}
    with gzip.open(ARCH[bench], "rt") as f:
        for r in csv.DictReader(f):
            s = int(r["seed_idx"])
            if s in seeds and (insts is None or r["instance"] in insts) and int(r["schedules"]) <= BUDGET:
                out.setdefault((r["instance"], s), []).append(
                    (int(r["iter"]), int(r["schedules"]), int(float(r["makespan"]))))
    return out


def rows_of(J, s):
    m = J.ado_rows(s, 1)
    return [(int(m[k, 0]), int(m[k, 1]), int(m[k, 2])) for k in range(m.shape[0])]


def cmd_run(a):
    J = jl()
    J.seval(JL)
    start = J.seval("ado_init13_start")
    adv = J.seval("ado_advance!")
    seeds = canonical_seeds()
    insts = instances(a.bench)
    i0, n = a.shard
    mine = insts[i0::n]
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"init13_{a.bench}_s{'-'.join(map(str, a.seeds))}_shard{i0}of{n}.csv"
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["instance", "seed_idx", "iter", "schedules", "makespan", "winner_rule", "n_tied"])
        for k, inst in enumerate(mine, 1):
            for si in a.seeds:
                s, win, nt = start(inst, seeds[si - 1], 1_000_003 * si + 7)
                adv(s, BUDGET - 39)
                for it, sch, mk in rows_of(J, s):
                    if sch <= BUDGET:
                        w.writerow([inst, si, it, sch, mk, RULES[int(win) - 1], int(nt)])
            if k % 25 == 0:
                print(f"shard {i0}: {k}/{len(mine)}", flush=True)
    print("done", path)


def cmd_sanity(a):
    J = jl()
    adv = J.seval("ado_advance!")
    seeds = canonical_seeds()
    insts = instances(a.bench)[:: max(1, len(instances(a.bench)) // a.n)][: a.n]
    arch = archive_rows(a.bench, {1}, set(insts))
    bad = 0
    for inst in insts:
        s = J.ado_start(inst, seeds[0], True, 1, 0.8)
        adv(s, BUDGET - 3)
        mine = [r for r in rows_of(J, s) if r[1] <= BUDGET]
        if mine != arch[(inst, 1)]:
            bad += 1
            print("MISMATCH", inst, mine[:5], arch[(inst, 1)][:5])
    print(f"sanity {a.bench}: {len(insts) - bad}/{len(insts)} identical to archive")
    assert bad == 0


def value_at(rows, B):
    return min(mk for _, s, mk in rows if s <= B)


def cmd_analyze(a):
    from scipy.stats import wilcoxon
    new = {}
    for p in sorted(OUT.glob(f"init13_{a.bench}_s{'-'.join(map(str, a.seeds))}_shard*.csv")):
        with open(p) as f:
            for r in csv.DictReader(f):
                k = (r["instance"], int(r["seed_idx"]))
                new.setdefault(k, {"rows": [], "rule": r["winner_rule"], "tied": int(r["n_tied"])})
                new[k]["rows"].append((int(r["iter"]), int(r["schedules"]), int(r["makespan"])))
    insts = sorted({k[0] for k in new})
    arch = archive_rows(a.bench, set(a.seeds), set(insts))
    assert all((i, s) in new and (i, s) in arch for i in insts for s in a.seeds), "incomplete"
    ref = load_optima() if a.bench == "j30" else {i: instance_features(i)["lb_cpm"] for i in insts}
    gap = lambda mk, i: 100.0 * (mk - ref[i]) / ref[i]
    res = {"bench": a.bench, "seeds": a.seeds, "n_instances": len(insts),
           "reference": "optima" if a.bench == "j30" else "lb_cpm", "budgets": {}}
    for B in (39, 100, 250, 500, 1000, 2500, 5000):
        ia = {i: mean(value_at(new[(i, s)]["rows"], B) for s in a.seeds) for i in insts}
        ib = {i: mean(value_at(arch[(i, s)], B) for s in a.seeds) for i in insts}
        d = [ia[i] - ib[i] for i in insts]
        nz = [x for x in d if x != 0]
        res["budgets"][B] = {
            "gap_init13": mean(gap(ia[i], i) for i in insts),
            "gap_B2": mean(gap(ib[i], i) for i in insts),
            "init13_better": sum(x < 0 for x in d), "tie": sum(x == 0 for x in d), "B2_better": sum(x > 0 for x in d),
            "wilcoxon_p": float(wilcoxon(nz).pvalue) if len(nz) > 10 else None,
        }
    from collections import Counter
    res["initial_rule"] = dict(Counter(v["rule"] for v in new.values()).most_common())
    res["mean_tied_rules"] = mean(v["tied"] for v in new.values())
    (OUT / f"summary_{a.bench}_s{'-'.join(map(str, a.seeds))}.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "sanity", "analyze"])
    ap.add_argument("bench")
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--shard", type=int, nargs=2, default=[0, 1])
    ap.add_argument("--n", type=int, default=20)
    a = ap.parse_args()
    {"run": cmd_run, "sanity": cmd_sanity, "analyze": cmd_analyze}[a.cmd](a)
