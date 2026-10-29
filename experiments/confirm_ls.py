
import argparse
import csv
import gzip
import json
import time
from statistics import mean

from experiments.a1_uniform import _read_traj
from experiments.confirm import SEEDS, _load
from experiments.e1_learned import JL as JL_E1
from experiments.init13 import BUDGET, instances, value_at
from experiments.pr2_stagnation import JL as JL_PR2
from experiments.pr_stagnation import CAP, JL as JL_PRS, K_ELITE, N_ORIG, S_STALL, STRIDE
from harness.evaluate import load_optima
from harness.instances import INPUTS, instance_features, parse_sm
from harness.provenance import REPO_ROOT
from harness.solvers.brdeh import canonical_seeds, jl

OUT = REPO_ROOT / "experiments" / "runs" / "confirm_ls"
RUNS = REPO_ROOT / "experiments" / "runs"
MODES = ("ls", "prfbi", "restart", "ls_periodic")
PERIOD = 557                 # 300 * 0.65 / 0.35: ~35 % of the budget, the pilot share
OFF_SAMPLE = 20              # instances per benchmark for the trigger-disabled check
JL_BEST = r"""
ado_best(e) = (i = argmin(e.mk); (e.mk[i], e.st[i]))
"""


def off_sample(bench):
    allx = instances(bench)
    return allx[::len(allx) // OFF_SAMPLE][:OFF_SAMPLE]


def cmd_run(a):
    J = jl()
    for code in (JL_E1, JL_PRS, JL_PR2, JL_BEST):
        J.seval(code)
    start = J.seval("ado_prs_start")
    run1, run2, best = J.seval("ado_prs_run!"), J.seval("ado_prs2_run!"), J.seval("ado_best")
    seeds = canonical_seeds()
    mine = instances(a.bench)[a.shard[0]::a.shard[1]]
    off = set(off_sample(a.bench))
    OUT.mkdir(parents=True, exist_ok=True)
    tag = f"{a.bench}_shard{a.shard[0]}of{a.shard[1]}"
    with open(OUT / f"traj_{tag}.csv", "w", newline="") as ft, \
            open(OUT / f"runs_{tag}.csv", "w", newline="") as fr, \
            gzip.open(OUT / f"bests_{tag}.csv.gz", "wt", newline="") as fb:
        wt, wr, wb = csv.writer(ft), csv.writer(fr), csv.writer(fb)
        wt.writerow(["instance", "seed_idx", "variant", "iter", "schedules", "makespan"])
        wr.writerow(["instance", "seed_idx", "variant", "tries", "episodes", "shake_schedules",
                     "shake_improvements", "schedules_total", "cpu_s"])
        wb.writerow(["instance", "seed_idx", "variant", "makespan", "starts"])
        for inst in mine:
            for si in SEEDS:
                for mode in MODES + (("ls_off",) if inst in off else ()):
                    t0 = time.process_time()
                    s, w, e, n = start(inst, seeds[si - 1], 1_000_003 * si + 7, K_ELITE)
                    if mode == "restart":
                        st = run1(s, BUDGET, 2, int(w), e, n, S_STALL, CAP, N_ORIG, 424_242 + 1009 * si)
                    else:
                        m, S = {"prfbi": (3, S_STALL), "ls": (4, S_STALL), "ls_periodic": (5, PERIOD),
                                "ls_off": (4, 10 ** 9)}[mode]
                        st = run2(s, BUDGET, m, e, n, S, CAP, N_ORIG)
                        mk, v = best(e)
                        wb.writerow([inst, si, mode, int(mk), " ".join(str(int(x)) for x in v)])
                    cpu = time.process_time() - t0
                    wr.writerow([inst, si, mode] + [int(x) for x in st] + [int(s.schedules), round(cpu, 3)])
                    rows = J.ado_rows(s, 1)
                    for k in range(rows.shape[0]):
                        if int(rows[k, 1]) <= BUDGET:
                            wt.writerow([inst, si, mode, int(rows[k, 0]), int(rows[k, 1]), int(rows[k, 2])])
            ft.flush(); fr.flush()
    print("done", tag)


def _feasible(inst, mk, starts):
    """Independent check (Python, from the .sm file): precedences, capacities, makespan."""
    s = parse_sm(INPUTS / inst, inst)
    n, k = len(s.durations), len(s.capacities)
    st = [int(x) for x in starts.split()]
    if len(st) != n or min(st) < 0:
        return "length/negative"
    for i in range(n):
        for j in s.successors[i]:
            if st[j] < st[i] + s.durations[i]:
                return f"precedence {i}->{j}"
    end = max(st[i] + s.durations[i] for i in range(n))
    if end != mk:
        return f"makespan {end} != {mk}"
    use = [[0] * k for _ in range(end + 1)]
    for i in range(n):
        for t in range(st[i], st[i] + s.durations[i]):
            for r in range(k):
                use[t][r] += s.demands[i][r]
    for t in range(end):
        for r in range(k):
            if use[t][r] > s.capacities[r]:
                return f"capacity r{r} t{t}"
    return None


def cmd_verify(a):
    """Every exported best schedule is feasible and its makespan equals the trajectory's final value."""
    from collections import Counter
    bad, n = [], Counter()
    for bench in ("j30", "j60", "j120"):
        paths = sorted(OUT.glob(f"traj_{bench}_shard*of7.csv"))
        if not paths:
            continue
        trs = {m: _read_traj(paths, m) for m in MODES + ("ls_off",)}
        for p in sorted(OUT.glob(f"bests_{bench}_shard*of7.csv.gz")):
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
    res = {"checked": dict(n), "violations": len(bad), "first": bad[:20]}
    (OUT / "verify.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


def _ci(d, reps=2000):
    import random
    r = random.Random(0)
    bs = sorted(mean(r.choices(d, k=len(d))) for _ in range(reps))
    return [round(bs[int(0.025 * reps)], 3), round(bs[int(0.975 * reps)], 3)]


def _holm(ps):
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    adj, run = [None] * len(ps), 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - rank) * ps[i]))
        adj[i] = run
    return adj


PAIRS = (("ls", "brdeh2026"), ("ls", "restart"), ("ls", "ls_periodic"), ("prfbi", "brdeh2026"),
         ("restart", "brdeh2026"), ("ls_periodic", "brdeh2026"), ("ls", "prfbi"), ("ls", "b2"))


def _cmp(tr, insts, seeds, ref, B, x, y):
    from scipy.stats import wilcoxon
    val = lambda t, i: mean(value_at(t[(i, s)], B) for s in seeds)
    d = [100 * (val(tr[x], i) - val(tr[y], i)) / ref[i] for i in insts]
    nz = [t for t in d if t != 0]
    return {"mean": round(mean(d), 3), "ci95": _ci(d),
            "better_tie_worse": [sum(t < 0 for t in d), sum(t == 0 for t in d), sum(t > 0 for t in d)],
            "p": float(wilcoxon(nz).pvalue) if len(nz) > 10 else 1.0}


def cmd_analyze(a):
    opt = load_optima()
    res = {"seeds": SEEDS, "budget": BUDGET,
           "params": {"S": S_STALL, "cap": CAP, "origins": N_ORIG, "elite": K_ELITE, "period": PERIOD},
           "benches": {}, "checks": {}}
    tests = []                                   # (bench, B, pair) for Holm
    for bench in ("j30", "j60", "j120"):
        base, insts = _load(bench)
        tr = {"b2": base["b2"], "brdeh2026": base["brdeh2026"]}
        paths = sorted(OUT.glob(f"traj_{bench}_shard*of7.csv"))
        for m in MODES + ("ls_off",):
            tr[m] = _read_traj(paths, m)
        missing = {v: sum((i, s) not in t for i in insts for s in SEEDS) for v, t in tr.items() if v != "ls_off"}
        if any(missing.values()):
            res["benches"][bench] = {"missing": missing}
            continue
        # known answers
        off = [(i, s) for i in off_sample(bench) for s in SEEDS]
        res["checks"][f"{bench}_trigger_off_equals_brdeh2026"] = [
            sum(tr["ls_off"].get(k) == tr["brdeh2026"][k] for k in off), len(off)]
        if bench == "j30":
            res["checks"]["j30_below_optimum"] = sum(value_at(t[(i, s)], BUDGET) < opt[i]
                                                     for t in tr.values() for i in insts for s in SEEDS if (i, s) in t)
        ref = {i: (opt[i] if bench == "j30" else instance_features(i)["lb_cpm"]) for i in insts}
        r = {"instances": len(insts), "gap": {}, "cmp": {}, "shake": {}, "cpu_s_per_run": {}}
        for B in (1000, 2500, 5000):
            r["gap"][B] = {v: round(mean(100 * (mean(value_at(t[(i, s)], B) for s in SEEDS) - ref[i]) / ref[i]
                                         for i in insts), 3) for v, t in tr.items() if v != "ls_off"}
            r["cmp"][B] = {}
            for x, y in PAIRS:
                r["cmp"][B][f"{x} vs {y}"] = _cmp(tr, insts, SEEDS, ref, B, x, y)
                if y != "b2":
                    tests.append((bench, B, f"{x} vs {y}"))
        if bench == "j30":
            r["optima_hit_5000"] = {v: sum(value_at(t[(i, s)], BUDGET) == opt[i] for i in insts for s in SEEDS)
                                    for v, t in tr.items() if v != "ls_off"}
        # clean subset: outside the pilot (J120 pilot instances removed) and seeds 4-10
        pilot = set(instances("j120")[::STRIDE]) if bench == "j120" else set()
        clean = [i for i in insts if i not in pilot]
        r["clean_subset"] = {"instances": len(clean), "seeds": SEEDS[3:],
                             **{f"{x} vs {y} @5000": _cmp(tr, clean, SEEDS[3:], ref, BUDGET, x, y)
                                for x, y in (("ls", "brdeh2026"), ("ls", "restart"), ("ls", "ls_periodic"))}}
        rows = []
        for p in sorted(OUT.glob(f"runs_{bench}_shard*of7.csv")):
            with open(p) as f:
                rows += list(csv.DictReader(f))
        for m in MODES:
            rm = [x for x in rows if x["variant"] == m]
            tot = lambda k: sum(int(x[k]) for x in rm)
            r["shake"][m] = {"runs": len(rm), "runs_with_episode": round(mean(int(x["episodes"]) > 0 for x in rm), 3),
                             "episodes_per_run": round(tot("episodes") / len(rm), 2),
                             "budget_share": round(tot("shake_schedules") / (len(rm) * BUDGET), 3),
                             "improvements_per_run": round(tot("shake_improvements") / len(rm), 3),
                             "runs_with_improvement": round(mean(int(x["shake_improvements"]) > 0 for x in rm), 3),
                             "max_schedules_total": max(int(x["schedules_total"]) for x in rm)}
            r["cpu_s_per_run"][m] = round(mean(float(x["cpu_s"]) for x in rm), 3)
        res["benches"][bench] = r
    # Holm over every benchmark x budget x comparison (B2 context excluded)
    ps = [res["benches"][b]["cmp"][B][k]["p"] for b, B, k in tests]
    for (b, B, k), q in zip(tests, _holm(ps)):
        res["benches"][b]["cmp"][B][k]["p_holm"] = q
    # pilot replica: the J120 pilot rows of ls / prfbi must equal the pilot run
    if "cmp" in res["benches"].get("j120", {}):
        paths = sorted(OUT.glob("traj_j120_shard*of7.csv"))
        for m in ("ls", "prfbi"):
            pil = _read_traj(sorted((RUNS / "pr2_stagnation").glob(f"traj_{m}_shard*of7.csv")), m)
            new = _read_traj(paths, m)
            res["checks"][f"j120_pilot_replica_{m}"] = [sum(new.get(k) == v for k, v in pil.items()), len(pil)]
    (OUT / "summary.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "verify", "analyze"])
    ap.add_argument("bench", nargs="?", choices=["j30", "j60", "j120"])
    ap.add_argument("--shard", type=int, nargs=2, default=[0, 1])
    a = ap.parse_args()
    {"run": cmd_run, "verify": cmd_verify, "analyze": cmd_analyze}[a.cmd](a)
