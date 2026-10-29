#!/usr/bin/env python3
"""Standalone analysis of frozen final-50k quality and local VPS timing records.

This program never invokes a solver.  All source locations are command-line
arguments; output files are written only to --outdir.
"""
from __future__ import annotations
import argparse, collections, csv, hashlib, json, platform, sys
from pathlib import Path
import numpy as np
import scipy
from scipy.stats import wilcoxon

VARIANTS = ("V0", "V5", "V6")
BENCHES = ("j30", "j60", "j120")
CONTRASTS = (("V6", "V0"), ("V6", "V5"), ("V5", "V0"))

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()

def parse_sm_cpm(path):
    """Minimal explicit parser for the fixed-offset PSPLIB .sm format."""
    lines = Path(path).read_text().splitlines()
    n = int(lines[5].split()[-1])
    succ = []
    for i in range(n): succ.append([int(x)-1 for x in lines[18+i].split()[3:]])
    base = 18 + n + 4
    dur = [int(lines[base+i].split()[2]) for i in range(n)]
    es = [0] * n
    for i, targets in enumerate(succ):
        for j in targets:
            if j <= i: raise ValueError(f"non-topological PSPLIB input: {path}")
            es[j] = max(es[j], es[i] + dur[i])
    return max(e+d for e, d in zip(es, dur))

def percentile_ci(samples):
    return [float(x) for x in np.percentile(samples, [2.5, 97.5])]

def boot_mean(values, rng, nboot=20000):
    a = np.asarray(values, float); n = len(a)
    # Batches retain reproducibility without an unnecessarily large matrix.
    out = np.empty(nboot)
    for start in range(0, nboot, 1000):
        k = min(1000, nboot-start)
        out[start:start+k] = a[rng.integers(0, n, size=(k, n))].mean(axis=1)
    return percentile_ci(out)

def holm(pvalues):
    m = len(pvalues); ordered = sorted(range(m), key=lambda i: pvalues[i])
    adjusted = [0.0] * m; prior = 0.0
    for rank, i in enumerate(ordered):
        prior = max(prior, (m-rank) * pvalues[i]); adjusted[i] = min(1.0, prior)
    return adjusted

def write_csv(path, rows):
    rows = list(rows)
    if not rows: return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

def project_member(instance): return Path(instance).stem.rsplit("_", 1)[-1]
def paired_gap_difference(makespans_a, makespans_b, reference):
    """Subtract integer totals before division so genuine ties stay exactly zero."""
    assert len(makespans_a) == len(makespans_b) and len(makespans_a) > 0
    return 100 * (sum(makespans_a) - sum(makespans_b)) / (len(makespans_a) * reference)

def psplib_cell(instance):
    stem = Path(instance).stem.split("_")[0]       # e.g. j6017
    return int(stem[1 + len(instance.split('/')[0])-1:])

def quality_analysis(rows, manifest, inputs, optcsv, rng):
    expected = {tuple(x) for x in manifest["queue"]}
    observed = [(r["instance"], r["seed_index"], r["variant"]) for r in rows]
    assert len(rows) == len(set(observed)) == len(expected) == 46800 and set(observed) == expected
    assert all(r["measurement_environment"] in ("VPS", "Modal") and r["feasible_at_cap"] for r in rows)
    opt = {r["instance"]: int(r["optimal_makespan"]) for r in csv.DictReader(open(optcsv))}
    refs = {}
    for inst in manifest["instances"]:
        actual_hash = sha256(inputs / inst)
        if actual_hash != manifest["input_sha256"][inst]:
            raise AssertionError(f"frozen input hash mismatch: {inst}")
        refs[inst] = opt[Path(inst).name] if inst.startswith("j30/") else parse_sm_cpm(inputs / inst)
    gaps = collections.defaultdict(list); gaps5 = collections.defaultdict(list)
    makespans = collections.defaultdict(list)
    for r in rows:
        key = (r["instance"], r["seed_index"], r["variant"])
        makespans[(r["instance"], r["variant"])].append(r["makespan_at_cap"])
        gaps[(r["instance"], r["variant"])].append(100*(r["makespan_at_cap"]-refs[r["instance"]])/refs[r["instance"]])
        visible=[x[2] for x in r["trajectory"] if x[1] <= 5000]
        assert visible
        gaps5[(r["instance"], r["variant"])].append(100*(min(visible)-refs[r["instance"]])/refs[r["instance"]])
    per = {}
    for (inst, v), x in gaps.items():
        assert len(x) == 10
        mk = makespans[inst, v]
        per[inst, v] = (100 * (sum(mk) - len(mk) * refs[inst]) / (len(mk) * refs[inst]),
                        100 * (min(mk) - refs[inst]) / refs[inst])
    means = []
    prefix5 = []
    for b in BENCHES:
        insts = sorted(i for i in manifest["instances"] if i.startswith(b+"/"))
        for v in VARIANTS:
            means.append(dict(benchmark=b, variant=v, budget=50000, n_instances=len(insts), n_seeds=10,
                mean_gap_pct=float(np.mean([per[i,v][0] for i in insts])), best_of_10_gap_pct=float(np.mean([per[i,v][1] for i in insts])), reference="J30 optimum" if b=="j30" else "CPM"))
            prefix5.append(dict(benchmark=b,variant=v,budget=5000,n_instances=len(insts),n_seeds=10,
                mean_gap_pct=float(np.mean([np.mean(gaps5[i,v]) for i in insts])),
                best_of_10_gap_pct=float(np.mean([np.min(gaps5[i,v]) for i in insts])),
                note="Read from 50k trajectories; not an independent short-cap campaign."))
    tests = []
    for b in BENCHES:
        insts = sorted(i for i in manifest["instances"] if i.startswith(b+"/"))
        for a, c in CONTRASTS:
            d = np.array([paired_gap_difference(makespans[i,a], makespans[i,c], refs[i]) for i in insts])
            if np.all(d == 0): stat, p, method = 0.0, 1.0, "all-zero->p=1"
            else:
                z = d[d != 0]
                res = wilcoxon(z, alternative="two-sided", zero_method="wilcox", correction=False, method="asymptotic")
                stat, p, method = float(res.statistic), float(res.pvalue), "scipy.wilcoxon(method=asymptotic, zero_method=wilcox, correction=False)"
            ci=boot_mean(d,rng)
            tests.append(dict(benchmark=b, contrast=f"{a}-{c}", variant_a=a, variant_b=c, n_instances=len(insts), n_seeds=10,
                mean_paired_gap_diff_pp=float(d.mean()), ci95_low_pp=ci[0], ci95_high_pp=ci[1],
                p_raw=p, wilcoxon_statistic=stat, wilcoxon_method=method,
                project_wins_a=int((d < 0).sum()), project_ties=int((d == 0).sum()), project_losses_a=int((d > 0).sum()),
                nonzero_differences=int((d != 0).sum())))
    adj = holm([x["p_raw"] for x in tests])
    for x, p in zip(tests, adj): x["p_holm_9"] = p
    sensitivity=[]
    for b in BENCHES:
        insts = sorted(i for i in manifest["instances"] if i.startswith(b+"/") and not (b in ("j60","j120") and project_member(i) in ("1","2")))
        for a,c in CONTRASTS:
            d=np.array([paired_gap_difference(makespans[i,a], makespans[i,c], refs[i]) for i in insts])
            ci=boot_mean(d,rng)
            sensitivity.append(dict(benchmark=b, contrast=f"{a}-{c}", n_instances=len(insts), n_seeds=10,
                mean_paired_gap_diff_pp=float(d.mean()), ci95_low_pp=ci[0], ci95_high_pp=ci[1],
                scope="descriptive sensitivity; excludes J60/J120 members _1,_2; no new confirmatory test"))
    return refs, means, tests, sensitivity, prefix5

def timing_analysis(local, manifest, rng):
    grouped=collections.defaultdict(dict)
    for r in local: grouped[(r["instance"],r["seed_index"])][r["variant"]]=r
    complete={k:v for k,v in grouped.items() if set(v)==set(VARIANTS)}
    assert len(local)==2348 and len(complete)==782
    bybench=collections.Counter(k[0].split("/")[0] for k in complete)
    assert bybench == {"j30":226,"j60":236,"j120":320}
    expected_instances={"j30":182,"j60":190,"j120":262}
    expected_cells={"j30":48,"j60":47,"j120":60}
    for b in BENCHES:
        bk=[k for k in complete if k[0].startswith(b+"/")]
        assert len({k[0] for k in bk}) == expected_instances[b]
        assert len({psplib_cell(k[0]) for k in bk}) == expected_cells[b]
    raw=[]; balanced=[]; ratios=[]
    for b in BENCHES:
        blocks={k:v for k,v in complete.items() if k[0].startswith(b+"/")}
        insts=sorted({k[0] for k in blocks})
        for v in VARIANTS:
            rr=[x[v] for x in blocks.values()]
            raw.append(dict(benchmark=b, variant=v, n_complete_blocks=len(rr), n_instances=len(insts),
                raw_mean_cpu_s=float(np.mean([x["cpu_s"] for x in rr])), raw_mean_wall_s=float(np.mean([x["wall_s"] for x in rr])),
                mean_cpu_over_wall=float(np.mean([x["cpu_s"]/x["wall_s"] for x in rr])),
                mean_wait_fraction_1_minus_cpu_over_wall=float(np.mean([1-x["cpu_s"]/x["wall_s"] for x in rr]))))
            perinst=[]
            for i in insts:
                q=[z[v] for k,z in blocks.items() if k[0]==i]
                perinst.append((np.mean([z["cpu_s"] for z in q]),np.mean([z["wall_s"] for z in q])))
            balanced.append(dict(benchmark=b,variant=v,n_instances=len(insts),balanced_complete_blocks=sum(len([1 for k in blocks if k[0]==i]) for i in insts),
                per_instance_balanced_mean_cpu_s=float(np.mean([q[0] for q in perinst])),per_instance_balanced_mean_wall_s=float(np.mean([q[1] for q in perinst]))))
        for a,c in (("V6","V0"),("V6","V5")):
            logs_cpu=[]; logs_wall=[]
            for i in insts:
                q=[z for k,z in blocks.items() if k[0]==i]
                logs_cpu.append(float(np.mean([np.log(z[a]["cpu_s"]/z[c]["cpu_s"]) for z in q])))
                logs_wall.append(float(np.mean([np.log(z[a]["wall_s"]/z[c]["wall_s"]) for z in q])))
            def gci(x):
                a=np.asarray(x); n=len(a); out=np.empty(20000)
                for start in range(0,20000,1000):
                    k=min(1000,20000-start); out[start:start+k]=np.exp(a[rng.integers(0,n,(k,n))].mean(1))
                return float(np.exp(a.mean())), percentile_ci(out)
            gc,gcic=gci(logs_cpu); gw,gciw=gci(logs_wall)
            ratios.append(dict(benchmark=b,contrast=f"{a}/{c}",n_instances=len(insts),ratio_unit="geometric mean of per-instance average complete-block ratios",
                cpu_ratio=gc,cpu_ci95_low=gcic[0],cpu_ci95_high=gcic[1],wall_ratio=gw,wall_ci95_low=gciw[0],wall_ci95_high=gciw[1]))
    counts={b:{"complete_blocks":bybench[b],"instances":len({k[0] for k in complete if k[0].startswith(b+"/")})} for b in BENCHES}
    return raw, balanced, ratios, counts, len(grouped)-len(complete)

def latex_tables(means, tests, timing):
    m="\\begingroup\\fontsize{10.5}{13}\\selectfont\n\\begin{tabular}{lrrr}\n\\toprule Set & V0 & V5 & V6\\\\\n\\midrule\n"
    for b in BENCHES:
        r={x['variant']:x for x in means if x['benchmark']==b}; m+=f"{b.upper()} & {r['V0']['mean_gap_pct']:.3f} & {r['V5']['mean_gap_pct']:.3f} & {r['V6']['mean_gap_pct']:.3f}\\\\\n"
    m+="\\bottomrule\n\\end{tabular}\n\\endgroup\n% Mean gap (%), 50,000 schedules; lower is better."
    c="\\begingroup\\fontsize{10.5}{13}\\selectfont\n\\begin{tabular}{llrrrrl}\n\\toprule Set & Contrast & $\\Delta$ pp & 95\\% CI & $p$ & $p_H$ & W/T/L\\\\\n\\midrule\n"
    for x in tests: c+=f"{x['benchmark'].upper()} & {x['contrast']} & {x['mean_paired_gap_diff_pp']:.3f} & [{x['ci95_low_pp']:.3f}, {x['ci95_high_pp']:.3f}] & {x['p_raw']:.3g} & {x['p_holm_9']:.3g} & {x['project_wins_a']}/{x['project_ties']}/{x['project_losses_a']}\\\\\n"
    c+="\\bottomrule\n\\end{tabular}\n\\endgroup\n% A-B; negative favours A; one Holm family of nine."
    t="\\begingroup\\fontsize{10.5}{13}\\selectfont\n\\begin{tabular}{llrrr}\n\\toprule Set & Contrast & CPU ratio & Wall ratio & $n$ instances\\\\\n\\midrule\n"
    for x in timing: t+=f"{x['benchmark'].upper()} & {x['contrast']} & {x['cpu_ratio']:.3f} [{x['cpu_ci95_low']:.3f}, {x['cpu_ci95_high']:.3f}] & {x['wall_ratio']:.3f} [{x['wall_ci95_low']:.3f}, {x['wall_ci95_high']:.3f}] & {x['n_instances']}\\\\\n"
    return m+"\n", c+"\\bottomrule\n\\end{tabular}\n\\endgroup\n", t+"\\bottomrule\n\\end{tabular}\n\\endgroup\n"

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--migration',type=Path,required=True); ap.add_argument('--inputs',type=Path,required=True); ap.add_argument('--j30-opt',type=Path,required=True); ap.add_argument('--outdir',type=Path,default=Path('.')); a=ap.parse_args(); a.outdir.mkdir(exist_ok=True)
    quality_path=a.migration/'merged-quality-records.jsonl'; local_path=a.migration/'preserved-local-records.jsonl'; manifest_path=a.migration/'preserved-local-manifest.json'
    rows=[json.loads(x) for x in quality_path.read_text().splitlines()]; local=[json.loads(x) for x in local_path.read_text().splitlines()]; manifest=json.loads(manifest_path.read_text())
    rng=np.random.default_rng(20261028)
    refs, means, tests, sensitivity, prefix5=quality_analysis(rows,manifest,a.inputs,a.j30_opt,rng)
    raw, balanced, ratios, counts, partial=timing_analysis(local,manifest,rng)
    write_csv(a.outdir/'final50k_means.csv',means); write_csv(a.outdir/'final50k_contrasts.csv',tests); write_csv(a.outdir/'final50k_sensitivity.csv',sensitivity); write_csv(a.outdir/'longprefix_5k_summary.csv',prefix5); write_csv(a.outdir/'local_timing_raw.csv',raw); write_csv(a.outdir/'local_timing_balanced.csv',balanced); write_csv(a.outdir/'local_timing_ratios.csv',ratios)
    report=dict(analysis_seed=20261028, bootstrap_resamples=20000, quality=dict(records=len(rows), exact_manifest_coverage=True, references="J30 known optimum; J60/J120 CPM parsed from .sm", means=means, contrasts=tests, heldout_ils_sensitivity=sensitivity, longprefix_5k=prefix5), timing=dict(source_records=len(local), environment="VPS only; cloud/Modal timings never pooled", complete_blocks=782, incomplete_groups_excluded=partial, counts=counts, raw_block_means=raw, per_instance_balanced_means=balanced, matched_ratios=ratios, contract=manifest['timing_contract']), input_sha256={str(p):sha256(p) for p in (quality_path,local_path,manifest_path,a.j30_opt)}, versions=dict(python=sys.version, numpy=np.__version__, scipy=scipy.__version__, platform=platform.platform()))
    (a.outdir/'analysis_results.json').write_text(json.dumps(report,indent=2)+"\n")
    (a.outdir/'frozen_input_hashes.json').write_text(json.dumps(dict(validated_inputs=len(manifest['input_sha256']), sha256=manifest['input_sha256']),indent=2)+"\n")
    (a.outdir/'analysis_config.json').write_text(json.dumps(dict(seed=20261028, bootstrap_resamples=20000, quality_contrasts=[f'{x}-{y}' for x,y in CONTRASTS], holm_family_size=9, timing_contrasts=['V6/V0','V6/V5'], timing_environment='VPS only'),indent=2)+"\n")
    m,c,t=latex_tables(means,tests,ratios); (a.outdir/'table_final50k_means.tex').write_text(m); (a.outdir/'table_final50k_contrasts.tex').write_text(c); (a.outdir/'table_local_timing.tex').write_text(t)
    (a.outdir/'versions_manifest.json').write_text(json.dumps(report['versions'],indent=2)+"\n")
    print(json.dumps({'quality_records':len(rows),'complete_vps_blocks':782,'outputs':str(a.outdir)},indent=2))
if __name__=='__main__': main()
