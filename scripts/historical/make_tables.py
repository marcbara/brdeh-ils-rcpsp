#!/usr/bin/env python3
"""Generate the LaTeX tables and pgfplots data of the manuscript from existing artifacts.

Inputs (read-only): data/main.json (written by extract.py) and the campaign summaries under
experiments/runs/. Outputs: tables/*.tex, *.csv and evidence.json under HISTORICAL_OUT.

    python scripts/reproduce_historical.py
"""

import hashlib
import os
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HIST = Path(os.environ["HISTORICAL_RAW_ROOT"])
HERE = Path(os.environ["HISTORICAL_OUT"])
RUNS = HIST / "experiments" / "runs"
TAB = HERE / "tables"
DATA = HERE
TAB.mkdir(exist_ok=True)
USED = {}


def load(rel):
    p = RUNS / rel
    USED[f"experiments/runs/{rel}"] = hashlib.sha256(p.read_bytes()).hexdigest()
    return json.loads(p.read_text())


M = json.loads((DATA / "main.json").read_text())["benches"]
DEC = {"j30": 3, "j60": 2, "j120": 2}
SETS = ("j30", "j60", "j120")
SETNAME = {"j30": "J30", "j60": "J60", "j120": "J120"}


def f(x, d=2, sign=False):
    s = f"{x:+.{d}f}" if sign else f"{x:.{d}f}"
    return s.replace("-", "$-$") if not sign else s.replace("-", "$-$").replace("+", "$+$")


def pfmt(p):
    if p is None:
        return "--"
    if p >= 0.001:
        return f"{p:.3f}"
    m, e = f"{p:.0e}".split("e")
    return f"$10^{{{int(e)}}}$" if m == "1" else f"${m}\\cdot10^{{{int(e)}}}$"


def ci(c, d=2):
    return f"[{f(c[0], d)}, {f(c[1], d)}]"


WIDE = {"tab_main.tex", "tab_attacks.tex", "tab_direction.tex", "tab_composite.tex", "tab_literature.tex",
        "tab_ilstest.tex"}


def write(name, body):
    # wide tables are scaled to the text width (layout only; cell contents unchanged)
    if name in WIDE:
        body = body.replace("\\begin{tabular}", "\\resizebox{\\textwidth}{!}{%\n\\begin{tabular}", 1)
        body = body.replace("\\end{tabular}", "\\end{tabular}}", 1)
    (TAB / name).write_text(body)


# ---------------------------------------------------------------- main progression (Table: tab:main)
VNAME = {"V0": "BR-DEH 2024 (re-implementation)", "V1": "13-rule BR-DEH", "V2": "\\quad + local search",
         "V3": "\\quad + direction", "V4": "\\quad + direction + local search",
         "V5": "BR-DEH 2026", "V6": "BR-DEH 2026 + ILS"}
STEP = {"V1": "V1 vs V0", "V2": "V2 vs V1", "V3": "V3 vs V1", "V4": "V4 vs V1", "V5": "V5 vs V4", "V6": "V6 vs V5"}
rows = []
for b in SETS:
    d = DEC[b]
    r = M[b]
    ref = "known optimum" if b == "j30" else "critical-path bound"
    rows.append(f"\\multicolumn{{8}}{{l}}{{\\textbf{{{SETNAME[b]}}} ({r['n_instances']} instances $\\times$ 10 seeds; "
                f"gap to the {ref})}}\\\\")
    for v in ("V0", "V1", "V2", "V3", "V4", "V5", "V6"):
        x = r["versions"][v]
        cells = [f(x["mean"][str(B)], d) for B in (1000, 2500, 5000)] + [f(x["best10_5000"], d)]
        cells.append(str(x["runs_at_reference_5000"]))
        if v in STEP:
            c = r["steps"]["5000"][STEP[v]]
            cells += [f"{f(c['mean'], d, True)} {ci(c['ci95'], d)}", pfmt(c["p_holm_family"])]
        else:
            cells += ["--", "--"]
        rows.append(f"{v} & {VNAME[v]} & " + " & ".join(cells) + "\\\\")
    rows.append("\\midrule" if b != "j120" else "")
write("tab_main.tex", r"""\begin{table*}[t]
\centering
\caption{Mean gap (\%) of every version on the complete PSPLIB sets, ten seeds per instance (seed indices 1--10),
budget counted in generated schedules. ``best-10'' is the mean over instances of the best of the ten runs at 5{,}000
schedules (ten times the budget). ``at ref.'' counts runs (out of $10\times$ instances) whose makespan equals the reference
(J30: optimum; J60/J120: critical-path bound, hence proven optimal). $\Delta$ is the paired per-instance
difference at 5{,}000 schedules against the version indicated in Table~\ref{tab:versions} (negative = better), with a
bootstrap 95\% interval; $p_{\mathrm{H}}$ is the two-sided Wilcoxon signed-rank $p$-value, Holm-adjusted within the
family of eight comparisons of that set and budget. V0: archived trajectories of our re-implementation; V1--V4: one
campaign; V5 and V6: two further campaigns with the same seeds.}
\label{tab:main}
\small
\setlength{\tabcolsep}{4pt}
\begin{tabular}{llrrrrrll}
\toprule
 & Version & 1{,}000 & 2{,}500 & 5{,}000 & best-10 & at ref. & $\Delta$ at 5{,}000 [95\% CI] & $p_{\mathrm{H}}$\\
\midrule
""" + "\n".join(rows).replace("\\multicolumn{8}", "\\multicolumn{9}") + r"""
\bottomrule
\end{tabular}
\end{table*}
""")

# 1000-schedule step deltas (inline in text), and V5/V6 vs V0
steps_inline = {b: {B: {k: {"mean": v["mean"], "ci95": v["ci95"], "p_holm": v["p_holm_family"],
                             "btw": v["better_tie_worse"]} for k, v in M[b]["steps"][B].items()}
                    for B in ("1000", "2500", "5000")} for b in SETS}

# ---------------------------------------------------------------- exploratory J120 3 seeds (tab:explore)
var = load("variants/summary_j120_s1-2-3.json")
fr = load("fixed_rule/summary_j120_s1-2-3.json")
a1 = load("a1_uniform/summary_j120_s1-2-3.json")
e1 = load("e1_learned/summary_j120_s1-2-3.json")
g = var["gap"]
ex = [
    ("BR-DEH 2024 (re-implementation)", [g[k]["b2"] for k in ("1000", "2500", "5000")], "--", "--"),
    ("uniform $\\beta$ only", [g[k]["nobeta"] for k in ("1000", "2500", "5000")],
     pfmt(var["vs_b2"]["1000"]["nobeta"]["wilcoxon_p"]), pfmt(var["vs_b2"]["5000"]["nobeta"]["wilcoxon_p"])),
    ("13-rule start only (inherited $\\beta$)", [g[k]["init13"] for k in ("1000", "2500", "5000")],
     pfmt(var["vs_b2"]["1000"]["init13"]["wilcoxon_p"]), pfmt(var["vs_b2"]["5000"]["init13"]["wilcoxon_p"])),
    ("13-rule start + uniform $\\beta$ (= V1)", [g[k]["init13_nobeta"] for k in ("1000", "2500", "5000")],
     pfmt(var["vs_b2"]["1000"]["init13_nobeta"]["wilcoxon_p"]), pfmt(var["vs_b2"]["5000"]["init13_nobeta"]["wilcoxon_p"])),
]
gb = fr["gap"]
ga = a1["gap"]
ge = e1["gap"]
ex2 = [
    ("V1 (reference of this block)", [gb[k]["base"] for k in ("1000", "2500", "5000")], "--", "--"),
    ("F1: start winner fixed for every try", [gb[k]["F1"] for k in ("1000", "2500", "5000")],
     pfmt(fr["cmp"]["1000"]["F1 vs base (x better/tie/y better, p)"][3]), pfmt(fr["cmp"]["5000"]["F1 vs base (x better/tie/y better, p)"][3])),
    ("F2: 80/20 around the start winner, never replaced", [gb[k]["F2"] for k in ("1000", "2500", "5000")],
     pfmt(fr["cmp"]["1000"]["F2 vs base (x better/tie/y better, p)"][3]), pfmt(fr["cmp"]["5000"]["F2 vs base (x better/tie/y better, p)"][3])),
    ("F3: uniform among the rules tied at the start", [gb[k]["F3"] for k in ("1000", "2500", "5000")],
     pfmt(fr["cmp"]["1000"]["F3 vs base (x better/tie/y better, p)"][3]), pfmt(fr["cmp"]["5000"]["F3 vs base (x better/tie/y better, p)"][3])),
    ("A1: rule drawn uniformly from the 13 at every try", [ga[k]["a1"] for k in ("1000", "2500", "5000")],
     pfmt(a1["vs_base"]["1000"]["wilcoxon_p"]), pfmt(a1["vs_base"]["5000"]["wilcoxon_p"])),
    ("E1: learned 14th rule (elite mean start) in 50\\% of tries$^{\\dagger}$", [ge[k]["e1"] for k in ("1000", "2500", "5000")],
     pfmt(e1["vs_base"]["1000"]["wilcoxon_p"]), pfmt(e1["vs_base"]["5000"]["wilcoxon_p"])),
]
lines = []
for name, vals, p1, p5 in ex:
    lines.append(f"{name} & " + " & ".join(f(v) for v in vals) + f" & {p1} & {p5}\\\\")
lines.append("\\midrule")
for name, vals, p1, p5 in ex2:
    lines.append(f"{name} & " + " & ".join(f(v) for v in vals) + f" & {p1} & {p5}\\\\")
write("tab_explore.tex", r"""\begin{table}[t]
\centering
\caption{Exploratory tests of the multistart parameters on J120 (600 instances, seeds 1--3). Mean gap to the
critical-path bound (\%). $p$: paired two-sided Wilcoxon test per instance, against BR-DEH 2024 (upper block) or
against V1 (lower block), unadjusted. $^{\dagger}$E1 used elite snapshots affected by a recording error described in
Section~\ref{sec:explore}; we treat it as weak evidence.}
\label{tab:explore}
\small
\setlength{\tabcolsep}{3.5pt}
\begin{tabular}{p{6.2cm}rrrrr}
\toprule
Variant & 1{,}000 & 2{,}500 & 5{,}000 & $p_{1000}$ & $p_{5000}$\\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
\end{table}
""")
explore_extra = {"duplicates": var["duplicates"], "a1_tries": a1["tries"],
                 "init13": load("init13/summary_j120_s1-2-3.json"), "det13": load("det13/summary.json"),
                 "e2": load("e2_fullorder/summary.json"), "pr1": load("pr_stagnation/summary.json")}

# ---------------------------------------------------------------- attacks on stagnation (tab:attacks)
cls = load("confirm_ls/summary.json")["benches"]
v5n = load("escoto_ideas/summary_j120v5n.json")
att = []
for key, label in (("restart vs brdeh2026", "restart from the initial state when stalled"),
                   ("prfbi vs brdeh2026", "path relinking + FBI when stalled"),
                   ("ls vs brdeh2026", "local search on the best schedule when stalled"),
                   ("ls_periodic vs brdeh2026", "local search at fixed intervals (every 557 schedules)")):
    cells = []
    for b in SETS:
        c = cls[b]["cmp"]["5000"][key]
        d = 3 if b == "j30" else 2
        cells.append(f"{f(c['mean'], d, True)} {ci(c['ci95'], d)}")
        cells.append(pfmt(c["p_holm"]))
    att.append(f"{label} & " + " & ".join(cells) + "\\\\")
for key, label in (("N_dyn1 vs N", "dynamic FBI (patience 1)$^{\\ast}$"), ("N_narrow vs N", "narrow inherited $\\beta$ ($\\pm0.3$)$^{\\ast}$")):
    c = v5n["cmp"]["5000"][key]
    att.append(f"{label} & \\multicolumn{{4}}{{c}}{{not run}} & {f(c['mean'], 2, True)} {ci(c['ci95'])} & {pfmt(c['p_holm'])}\\\\")
shake = {b: cls[b]["shake"] for b in SETS}
write("tab_attacks.tex", r"""\begin{table*}[t]
\centering
\caption{Interventions against stagnation, added to the 13-rule BR-DEH (V1): paired difference of the mean gap at 5{,}000
schedules (negative = better), bootstrap 95\% interval and Holm-adjusted Wilcoxon $p$ within the campaign family.
Complete sets, seeds 1--10; every FBI pass and every local-search decode is charged to the same budget. The trigger
is 500 schedules without improving the incumbent; one episode spends at most 300 schedules. $^{\ast}$Separate
campaign on J120 only (same seeds, same base).}
\label{tab:attacks}
\small
\setlength{\tabcolsep}{3.5pt}
\begin{tabular}{p{5.4cm}llllll}
\toprule
 & \multicolumn{2}{c}{J30} & \multicolumn{2}{c}{J60} & \multicolumn{2}{c}{J120}\\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}
Intervention & $\Delta$ [95\% CI] & $p_{\mathrm{H}}$ & $\Delta$ [95\% CI] & $p_{\mathrm{H}}$ & $\Delta$ [95\% CI] & $p_{\mathrm{H}}$\\
\midrule
""" + "\n".join(att) + r"""
\bottomrule
\end{tabular}
\end{table*}
""")

# ---------------------------------------------------------------- direction (tab:direction)
cb = load("confirm_bidir/summary.json")["benches"]
dl = []
for key, label in (("winner vs base", "winning direction vs V1"), ("fwd78 vs base", "always forward (78-schedule start) vs V1"),
                   ("back78 vs base", "always backward (78-schedule start) vs V1"),
                   ("winner vs fwd78", "winning vs always forward"), ("winner vs back78", "winning vs always backward"),
                   ("winner vs mix", "winning vs 50/50 per try"), ("loser vs fwd78", "losing vs always forward"),
                   ("loser vs back78", "losing vs always backward"), ("winner vs oracle_honest", "winning vs honest oracle")):
    cells = []
    for b in SETS:
        c = cb[b]["cmp"]["5000"][key]
        d = 3 if b == "j30" else 2
        cells.append(f"{f(c['mean'], d, True)} {ci(c['ci95'], d)}")
        cells.append(pfmt(c.get("p_holm")) if "p_holm" in c else "--")
    dl.append(f"{label} & " + " & ".join(cells) + "\\\\")
sel = []
for b in SETS:
    s = cb[b]["selection"]["5000"]
    sel.append(f"{SETNAME[b]}: backward start wins {100 * cb[b]['init_backward_wins_share']:.0f}\\%, ties "
               f"{100 * cb[b]['init_tie_share']:.0f}\\%, agreement with honest oracle {100 * s['init_hit_rate_vs_honest_oracle']:.0f}\\%, "
               f"captured share of the oracle gain {100 * s['capture_of_honest_oracle']:.0f}\\%")
write("tab_direction.tex", r"""\begin{table*}[t]
\centering
\caption{Direction of construction. All arms share the 78-schedule bidirectional start (13 rules forward and backward,
build + FBI), except V1 (39-schedule forward start); they differ only in the direction of the later tries. Paired
difference of the mean gap at 5{,}000 schedules (seeds 1--10), bootstrap 95\% interval, Holm-adjusted Wilcoxon $p$
over the campaign family. The honest oracle fixes, per instance, the direction that was better on the disjoint seeds
11--20 and is evaluated on seeds 1--10.}
\label{tab:direction}
\small
\setlength{\tabcolsep}{3.5pt}
\begin{tabular}{p{4.6cm}llllll}
\toprule
 & \multicolumn{2}{c}{J30} & \multicolumn{2}{c}{J60} & \multicolumn{2}{c}{J120}\\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}
Comparison & $\Delta$ [95\% CI] & $p_{\mathrm{H}}$ & $\Delta$ [95\% CI] & $p_{\mathrm{H}}$ & $\Delta$ [95\% CI] & $p_{\mathrm{H}}$\\
\midrule
""" + "\n".join(dl) + r"""
\bottomrule
\end{tabular}
\end{table*}
""")

# ---------------------------------------------------------------- composite rule (tab:composite)
v5 = {b: load(f"escoto_ideas/summary_{b}v5.json") for b in SETS}
cl = []
for key, label in (("hyb", "composite rule alone (replaces the 13)"), ("hyb14", "composite as a 14th rule"),
                   ("lft", "LFT alone (control)")):
    c = v5n["cmp"]["5000"][f"N_{key} vs N"]
    cells = [f"{f(c['mean'], 2, True)} {ci(c['ci95'])}", pfmt(c["p_holm"])]
    for b in SETS:
        c = v5[b]["cmp"]["5000"][f"I_{key} vs I"]
        d = 3 if b == "j30" else 2
        cells += [f"{f(c['mean'], d, True)} {ci(c['ci95'], d)}", pfmt(c["p_holm"])]
    cl.append(f"{label} & " + " & ".join(cells) + "\\\\")
write("tab_composite.tex", r"""\begin{table*}[t]
\centering
\caption{The composite priority rule, at 5{,}000 schedules (seeds 1--10). First pair of columns: on the 13-rule BR-DEH
(V1), J120. Remaining columns: on V4 (direction + local search), all sets; V4 with the composite rule is BR-DEH 2026
(V5). Paired difference of the mean gap, bootstrap 95\% interval, Holm-adjusted Wilcoxon $p$ within each campaign.}
\label{tab:composite}
\small
\setlength{\tabcolsep}{3pt}
\begin{tabular}{p{3.8cm}llllllll}
\toprule
 & \multicolumn{2}{c}{on V1, J120} & \multicolumn{2}{c}{on V4, J30} & \multicolumn{2}{c}{on V4, J60} & \multicolumn{2}{c}{on V4, J120}\\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}
Rule & $\Delta$ [95\% CI] & $p_{\mathrm{H}}$ & $\Delta$ [95\% CI] & $p_{\mathrm{H}}$ & $\Delta$ [95\% CI] & $p_{\mathrm{H}}$ & $\Delta$ [95\% CI] & $p_{\mathrm{H}}$\\
\midrule
""" + "\n".join(cl) + r"""
\bottomrule
\end{tabular}
\end{table*}
""")

# ---------------------------------------------------------------- ILS test (tab:ilstest)
it = {b: load(f"br_ils/summary_{b}test.json")["by_set"][b] for b in SETS}
it_mech = {b: load(f"br_ils/summary_{b}test.json")["mech"] for b in SETS}
arms = [("br_r10_l5", "biased window rebuild (adopted)"), ("grd_r10_l5", "greedy window rebuild ($\\beta=1$)"),
        ("uni_r10_l5", "uniform window rebuild"), ("ls_l5", "local search again, no perturbation"),
        ("rm_r10_l5", "random insertion moves")]
il = []
il.append("BR-DEH 2026 (V5), mean gap & " + " & ".join(
    f"{f(it[b]['gap'][B]['B'], DEC[b])}" for b in SETS for B in ("1000", "2500", "5000")) + " & \\\\")
il.append("\\midrule")
for a, label in arms:
    cells = []
    for b in SETS:
        for B in ("1000", "2500", "5000"):
            c = it[b]["cmp"][B][f"{a} vs B"]
            s = f(c["mean"], DEC[b], True)
            if c["p_holm"] < 0.05:
                s = f"\\textbf{{{s}}}"
            cells.append(s)
    il.append(f"{label} & " + " & ".join(cells) + " & \\\\")
write("tab_ilstest.tex", r"""\begin{table*}[t]
\centering
\caption{Iterated local search on held-out instances (J30: all 480; J60: 384 and J120: 480 instances of indices
\_3 to \_10, none used for tuning) with seeds 1--10, disjoint from the tuning seeds. First row: mean gap of BR-DEH 2026;
other rows: paired difference with BR-DEH 2026 (negative = better). Bold: Holm-adjusted Wilcoxon $p<0.05$ within the
five arms of that set and budget. All arms are identical to BR-DEH 2026 up to its first stall and spend exactly the
same budget.}
\label{tab:ilstest}
\small
\setlength{\tabcolsep}{3pt}
\begin{tabular}{p{4.3cm}rrrrrrrrrl}
\toprule
 & \multicolumn{3}{c}{J30} & \multicolumn{3}{c}{J60} & \multicolumn{3}{c}{J120} &\\
\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-10}
 & 1{,}000 & 2{,}500 & 5{,}000 & 1{,}000 & 2{,}500 & 5{,}000 & 1{,}000 & 2{,}500 & 5{,}000 &\\
\midrule
""" + "\n".join(il) + r"""
\bottomrule
\end{tabular}
\end{table*}
""")

# ---------------------------------------------------------------- ILS dev grid (tab:ilsdev)
dev = load("br_ils/summary_dev.json")["by_set"]
devax = load("br_ils/summary_devax.json")["by_set"]
gl = []
for p, lab in (("br", "biased"), ("grd", "greedy"), ("uni", "uniform"), ("rm", "random moves")):
    for lam in (5, 10):
        cells = []
        for b in ("j60", "j120"):
            base = dev[b]["gap"]["5000"]["B"]
            for rho in (10, 20, 30):
                cells.append(f(dev[b]["gap"]["5000"][f"{p}_r{rho}_l{lam}"] - base, 2, True))
        gl.append(f"{lab}, $\\eta={'1/2' if lam == 5 else '1'}$ & " + " & ".join(cells) + "\\\\")
gl.append("\\midrule")
gl.append("local search only, no perturbation & \\multicolumn{3}{c}{" + f(dev["j60"]["gap"]["5000"]["ls_l5"] - dev["j60"]["gap"]["5000"]["B"], 2, True)
          + "} & \\multicolumn{3}{c}{" + f(dev["j120"]["gap"]["5000"]["ls_l5"] - dev["j120"]["gap"]["5000"]["B"], 2, True) + "}\\\\")
for a, lab in (("ax_br", "suffix rebuild, adaptive $p$, demon, no LS"), ("ax_br_ls", "same + our local search"),
               ("ax_br_hils", "same, two-stage (5 elite bases), no LS"), ("ax_br_ls_hils", "two-stage + local search"),
               ("ax_uni", "suffix rebuild, uniform, no LS")):
    gl.append(f"{lab} & \\multicolumn{{3}}{{c}}{{{f(devax['j60']['gap']['5000'][a] - devax['j60']['gap']['5000']['B'], 2, True)}}} & "
              f"\\multicolumn{{3}}{{c}}{{{f(devax['j120']['gap']['5000'][a] - devax['j120']['gap']['5000']['B'], 2, True)}}}\\\\")
write("tab_ilsdev.tex", r"""\begin{table}[t]
\centering
\caption{Tuning campaign of the ILS (development instances \_1 and \_2 of every J60/J120 parameter class, 96 + 120
instances, seeds 21--30): difference of the mean gap at 5{,}000 schedules with BR-DEH 2026 (negative = better), for
window sizes $\rho J$ and local-search patience $L=\eta\,(J+2)$. Lower block: transfer of the BR-ILS of \citet{arnau2022} (last
$p\%$ of the order rebuilt, adaptive $p\in[10\%,50\%]$, demon acceptance). The configuration in the first row and first
column was frozen before any test run.}
\label{tab:ilsdev}
\small
\setlength{\tabcolsep}{3pt}
\begin{tabular}{p{3.9cm}rrrrrr}
\toprule
 & \multicolumn{3}{c}{J60, $\rho=$} & \multicolumn{3}{c}{J120, $\rho=$}\\
\cmidrule(lr){2-4}\cmidrule(lr){5-7}
Perturbation & 0.1 & 0.2 & 0.3 & 0.1 & 0.2 & 0.3\\
\midrule
""" + "\n".join(gl) + r"""
\bottomrule
\end{tabular}
\end{table}
""")

# ---------------------------------------------------------------- long budget (tab:long + csv)
lg = load("long_ils_stratified/summary.json")
with open(DATA / "long_budget.csv", "w") as fcsv:
    fcsv.write("B j60br j60grd j60uni j120br j120grd j120uni\n")
    for B in ("5000", "10000", "25000", "50000"):
        fcsv.write(B + " " + " ".join(f"{lg['by_set'][b]['schedules'][B]['mean_gap_pct_cpm'][a]:.5f}"
                                       for b in ("j60", "j120") for a in ("br_r10_l5", "grd_r10_l5", "uni_r10_l5")) + "\n")
ll = []
for c in lg["primary_contrasts"]:
    ll.append(f"{SETNAME[c['benchmark']]} ($n={c['n_instances']}$) & {'greedy' if c['control'].startswith('grd') else 'uniform'} & "
              f"{f(c['difference_pp'], 3, True)} & {ci(c['ci95_pp'], 3)} & {c['p_raw']:.3f} & {c['p_holm']:.3f} & "
              f"{'/'.join(map(str, c['instance_win_tie_loss']))}\\\\")
write("tab_long.tex", r"""\begin{table}[t]
\centering
\caption{Registered long-budget extension (all instances with index \_3: 48 J60 and 60 J120; seeds 11--13; budget
50{,}000 schedules; adopted ILS with biased, greedy or uniform window rebuild). Biased minus control at 50{,}000,
in points of gap to the critical-path bound; paired bootstrap interval (20{,}000 resamples), two-sided sign-flip
randomisation test (100{,}000 draws) and Holm adjustment over exactly these four contrasts, all fixed before the run.}
\label{tab:long}
\small
\setlength{\tabcolsep}{3pt}
\begin{tabular}{llrlrrl}
\toprule
Set & Control & $\Delta$ & 95\% CI & $p$ & $p_{\mathrm{H}}$ & W/T/L\\
\midrule
""" + "\n".join(ll) + r"""
\bottomrule
\end{tabular}
\end{table}
""")

# ---------------------------------------------------------------- constructor (tab:constructor)
cc = load("constructor_compare/detailed_summary.json")
q, t = cc["quality"], cc["timing"]
write("tab_constructor.tex", r"""\begin{table}[t]
\centering
\caption{DEH versus a serial SGS as the constructor of BR-DEH 2026 + ILS, everything else identical (12 J120
instances, one per pair of parameter classes; seeds 1--3; 36 paired runs; one pinned CPU, single-threaded Julia,
JIT excluded). Quality: mean gap to the critical-path bound (\%), and serial-SGS wins/ties/losses over the 36 pairs.
Time: CPU per constructor call and per run. Raw/post-FBI: mean makespan of one constructor call before and after its
FBI pass.}
\label{tab:constructor}
\small
\begin{tabular}{lrr}
\toprule
 & DEH & serial SGS\\
\midrule
""" + "\n".join([
    f"gap at 5{{,}}000 schedules & {f(q['schedules@5000']['mean_gap_pct_cpm']['DEH'])} & {f(q['schedules@5000']['mean_gap_pct_cpm']['SSGS'])} ({'/'.join(map(str, q['schedules@5000']['SSGS_win_tie_loss']))})\\\\",
    f"gap at 50{{,}}000 schedules & {f(q['schedules@50000']['mean_gap_pct_cpm']['DEH'])} & {f(q['schedules@50000']['mean_gap_pct_cpm']['SSGS'])} ({'/'.join(map(str, q['schedules@50000']['SSGS_win_tie_loss']))})\\\\",
    f"gap at {cc['horizons']['cpu_s']:.2f} s CPU & {f(q['cpu_s@' + str(cc['horizons']['cpu_s'])]['mean_gap_pct_cpm']['DEH'])} & {f(q['cpu_s@' + str(cc['horizons']['cpu_s'])]['mean_gap_pct_cpm']['SSGS'])} ({'/'.join(map(str, q['cpu_s@' + str(cc['horizons']['cpu_s'])]['SSGS_win_tie_loss']))})\\\\",
    "\\midrule",
    f"CPU per constructor call (ms) & {t['DEH']['mean_constructor_cpu_ms']:.3f} & {t['SSGS']['mean_constructor_cpu_ms']:.3f}\\\\",
    f"CPU per constructor + FBI (ms) & {t['DEH']['mean_constructor_fbi_cpu_ms']:.3f} & {t['SSGS']['mean_constructor_fbi_cpu_ms']:.3f}\\\\",
    f"constructor calls per run & {t['DEH']['mean_invocations']:.0f} & {t['SSGS']['mean_invocations']:.0f}\\\\",
    f"raw / post-FBI makespan & {t['DEH']['mean_raw']:.1f} / {t['DEH']['mean_post_fbi']:.1f} & {t['SSGS']['mean_raw']:.1f} / {t['SSGS']['mean_post_fbi']:.1f}\\\\",
    f"total CPU per 50{{,}}000-schedule run (s) & {t['DEH']['mean_total_cpu_s']:.2f} & {t['SSGS']['mean_total_cpu_s']:.2f}\\\\",
]) + r"""
\bottomrule
\end{tabular}
\end{table}
""")

# ---------------------------------------------------------------- literature (tab:literature)
import csv as _csv  # noqa: E402
lit = list(_csv.DictReader(open(ROOT / "archived" / "data" / "literature.csv"), delimiter=";"))
lr = []
for r in sorted(lit, key=lambda r: float(r["j120_5k"])):
    cells = [r[c] for c in ("j30_1k", "j30_5k", "j30_50k", "j60_1k", "j60_5k", "j60_50k", "j120_1k", "j120_5k", "j120_50k")]
    lr.append(f"{r['label']} \\citep{{{r['cite']}}} & " + " & ".join(cells) + "\\\\")
ours = []
for v, name in (("V0", "BR-DEH 2024 (V0, our re-impl.)"), ("V5", "BR-DEH 2026 (V5)"), ("V6", "BR-DEH 2026 + ILS (V6)")):
    cells = []
    for b in SETS:
        x = M[b]["versions"][v]["mean"]
        cells += [f(x["1000"], 2), f(x["5000"], 2)]
        cells.append(f(M[b]["V0_50000"]["mean"], 2) if v == "V0" else "n/a")
    ours.append(f"\\textbf{{{name}}} & " + " & ".join(cells) + "\\\\")
write("tab_literature.tex", r"""\begin{table*}[t]
\centering
\caption{Indicative position among published RCPSP heuristics: average deviation (\%) from the optimum (J30) or the
critical-path bound (J60, J120) at 1{,}000 / 5{,}000 / 50{,}000 generated schedules. Literature rows (sorted by J120 at
5{,}000): numbers as tabulated by \citet{kolisch2006experimental} for the methods it covers, otherwise from at least one
secondary comparison table (sources and labels in \texttt{data/literature.csv}); published figures follow each paper's
own counting and may be single runs, averages or best of several runs. Our rows: mean of ten independent runs per
instance, every schedule counted as in Section~\ref{sec:counting}. ``n/a'': not yet measured on the complete sets.}
\label{tab:literature}
\small
\setlength{\tabcolsep}{3.2pt}
\begin{tabular}{lrrrrrrrrr}
\toprule
 & \multicolumn{3}{c}{J30} & \multicolumn{3}{c}{J60} & \multicolumn{3}{c}{J120}\\
\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-10}
Method & 1k & 5k & 50k & 1k & 5k & 50k & 1k & 5k & 50k\\
\midrule
""" + "\n".join(lr) + "\n\\midrule\n" + "\n".join(ours) + r"""
\bottomrule
\end{tabular}
\end{table*}
""")

# ---------------------------------------------------------------- stagnation csv for the figure
with open(DATA / "stagnation.csv", "w") as fcsv:
    fcsv.write("set V0 V1 V4 V5 V6\n")
    for k, b in enumerate(SETS):
        fcsv.write(f"{k} " + " ".join(f"{100 * M[b]['versions'][v]['stagnation']['budget_without_gain']:.1f}"
                                      for v in ("V0", "V1", "V4", "V5", "V6")) + "\n")

evidence = {"steps_inline": steps_inline, "confirm_ls_shake": shake, "direction_selection": {b: cb[b]["selection"] for b in SETS},
            "direction_shares": {b: [cb[b]["init_backward_wins_share"], cb[b]["init_tie_share"]] for b in SETS},
            "direction_gap": {b: cb[b]["gap"] for b in SETS},
            "ils_test_mech": it_mech, "ils_test_gap": {b: it[b]["gap"] for b in SETS},
            "ils_test_cmp": {b: it[b]["cmp"] for b in SETS},
            "brils2_test": load("br_ils2/test_summary.json"),
            "long": {"by_set_means": {b: {B: lg["by_set"][b]["schedules"][B]["mean_gap_pct_cpm"] for B in lg["by_set"][b]["schedules"]}
                                      for b in lg["by_set"]}, "contrasts": lg["primary_contrasts"],
                     "cpu": {k: lg[k] for k in ("total_measured_cpu_s", "total_measured_wall_s", "max_compile_s")}},
            "constructor": {"quality": q, "timing": t, "horizons": cc["horizons"]},
            "v5_gaps": {b: v5[b]["gap"] for b in SETS}, "v5_mech": {b: v5[b]["mech"] for b in SETS},
            "v5n_gap": v5n["gap"], "explore_extra": explore_extra,
            "V0_50000": {b: M[b]["V0_50000"] for b in SETS}}
(DATA / "evidence.json").write_text(json.dumps(evidence, indent=1))
(DATA / "source_hashes_tables.json").write_text(json.dumps(USED, indent=1, sort_keys=True))
print("tables:", sorted(p.name for p in TAB.glob("*.tex")))
