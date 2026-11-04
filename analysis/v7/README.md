# V7: weighted LFT/HWRU hybrid rule (follow-up to V6)

This folder holds a follow-up experiment that is **not part of the paper's reported results**. It sits on the `v7sb` branch, so `main` keeps matching the paper.

V6 uses the hybrid priority rule with a fixed equal weight:

`score = w · LFT_norm + (1 − w) · (1 − HWRU_norm)`, with w = 0.5.

V7sb lets the weight w vary. Everything else is V6 as published (arm `br_r10_l5`): the same budget accounting, BR-DEH phase, ILS window rebuild, stall and cap parameters, and seeds.

| Phase | V6 | V7sb |
|---|---|---|
| Start | 1 rule (w = 0.5) × 2 directions × 3 = 6 schedules | 11 rules (w = 0, 0.1, …, 1) × 2 directions × 3 = 66 schedules |
| BR-DEH | one rule | keep the favourite weight with probability 0.8, otherwise try a random one; a weight that improves becomes the favourite |
| ILS | w = 0.5 | w drawn uniformly from the 11 weights at every perturbation |

## Ablation arms

| Arm | Start | BR-DEH | ILS |
|---|---|---|---|
| V7sb | 11 weights | keep probability 0.8 | weight sampled per perturbation |
| V7s | 11 weights | keep probability 0.8 | w = 0.5, as V6 |
| V7b | as V6 | as V6 | weight sampled per perturbation |
| V7sbk1 | 11 weights | keep probability 1.0 (never switch) | weight sampled per perturbation |

## Results

The protocol is the same as the paper's 50k tables:

- the complete J30, J60 and J120 sets (480, 480 and 600 instances);
- seeds 1–10 and 50 000 schedules;
- V0, V5 and V6 taken from the archived runs in `data/final50k/`.

Gaps reproduce the paper exactly for those three (for example, V6 is 32.425 % on J120 at 50k).

Mean gap (%), from [`expected/v7_gaps.csv`](expected/v7_gaps.csv):

| Set | Budget | V0 | V5 | V6 | **V7sb** | V7s | V7b | V7sbk1 |
|---|---|---|---|---|---|---|---|---|
| J30 | 50k | 0.074 | 0.043 | 0.072 | **0.064** | 0.065 | 0.074 | 0.068 |
| J60 | 1k | 12.377 | 11.936 | 11.936 | **11.841** | 11.840 | 11.932 | 11.837 |
| J60 | 50k | 11.558 | 11.226 | 10.987 | **10.968** | 10.967 | 10.968 | 10.976 |
| J120 | 1k | 36.184 | 35.034 | 35.029 | **34.727** | 34.725 | 35.038 | 34.741 |
| J120 | 5k | 35.319 | 34.112 | 33.920 | **33.765** | 33.770 | 33.938 | 33.776 |
| J120 | 50k | 34.469 | 33.256 | 32.425 | **32.342** | 32.364 | 32.407 | 32.345 |

Primary contrast V7sb − V6, from [`expected/v7_contrasts.csv`](expected/v7_contrasts.csv). The table gives the gap difference in percentage points, with the Holm-adjusted p-value over these 12 tests in brackets:

| Set | 1k | 5k | 25k | 50k |
|---|---|---|---|---|
| J30 | −0.013 (1) | −0.012 (1) | −0.010 (1) | −0.008 (1) |
| J60 | −0.095 (0.013) | −0.050 (0.23) | −0.032 (0.28) | −0.019 (1) |
| J120 | −0.302 (6e-16) | −0.155 (2e-5) | −0.097 (0.004) | −0.083 (0.004) |

Reading of the results:

- **Where V7sb helps:** it improves V6 on J120 at every budget and on J60 at 1k. On J30 there is no difference.
- **What causes it:** almost all of the gain comes from the 11-weight start. V7s matches V7sb at every budget, with no significant difference.
- **The ILS sampling:** on its own (V7b) it does not improve on V6 significantly anywhere.
- **The 20 % switch:** removing it (V7sbk1) changes nothing. No V7sbk1 − V7sb difference is significant, and all of them are below 0.02 pp.

[`report.html`](report.html) is a self-contained page with figures that explains the method and these results. Its confidence intervals come from an earlier version of the bootstrap. They differ from `v7_contrasts.csv` in the second decimal, but the means, p-values and conclusions are the same.

## Files

| Path | Content |
|---|---|
| `experiments/v7_weighted.jl` | weighted rule, 11-weight start, BR-DEH rule pool, ILS with weight sampling |
| `experiments/v7_driver.jl` | driver for V6 and V7sb, V7s and V7b, without Python |
| `experiments/v7k_driver.jl` | adds V7sbk1 (keep probability 1.0); otherwise calls `v7_driver.jl` |
| `experiments/escoto_hybrid_policies_90f8ab1.jl` | LFT/HWRU hybrid policies, a copy of commit `90f8ab1` of `Adaptative-BR-DEH-RCPSP` (`rule_based_decisions`) |
| `experiments/v7sb_frozen.sha256` | hashes of the code as frozen before the test runs (2026-10-30) |
| `analysis/v7/data/<arm>_runs.csv.gz` | one row per run, 15 600 per arm, all feasible (checked against the instance) |
| `analysis/v7/data/<arm>_traj.csv.gz` | improvement trajectory (schedules, makespan) per run |
| `analysis/v7/analyse_v7.js` | computes everything in `expected/` from the data above and `data/final50k/` |
| `analysis/v7/expected/` | `v7_gaps.csv`, `v7_contrasts.csv`, `v7_mechanism.csv` (weight use, ILS counters, CPU) |
| `analysis/v7/run_campaign.sh` | reruns one arm on all 1560 instances |
| `analysis/v7/julia-env/` | the Julia environment used (Julia 1.10.4, same package versions as the root `Project.toml`, without PythonCall) |

## Reproduce the analysis (no solver run)

```sh
node analysis/v7/analyse_v7.js results/v7-check
diff -r analysis/v7/expected results/v7-check
```

This needs Node.js 18 or later, and takes about 10 s.

The statistics are computed as follows:

- **Wilcoxon test:** paired signed-rank over instances, using the normal approximation and dropping zero differences.
- **Confidence intervals:** 95 % percentile bootstrap with 10 000 resamples and a fixed seed.
- **Holm correction:** applied within each family. The families are the primary family (12 tests), ablation vs V6 (36) and ablation vs V7sb (36).

## Rerun the solver

```sh
julia --project=analysis/v7/julia-env -e 'using Pkg; Pkg.instantiate()'
# one instance, two seeds (a few seconds):
printf "j120/j1201_3.sm\n" > /tmp/one.txt
julia --project=analysis/v7/julia-env experiments/v7_driver.jl /tmp/one.txt 1:2 V7sb 50000 /tmp/one
# a whole arm (about 6 h with 6 processes):
analysis/v7/run_campaign.sh V7sb 6
```

Runs are deterministic. Seed index s gives:

- BR-DEH seed `SEEDS[s]`;
- tie-break seed `1_000_003·s + 7`;
- ILS seed `7919·SEEDS[s] + 13`.

A fresh run on a separate checkout reproduced the stored rows exactly for V7sb, V7s, V7b and V7sbk1. The check used j30/j301_6 and j120/j1201_3, with seeds 1–2.

## Caveats

- **Tuning instances.** The design choices were made on development runs: J60/J120 instances `_1` and `_2`, seeds 21–30. The frozen code was then tested on the rest, with seeds 1–10. The tables above include `_1`/`_2` so they are comparable with the paper. On the held-out instances alone, the J120 result is the same (V7sb is better than V6 at every budget).
- **CPU time.** The runs were made on a different machine from the archived V0/V5/V6, so CPU times in `v7_mechanism.csv` should only be compared between V7 arms.
- **Julia version.** The runs used Julia 1.10.4, not the 1.11.9 pinned for the paper. V6 rerun with this setup gives the same trajectories as the archive (checked on j30/j301_6 and j120/j1201_3, seeds 1–2).
- **Hashes.** `v7sb_frozen.sha256` hashes the files with LF line endings, as git stores them. On a Windows checkout with `core.autocrlf=true`, check them with `git show :<path> | sha256sum`.
