# Reproduce the completed 50,000-schedule analysis

Run these commands from the repository root with Python 3.12 and the tested dependencies in `requirements.txt` installed:

```sh
python analysis/final50k/unpack_data.py
python analysis/final50k/analyse_final50k.py \
  --migration data/final50k \
  --inputs solvers/br-deh/inputs \
  --j30-opt solvers/br-deh/experiments/j30_optimal.csv \
  --outdir results/reproduced-final50k
```

This regenerates the final long-budget quality means, nine paired statistical contrasts with one Holm correction family, and descriptive paired local timing results from the bundled compressed records. It does not execute any solver or require a private development repository. The parser is included in the analysis script. Numerical reference CSVs are in `expected/`.

The data comprise 46,800 unique instance/seed/variant keys across complete J30/J60/J120 sets, ten seeds and three versions. Quality merges compatible deterministic results from the original local run and its cloud continuation. Timing uses only the 782 complete three-version blocks from the original local queue; cloud seconds are never pooled with local seconds. The incomplete final local block is excluded from paired timing only, not from the quality dataset.

Quality differences are computed from integer makespan totals before normalization, preserving exact project ties. Confidence intervals are pointwise paired instance-bootstrap intervals; the nine 50k Wilcoxon comparisons form their own Holm family. The local timing sample is a complete paired prefix of a prospectively randomized queue interrupted for logistical migration, not a fixed-size confirmatory timing study. Multiple seeds of one instance are grouped in the timing uncertainty analysis.

The `.sm` files and known optimum CSV must match their frozen checksums. The tested local extraction currently contains these benchmark files, but redistribution permissions for the solver and PSPLIB inputs remain a release prerequisite. No license or public release is asserted here. Reproduction of this final50k analysis does not by itself claim complete regeneration coverage for every older paper figure/table.
