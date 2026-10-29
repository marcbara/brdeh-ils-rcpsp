# Paper-output coverage map

Commands below run from the repository root. `scripts/reproduce.py` is the advertised solver-free entrypoint; it runs both aggregation paths and performs the seven-file final-50k comparison.

| Paper output | Input and command | Coverage |
|---|---|---|
| Main Table 1 (configuration definitions) | Frozen Julia/Python method source and original extraction definitions | Method specification, not an empirical numerical table. |
| Main Table 2; Supplement 5k/ablation tables | 1,405 original numerical source files in `data/historical/historical-inputs.tar.gz`; `python scripts/reproduce_historical.py` | Preserved original extractors regenerate `main.json`, `evidence.json`, `ils_pairs.json`, profiles, plot CSVs and tables; numeric snapshots match. Main quality and direct ILS contrasts derive from per-run trajectories; supporting evidence also uses the original campaign summaries. |
| Main Tables 3--7 (ablations, direction, composite, ILS controls, long budget) | Same historical command and original `make_tables.py` closure | Full raw-record / original-summary derivation; source hashes are regenerated. |
| Main Table 8; Supplement S15--S18 | compressed `data/final50k`; `python analysis/final50k/analyse_final50k.py --migration data/final50k --inputs solvers/br-deh/inputs --j30-opt solvers/br-deh/experiments/j30_optimal.csv --outdir OUT` | Full raw-record derivation. Validates 46,800 keys and seven byte-identical expected CSVs. |
| Main Table 10 (CPU time) | Same final-50k analysis; 782 complete local paired blocks | Per-run CPU/wall observations, coverage and paired timing ratios regenerate from the bundled local records. |
| Main Table 9 and Supplement S12--S14 literature context | `archived/data/literature.csv`; aggregate command → `literature_context.csv` | Sourced context, not solver reproduction. |
| Algorithms and rule catalogue | `solvers/br-deh/src`, `experiments/*.py`, `experiments/extensions.jl` | Executable method source and original priority rules; these are specifications rather than numerical outputs. |
| Figures 1--4 | `figures/paper/figure*.tex`; `python scripts/render_paper_figures.py` after numerical reproduction | Original editable standalone sources, including the revised readable Figure 1. Plot CSVs come from the historical extractor. TeX is optional; rendering is separate from numerical regeneration and PDF bytes need not match across TeX versions. |

The historical archive contains all 1,405 files referenced by its source-hash maps (26,838,832 uncompressed bytes). `scripts/unpack_historical.py` checksum-validates and path-validates it before extraction.
