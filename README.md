# BR-DEH + ILS: reader reproducibility artifact

This artifact reproduces the paper's stored numerical analyses without rerunning a solver. It also contains the frozen Julia/Python implementation and optional smoke/campaign drivers. `V0` is the BR-DEH-2024-derived multistart baseline (without terminal path relinking), `V5` is BR-DEH 2026, and `V6` is BR-DEH 2026 plus the permanent-switch biased-rebuild ILS arm. Historical `ado_*` Julia names are retained to preserve behavior.

## Quick numerical reproduction (no solver run)

Python 3.12 with `requirements.txt` is required. The recorded clean environment is Python 3.12.14; create a fresh environment and install the pinned requirements.

```sh
python -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python scripts/reproduce.py --outdir results/reproduced-reader
```

This expands the bundled historical inputs and 50k records, runs the original 5k/ablation extractors, checks their numerical snapshots, and checks that seven 50k CSV outputs are byte-identical to `analysis/final50k/expected/`. It does not invoke Julia or a solver. See `docs/COVERAGE.md` for the output-by-output mapping and the distinction between per-run records, original campaign summaries and literature context.

The four manuscript figures also have editable standalone TeX sources. After the numerical reproduction, an optional TeX installation with `standalone`, TikZ/PGFPlots and Latin Modern can render them:

```sh
.venv/bin/python scripts/render_paper_figures.py
```

This writes `results/paper-figures/figure1.pdf` through `figure4.pdf` from the preserved sources and regenerated plot data. PDF bytes can differ between TeX installations.

## Optional method execution

Julia 1.11.9 and the root `Project.toml`/`Manifest.toml` are pinned. Provide your own writable Julia depot and executable; no private absolute path is needed.

```sh
export JULIA_DEPOT_PATH="$PWD/.julia-depot"
export PYTHON_JULIACALL_EXE="$(command -v julia)"
export PYTHON_JULIACALL_PROJECT="$PWD"
python scripts/run_smoke.py --budget 1000
python scripts/verify_smoke.py
```

Full PSPLIB campaigns are optional and never run by the reproduction command. They require the official PSPLIB corpus in `solvers/br-deh/inputs/{j30,j60,j120}`:

```sh
python scripts/run_campaign.py --campaign selected-5k --budget 5000 --instances j30/j301_1.sm --variants V0 V5 V6
```

This is a new computation, not a check of archived paper numbers. Read `docs/DATA_POLICY.md`, `PROVENANCE.md`, and `LICENSE-STATUS.md` before redistribution.
