# Archived numerical evidence

The compact files in `data/` are paper-only aggregates copied from the historical manuscript data folder. `SOURCE.sha256` records their source-relative paths and hashes. They preserve reported values, including all variants V0--V6, exploratory comparisons, missing values, and literature context. They are not presented as independently recomputed raw campaigns.

Run `python scripts/regenerate_archived.py` to regenerate `generated/table_main.csv` and `generated/main_gap_5000.svg`. The script has no solver dependency and the expected SHA-256 values are recorded in the build report.
