#!/usr/bin/env python3
"""Direct comparisons among ILS perturbations on the held-out test instances (br_ils test campaigns).

The campaign summaries only compare every arm with B; the attribution of the gain to the bias needs the direct
contrasts biased vs greedy and biased vs uniform. Read-only; writes data/ils_pairs.json.

    python scripts/reproduce_historical.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import extract as X  # noqa: E402

ARMS = ["B", "br_r10_l5", "grd_r10_l5", "uni_r10_l5", "rm_r10_l5", "ls_l5"]
out = {}
for b in ("j30", "j60", "j120"):
    tr = X.read_traj(sorted((X.RUNS / "br_ils").glob(f"traj_{b}test_shard*of*.csv")), ARMS)
    insts = sorted({k[0] for k in tr["B"]})
    ref = X.refs(insts)
    out[b] = {"n": len(insts), "gap": {}, "pairs": {}}
    for B in (1000, 2500, 5000):
        out[b]["gap"][B] = {a: X.set_gap(tr[a], insts, B, ref) for a in ARMS}
        pairs = [("br_r10_l5", "grd_r10_l5"), ("br_r10_l5", "uni_r10_l5"), ("grd_r10_l5", "uni_r10_l5"),
                 ("br_r10_l5", "ls_l5")]
        c = {f"{x} vs {y}": X.paired(tr[x], tr[y], insts, B, ref) for x, y in pairs}
        for (k, v), ph in zip(c.items(), X.holm([v["wilcoxon_p"] for v in c.values()])):
            v["p_holm_family"] = ph
        out[b]["pairs"][B] = c
    print(b, {k: (round(v["mean"], 3), round(v["wilcoxon_p"], 4)) for k, v in out[b]["pairs"][5000].items()}, flush=True)
(X.OUT / "ils_pairs.json").write_text(json.dumps(out, indent=1))
(X.OUT / "source_hashes_ils_pairs.json").write_text(json.dumps(X.HASHES, indent=1, sort_keys=True))
