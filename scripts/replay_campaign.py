#!/usr/bin/env python3
"""Replay every saved campaign key in a new Python/Julia process; clocks excluded."""
from __future__ import annotations
import argparse,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
 p=argparse.ArgumentParser();p.add_argument('campaign');p.add_argument('--max-runs',type=int);a=p.parse_args(); rec=[json.loads(x) for x in (ROOT/'results/campaigns'/a.campaign/'records.jsonl').read_text().splitlines()]; rec=rec[:a.max_runs] if a.max_runs else rec
 for r in rec:
  q=json.dumps({'instance':r['instance'],'seed':r['seed_index'],'variant':r['variant'],'budget':r['charged_cap']}); got=json.loads(subprocess.check_output([sys.executable,str(ROOT/'scripts/replay_one.py'),q],text=True))
  # Julia's legacy fourth trajectory column is a clock and is intentionally excluded.
  got['trajectory']=[x[:3] for x in got['trajectory']]; r['trajectory']=[x[:3] for x in r['trajectory']]
  for k in ('trajectory','starts_at_cap','counters','charged_total','actual_sgs_total','makespan_at_cap'):
   if got[k]!=r[k]: raise SystemExit(f"replay mismatch {r['instance']}:{r['seed_index']}:{r['variant']} {k}")
 print(f'replay passed {len(rec)} records')
if __name__=='__main__':main()
