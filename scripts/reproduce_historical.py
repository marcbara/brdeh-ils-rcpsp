#!/usr/bin/env python3
"""Raw-record regeneration of the historical 5k/ablation snapshot."""
from __future__ import annotations
import argparse, json, os, subprocess, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
 p=argparse.ArgumentParser(); p.add_argument("--outdir",type=Path,default=ROOT/"results/historical-raw"); a=p.parse_args(); raw=a.outdir/"raw"; out=a.outdir/"generated"; out.mkdir(parents=True,exist_ok=True)
 subprocess.run([sys.executable,"scripts/unpack_historical.py","--outdir",str(raw)],cwd=ROOT,check=True)
 env={**os.environ,"HISTORICAL_RAW_ROOT":str(raw),"HISTORICAL_OUT":str(out)}
 for script in ("scripts/historical/extract.py","scripts/historical/extract_ils_pairs.py","scripts/historical/make_tables.py"):
  subprocess.run([sys.executable,script],cwd=ROOT,env=env,check=True)
 frozen=ROOT/"archived/data"; checks={}
 for n in ("main.json","evidence.json","ils_pairs.json"):
  aobj=json.loads((out/n).read_text()); bobj=json.loads((frozen/n).read_text())
  if n=="main.json": aobj.pop("conventions",None); bobj.pop("conventions",None)
  checks[n]=aobj==bobj
 if not all(checks.values()): raise SystemExit("historical numerical mismatch: "+str(checks))
 print("PASS: raw historical extraction matches frozen numeric snapshots",checks)
if __name__=="__main__": main()
