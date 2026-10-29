#!/usr/bin/env python3
"""One-command, solver-free numerical reproduction of archived paper outputs."""
from __future__ import annotations
import argparse,filecmp,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
EXPECTED=("final50k_means.csv","final50k_contrasts.csv","final50k_sensitivity.csv","longprefix_5k_summary.csv","local_timing_raw.csv","local_timing_balanced.csv","local_timing_ratios.csv")
def main():
 p=argparse.ArgumentParser(); p.add_argument("--outdir",type=Path,default=ROOT/"results"/"reproduced-reader"); a=p.parse_args(); a.outdir.mkdir(parents=True,exist_ok=True)
 subprocess.run([sys.executable,"scripts/regenerate_archived.py","--outdir",str(a.outdir/"archived-5k")],cwd=ROOT,check=True)
 subprocess.run([sys.executable,"scripts/reproduce_historical.py","--outdir",str(a.outdir/"historical-raw")],cwd=ROOT,check=True)
 subprocess.run([sys.executable,"analysis/final50k/unpack_data.py"],cwd=ROOT,check=True)
 subprocess.run([sys.executable,"analysis/final50k/analyse_final50k.py","--migration","data/final50k","--inputs","solvers/br-deh/inputs","--j30-opt","solvers/br-deh/experiments/j30_optimal.csv","--outdir",str(a.outdir)],cwd=ROOT,check=True)
 bad=[x for x in EXPECTED if not filecmp.cmp(a.outdir/x,ROOT/"analysis/final50k/expected"/x,shallow=False)]
 if bad: raise SystemExit("final50k regression mismatch: "+", ".join(bad))
 print("PASS: 7 final50k CSV outputs byte-identical; archived aggregate CSVs generated at",a.outdir/"archived-5k")
if __name__=="__main__": main()
