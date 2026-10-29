#!/usr/bin/env python3
"""Materialize frozen 5k/ablation evidence as portable CSV files; never runs a solver."""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
def write(path, rows):
    rows=list(rows)
    with path.open("w", newline="") as f:
        w=csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["empty"]); w.writeheader(); w.writerows(rows)
def flatten(x, path=()):
    if isinstance(x,dict):
        for k in sorted(x): yield from flatten(x[k],path+(str(k),))
    elif isinstance(x,list):
        for i,v in enumerate(x): yield from flatten(v,path+(str(i),))
    else: yield {"path":"/".join(path),"value":x}
def main():
    p=argparse.ArgumentParser(); p.add_argument("--outdir",type=Path,default=ROOT/"archived"/"generated"); a=p.parse_args(); a.outdir.mkdir(parents=True,exist_ok=True)
    data=ROOT/"archived"/"data"; main=json.loads((data/"main.json").read_text()); rows=[]
    for b in ("j30","j60","j120"):
        for v in ("V0","V1","V2","V3","V4","V5","V6"):
            x=main["benches"][b]["versions"][v]; rows.append(dict(benchmark=b,variant=v,n_instances=main["benches"][b]["n_instances"],gap_1000_pct=x["mean"]["1000"],gap_2500_pct=x["mean"]["2500"],gap_5000_pct=x["mean"]["5000"],best10_5000_pct=x["best10_5000"],runs_at_reference_5000=x["runs_at_reference_5000"]))
    write(a.outdir/"main_5k.csv",rows); write(a.outdir/"table_main.csv",rows)
    write(a.outdir/"ablations_and_controls.csv",flatten(json.loads((data/"evidence.json").read_text())))
    write(a.outdir/"ils_heldout_pairs.csv",flatten(json.loads((data/"ils_pairs.json").read_text())))
    write(a.outdir/"literature_context.csv",csv.DictReader((data/"literature.csv").open()))
    colors={"V0":"#555555","V1":"#777777","V2":"#4c78a8","V3":"#f58518","V4":"#54a24b","V5":"#e45756","V6":"#b279a2"}; mx=max(r["gap_5000_pct"] for r in rows)
    parts=['<svg xmlns="http://www.w3.org/2000/svg" width="700" height="360" viewBox="0 0 700 360">','<style>text{font-family:sans-serif;font-size:12px}</style>','<text x="20" y="22">Frozen aggregate mean gap at 5,000 schedules (%)</text>']
    for bi,b in enumerate(("j30","j60","j120")):
        parts.append(f'<text x="{100+bi*190}" y="50">{b.upper()}</text>')
        for vi,v in enumerate(("V0","V1","V2","V3","V4","V5","V6")):
            z=next(r["gap_5000_pct"] for r in rows if r["benchmark"]==b and r["variant"]==v); h=z/mx*260; x=100+bi*190+vi*22; parts.append(f'<rect x="{x:.2f}" y="{330-h:.2f}" width="16" height="{h:.2f}" fill="{colors[v]}"/><text transform="translate({x+13:.2f},350) rotate(-60)">{v}</text>')
    (a.outdir/"main_gap_5000.svg").write_text("\n".join(parts+["</svg>"])+"\n")
    print(json.dumps({"outputs":[str(x) for x in sorted(a.outdir.iterdir())]}))
if __name__=="__main__": main()
