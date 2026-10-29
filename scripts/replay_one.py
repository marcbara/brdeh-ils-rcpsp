#!/usr/bin/env python3
"""One clean-process semantic replay; JSON request is the sole argument."""
from __future__ import annotations
import json, os, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from run_smoke import rows
def main():
 q=json.loads(sys.argv[1]); os.environ.setdefault('JULIA_NUM_THREADS','1'); os.environ.setdefault('JULIA_NUM_GC_THREADS','1')
 from harness.solvers.brdeh import canonical_seeds,jl,open_process
 from experiments import br_ils
 from experiments.bidir import write_reversed
 J=jl(); before=int(J.seval('SCHEDULE_COUNT[]')); seed=canonical_seeds()[q['seed']-1]
 if q['variant']=='V0':
  p=open_process(q['instance'],{'rules':'klein13','fbi_max_passes':1,'policy_keep_prob':.8},seed);p.advance(q['budget']); actual=int(J.seval('SCHEDULE_COUNT[]'))-before; at=min(x[2] for x in p.rows if x[1]<=q['budget']);dm,st=J.ado_state_starts_at(p._s,at);assert int(dm)==at; out={'trajectory':[list(x) for x in p.rows],'starts_at_cap':[int(x) for x in st],'counters':None,'charged_total':int(p.schedules),'actual_sgs_total':actual,'makespan_at_cap':at}
 else:
  f=br_ils._setup_ils(); arm='B' if q['variant']=='V5' else 'br_r10_l5';s,e,c=br_ils.run_one(f,(f.seval('ado_gen_start'),f.seval('ado_ils_run!')),q['instance'],write_reversed(q['instance']),q['seed'],arm,q['budget']); actual=int(f.seval('SCHEDULE_COUNT[]'))-before;rr=rows(f,s);at=min(x[2] for x in rr if x[1]<=q['budget']);mk,st=f.ado_elite_starts_at(e,at); out={'trajectory':[list(x) for x in rr],'starts_at_cap':[int(x) for x in st],'counters':[int(x) for x in c],'charged_total':int(s.schedules),'actual_sgs_total':actual,'makespan_at_cap':at}
 print(json.dumps(out,sort_keys=True))
if __name__=='__main__':main()
