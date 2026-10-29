#!/usr/bin/env python3
"""Frozen, single-worker, fsync-resumable V0/V5/V6 campaign runner."""
from __future__ import annotations
import argparse, fcntl, hashlib, json, os, platform, random, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from run_smoke import feasible, rows
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def durable(p,x):
    with p.open('a') as f: f.write(json.dumps(x,sort_keys=True)+'\n'); f.flush(); os.fsync(f.fileno())
def complete(p):
    out=[]
    if not p.exists(): return out,None
    for n,line in enumerate(p.read_bytes().splitlines(keepends=True),1):
        try: out.append(json.loads(line))
        except json.JSONDecodeError: return out,n
    return out,None
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--campaign',required=True); ap.add_argument('--budget',type=int,choices=(5000,50000),required=True); ap.add_argument('--instances',nargs='+',required=True); ap.add_argument('--seed-indices',type=int,nargs='+',default=list(range(1,11))); ap.add_argument('--variants',nargs='+',choices=('V0','V5','V6'),default=('V0','V5','V6')); ap.add_argument('--queue-seed',type=int,default=20261028); ap.add_argument('--max-runs',type=int); a=ap.parse_args()
 out=ROOT/'results/campaigns'/a.campaign; out.mkdir(parents=True,exist_ok=True); lock=(out/'.lock').open('a')
 try: fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 except BlockingIOError: raise SystemExit('campaign already locked')
 blocks=[(i,s) for i in a.instances for s in a.seed_indices]; random.Random(a.queue_seed).shuffle(blocks); queue=[]
 for i,s in blocks:
  vs=list(a.variants); random.Random(f'{a.queue_seed}:{i}:{s}').shuffle(vs); queue += [[i,s,v] for v in vs]
 source=sorted({str(p.relative_to(ROOT)) for folder in ('scripts','harness','experiments','solvers/br-deh') for p in (ROOT/folder).rglob('*') if p.is_file() and p.suffix in ('.py','.jl')} | {'Project.toml','Manifest.toml'})
 affinity=sorted(os.sched_getaffinity(0)); os.sched_setaffinity(0,{affinity[0]})
 manifest={'schema':3,'budget_charged_cap':a.budget,'instances':a.instances,'seed_indices':a.seed_indices,'variants':a.variants,'queue_seed':a.queue_seed,'queue':queue,'input_sha256':{i:sha(ROOT/'solvers/br-deh/inputs'/i) for i in a.instances},'source_sha256':{x:sha(ROOT/x) for x in source},'platform':platform.platform(),'python':sys.version,'affinity':list(os.sched_getaffinity(0)),'timing_contract':'Total CPU/wall surrounds original input reading, RNG, policy/state initialization and solve. Warmup, validation decodes and export are separately measured and excluded.'}
 mp,rp=out/'manifest.json',out/'records.jsonl'
 if mp.exists() and json.loads(mp.read_text())!=manifest: raise SystemExit('manifest differs; use a new campaign name')
 if not mp.exists(): mp.write_text(json.dumps(manifest,indent=2)+'\n'); (out/'queue.json').write_text(json.dumps(queue,indent=2)+'\n')
 old,bad=complete(rp)
 if bad: (out/'INCIDENT-partial-jsonl.txt').write_text(f'partial line {bad}; preserved\n'); raise SystemExit('partial JSONL preserved for audit')
 done={(r['instance'],r['seed_index'],r['variant']) for r in old}; assert len(done)==len(old)
 assert len(old)<=len(queue), 'more records than frozen queue'
 for index, record in enumerate(old):
  assert record['order']==index and [record['instance'],record['seed_index'],record['variant']]==queue[index], 'records are not the exact frozen queue prefix'
 os.environ.setdefault('JULIA_NUM_THREADS','1'); os.environ.setdefault('JULIA_NUM_GC_THREADS','1'); os.environ.setdefault('OPENBLAS_NUM_THREADS','1'); os.environ.setdefault('OMP_NUM_THREADS','1')
 from harness.solvers.brdeh import canonical_seeds,jl,open_process
 from experiments import br_ils
 from experiments.bidir import write_reversed
 J=jl(); seeds=canonical_seeds(); ils=br_ils._setup_ils() if {'V5','V6'}&set(a.variants) else None; fns=(ils.seval('ado_gen_start'),ils.seval('ado_ils_run!')) if ils else None; made=0
 # Symmetric branch warmup outside analysis; each V6 crosses the stall.
 warm=[]
 for inst in a.instances:
  for var in a.variants:
   b=int(J.seval('SCHEDULE_COUNT[]')); c,w=time.process_time(),time.perf_counter()
   if var=='V0': p=open_process(inst,{'rules':'klein13','fbi_max_passes':1,'policy_keep_prob':.8},seeds[0]); p.advance(1000)
   else: br_ils.run_one(ils,fns,inst,write_reversed(inst),1,'B' if var=='V5' else 'br_r10_l5',1000)
   warm.append({'instance':inst,'variant':var,'cpu_s':time.process_time()-c,'wall_s':time.perf_counter()-w,'actual_sgs':int(J.seval('SCHEDULE_COUNT[]'))-b})
 durable(out/'setup-segments.jsonl', {'runtime':str(J.ado_runtime_info()),'warmups':warm,'resume_after_records':len(old),'started_at_unix':time.time()})
 J.ado_runtime_counters()
 for order,(inst,si,var) in enumerate(queue):
  if (inst,si,var) in done: continue
  if a.max_runs is not None and made>=a.max_runs: break
  rev=write_reversed(inst) if var!='V0' else None; revhash=sha(ROOT/'solvers/br-deh/inputs'/rev) if rev else None
  before=int(J.seval('SCHEDULE_COUNT[]')); gc0,compile0=J.ado_runtime_counters(); c0,w0=time.process_time(),time.perf_counter()
  if var=='V0':
   p=open_process(inst,{'rules':'klein13','fbi_max_passes':1,'policy_keep_prob':.8},seeds[si-1]); sl=p.advance(a.budget); cpu,wall=time.process_time()-c0,time.perf_counter()-w0; actual=int(J.seval('SCHEDULE_COUNT[]'))-before; charged_total=int(p.schedules); mk=int(p.best); trajectory=p.rows; counters=None; stop='iteration-boundary'
  else:
   arm='B' if var=='V5' else 'br_r10_l5'; s,e,c=br_ils.run_one(ils,fns,inst,rev,si,arm,a.budget); cpu,wall=time.process_time()-c0,time.perf_counter()-w0; actual=int(J.seval('SCHEDULE_COUNT[]'))-before; mk,st=ils.ado_best(e); mk=int(mk); starts=[int(x) for x in st]; trajectory=rows(ils,s); charged_total=int(s.schedules); counters=[int(x) for x in c]; stop='method-guard'
  gc1,compile1=J.ado_runtime_counters()
  compile_s=(int(compile1)-int(compile0))/1e9
  if compile_s>0:
   durable(out/'compilation-incidents.jsonl', {'order':order,'instance':inst,'variant':var,'compile_s':compile_s,'cpu_s':cpu,'wall_s':wall})
   raise RuntimeError('JIT occurred after warmup; preserve the incident and extend preflight warmup before freezing the full campaign')
  visible=[x for x in trajectory if x[1]<=a.budget]; assert visible; atcap=min(x[2] for x in visible)
  # Counter proof is before any validation decoder/conversion.
  expected=charged_total if var=='V0' else charged_total+3+(counters[1] if var=='V5' else 0)
  assert actual==expected,(var,actual,expected,counters)
  vb=int(J.seval('SCHEDULE_COUNT[]'))
  if var=='V0': vm,vs=J.ado_state_starts_at(p._s,atcap)
  else: vm,vs=J.ado_elite_starts_at(e,atcap)
  verify_sgs=int(J.seval('SCHEDULE_COUNT[]'))-vb; starts_atcap=[int(x) for x in vs]; assert int(vm)==atcap
  rec={'order':order,'instance':inst,'seed_index':si,'canonical_seed':seeds[si-1],'variant':var,'charged_cap':a.budget,'charged_total':charged_total,'actual_sgs_total':actual,'validation_sgs':verify_sgs,'makespan_terminal':mk,'makespan_at_cap':atcap,'starts_at_cap':starts_atcap,'trajectory':trajectory,'counters':counters,'stop_reason':stop,'cpu_s':cpu,'wall_s':wall,'feasible_at_cap':feasible(inst,atcap,starts_atcap),'reverse_input_sha256':revhash}
  if not rec['feasible_at_cap']: raise RuntimeError(f'feasibility failure {inst} {si} {var}')
  durable(rp,rec); made+=1; print(order,inst,si,var,charged_total,actual,round(cpu,3),flush=True)
 fcntl.flock(lock,fcntl.LOCK_UN)
if __name__=='__main__': main()
