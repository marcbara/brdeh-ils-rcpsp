#!/usr/bin/env python3
"""Freeze the protocol-required 36-key long preflight without launching it."""
from __future__ import annotations
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
# Lexical endpoints of the official on-disk PSPLIB filenames; frozen before launch.
fixtures=['j30/j3010_1.sm','j30/j309_9.sm','j60/j6010_1.sm','j60/j609_9.sm','j120/j12010_1.sm','j120/j1209_9.sm']
missing=[x for x in fixtures if not (ROOT/'solvers/br-deh/inputs'/x).exists()]
out=ROOT/'results/preflight-frozen.json'
out.write_text(json.dumps({'schema':1,'budget':50000,'variants':['V0','V5','V6'],'seed_indices':[1,10],'fixtures':fixtures,'expected_keys':36,'missing_inputs':missing,'queue_seed':20261028,'status':'frozen-not-launched'},indent=2)+'\n')
print(out)
