#!/usr/bin/env python3
"""Safely unpack the bundled historical numeric input archive."""
from __future__ import annotations
import argparse, hashlib, tarfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
 p=argparse.ArgumentParser(); p.add_argument("--outdir",type=Path,required=True); a=p.parse_args()
 archive=ROOT/"data/historical/historical-inputs.tar.gz"; expected="fecc61b024f072f8d917938bee76e86099d20066214520779fc19640424910fe"
 if hashlib.sha256(archive.read_bytes()).hexdigest()!=expected: raise SystemExit("historical archive checksum mismatch")
 with tarfile.open(archive,"r:gz") as t:
  members=t.getmembers()
  if any(m.name.startswith("/") or ".." in Path(m.name).parts or m.issym() or m.islnk() for m in members): raise SystemExit("unsafe archive member")
  t.extractall(a.outdir, members, filter="data")
 print(f"unpacked {len(members)} historical inputs to {a.outdir}")
if __name__=="__main__": main()
