#!/usr/bin/env python3
"""Validate structural, accounting, and feasibility evidence in a smoke artifact."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def main():
    p = argparse.ArgumentParser(); p.add_argument("artifact", type=Path, nargs="?", default=ROOT / "results/smoke/smoke.json"); p.add_argument("--expect", nargs="*", choices=("V0", "V5", "V6"))
    a = p.parse_args(); data = json.loads(a.artifact.read_text()); budget = data["budget_charged"]
    expected = set(a.expect or ()); found = {r["variant"] for r in data["runs"]}
    assert expected <= found, (expected, found)
    for r in data["runs"]:
        assert r["charged_schedules"] == budget
        assert r["executed_schedules"] <= budget + 2
        assert r["makespan"] > 0
        assert r["trajectory"]
        assert max(row[1] for row in r["trajectory"]) <= budget + 2
        if r["variant"] in {"V5", "V6"}:
            assert r["feasible"] is True
    print("verified", a.artifact)

if __name__ == "__main__": main()
