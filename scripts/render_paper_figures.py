#!/usr/bin/env python3
"""Render the four manuscript figures from their standalone TeX sources."""
from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--historical-output', type=Path,
                        default=ROOT / 'results/reproduced-reader/historical-raw/generated')
    parser.add_argument('--outdir', type=Path, default=ROOT / 'results/paper-figures')
    args = parser.parse_args()
    if not shutil.which('pdflatex'):
        parser.error('Install a TeX distribution with standalone, TikZ/PGFPlots and Latin Modern.')
    if not args.historical_output.is_dir():
        parser.error('Run scripts/reproduce.py first, or pass --historical-output GENERATED_DIRECTORY.')
    output = args.outdir.resolve()
    (output / 'data').mkdir(parents=True, exist_ok=True)
    for source in args.historical_output.glob('*.csv'):
        shutil.copy2(source, output / 'data' / source.name)
    for index in range(1, 5):
        filename = f'figure{index}.tex'
        shutil.copy2(ROOT / 'figures/paper' / filename, output / filename)
        subprocess.run(['pdflatex', '-interaction=nonstopmode', '-halt-on-error', filename],
                       cwd=output, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        print(output / f'figure{index}.pdf')


if __name__ == '__main__':
    main()
