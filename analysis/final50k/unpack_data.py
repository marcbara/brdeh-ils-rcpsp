"""Unpack the bundled final50k records without accessing a private workspace."""
from pathlib import Path
import gzip
import shutil

root=Path(__file__).resolve().parents[2]
data=root/'data/final50k'
for source in sorted(data.glob('*.gz')):
    target=source.with_suffix('')
    if target.exists():
        continue
    with gzip.open(source,'rb') as src,target.open('wb') as dst:
        shutil.copyfileobj(src,dst)
    print(target.relative_to(root))
