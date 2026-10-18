"""Restore the exact published 100M-word input without model generation."""
import gzip, hashlib, json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
manifest=json.loads((ROOT/'reproduction/corpus_100m/manifest.json').read_text())
destination=ROOT/manifest['output']
destination.parent.mkdir(parents=True,exist_ok=True)
temporary=destination.with_suffix(destination.suffix+'.partial')
h=hashlib.sha256();total=0
with temporary.open('wb') as out:
    for chunk in manifest['chunks']:
        c=hashlib.sha256();n=0
        with gzip.open(ROOT/chunk['path'],'rb') as inp:
            while b:=inp.read(8*1024*1024):out.write(b);h.update(b);c.update(b);n+=len(b)
        assert c.hexdigest()==chunk['decoded_sha256'] and n==chunk['decoded_bytes']
        total+=n
assert h.hexdigest()==manifest['sha256'] and total==manifest['bytes']
temporary.replace(destination)
print(destination,total,h.hexdigest())
