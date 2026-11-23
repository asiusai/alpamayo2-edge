#!/usr/bin/env python3
"""Generate define variants of the frozen vision GEMM template and compile them. Writes kern/<name>.cpp/.hsaco and kern/<set>.json."""
import json, re, subprocess, sys
from pathlib import Path
A = Path('src/openpilot/selfdrive/alpamayo/selected/assets/gfx1200')
BASES = {'fc2': 'f5e4e80f', 'qkv': 'cf644377', 'o': '5fbdf8a5', 'fc1': 'c85f36b4'}
def defs(src): return {k: int(v) for k, v in re.findall(r'^#define (\w+) (\d+)$', src, re.M)}
def launch(d):
  mb, nb = d['M']//16, (d['N']+15)//16
  mt, nt = -(-mb//d['TM']), -(-nb//d['TN'])
  return -(-mt//d['WM'])*-(-nt//d['WN']), 32*d['WM']*d['WN']
def make(kernel, name, **changes):
  src = next(A.glob(BASES[kernel]+'*.cpp')).read_text()
  for k, v in changes.items():
    src, n = re.subn(rf'^#define {k} \d+$', f'#define {k} {v}', src, count=1, flags=re.M); assert n == 1, k
  grid, local = launch(defs(src))
  src = re.sub(r'^// LAUNCH \d+ \d+', f'// LAUNCH {grid} {local}', src, count=1)
  Path(f'kern/{name}.cpp').write_text(src)
  out = subprocess.run([sys.executable, str(Path(__file__).resolve().with_name('hipc.py')), f'kern/{name}.cpp', f'kern/{name}.hsaco'], capture_output=True, text=True)
  ok = out.returncode == 0
  print(name, changes, 'launch', grid, local, 'ok' if ok else out.stderr[-300:])
  return {'grid': grid, 'local': local} if ok else None
SETS = {
  'fc2': [dict(PF=1), dict(PF=2), dict(PF=4), dict(TM=3, TN=4), dict(TM=2, TN=6), dict(TM=3, TN=8), dict(TM=5, TN=4), dict(TM=6, TN=4), dict(WM=2), dict(WN=2)],
  'qkv': [dict(PF=1), dict(PF=2), dict(PF=4), dict(TM=3, TN=6), dict(TM=2, TN=6), dict(TM=3, TN=3), dict(TM=5, TN=4), dict(TM=6, TN=4), dict(WM=2), dict(WN=2)],
  'o': [dict(PF=1), dict(PF=2), dict(PF=4), dict(TM=3, TN=6), dict(TM=3, TN=4), dict(TM=2, TN=4), dict(TM=5, TN=4), dict(WM=2), dict(WN=2)],
}
for kernel in sys.argv[1:]:
  res = {}
  for ch in SETS[kernel]:
    name = kernel+'_'+'_'.join(f'{k.lower()}{v}' for k, v in ch.items())
    r = make(kernel, name, **ch)
    if r: res[name] = r
  Path(f'kern/{kernel}.json').write_text(json.dumps(res))
