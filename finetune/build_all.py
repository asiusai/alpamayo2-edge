# Build training samples for the top human-driven segments: payload -> samples -> delete payload. Parallel on the CPU.
import json, os, shutil, subprocess, sys, zlib, collections
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import paths
ROOT = Path('/mnt/nvme-500/alpamayo-train'); FOOT = paths.FOOTAGE
scan = json.load(open('/home/john/alpamayo-work/alpamayo-inputsweep-20261121/scan.json'))
ranked = sorted(((v['turns']*3+v['straight'], k) for k, v in scan.items() if v and 'turns' in v and v['turns']+v['straight'] >= 15), reverse=True)
per, chosen = collections.Counter(), []
for _, seg in ranked:
  route = seg.rsplit('--', 1)[0]
  if per[route] >= 2: continue
  per[route] += 1; chosen.append(seg)
chosen = chosen[:int(sys.argv[1]) if len(sys.argv) > 1 else 100]
split = lambda seg: 'heldout' if zlib.crc32(seg.rsplit('--', 1)[0].encode()) % 7 == 0 else 'train'
env = dict(os.environ)
def one(seg):
  out = ROOT/'samples'/split(seg)
  if (ROOT/'done'/seg).exists(): return seg, 'skip'
  pay = ROOT/'payloads'/seg
  shutil.rmtree(pay, ignore_errors=True)
  r = subprocess.run([sys.executable, 'payload.py', str(pay), str(FOOT/seg), '40'], cwd=str(paths.HERE), env=env, capture_output=True, text=True)
  if r.returncode: return seg, 'payload failed: '+r.stderr[-200:]
  r = subprocess.run([sys.executable, 'samples.py', str(pay), str(out), '--recovery', '2', '--seed', str(zlib.crc32(seg.encode()))],
                     cwd=str(paths.HERE), env=env, capture_output=True, text=True)
  shutil.rmtree(pay, ignore_errors=True)
  if r.returncode: return seg, 'samples failed: '+r.stderr[-200:]
  (ROOT/'done').mkdir(exist_ok=True); (ROOT/'done'/seg).write_text(r.stdout[-200:])
  return seg, r.stdout.strip().splitlines()[-1]
print('segments', len(chosen), collections.Counter(split(s) for s in chosen), flush=True)
with ThreadPoolExecutor(6) as pool:
  for seg, status in pool.map(one, chosen):
    print(seg, status, flush=True)
