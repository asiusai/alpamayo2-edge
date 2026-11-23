# comma5k comma four (mici) segments -> samples at 180/60 and 120/30: comma payload -> samples -> delete payload, 6 CPU workers.
# Every human-driven moment (cruise off, >= 4 m/s, 1 s grid); "Follow the road."; about 1 segment in 7 goes to heldout.
# Usage: build_comma.py OUT_ROOT
import json, shutil, subprocess, sys, zlib, collections
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import paths
ROOT = Path(sys.argv[1]); DATA = Path('/mnt/nvme-500/comma5k/data'); TMP = Path('/mnt/nvme-500/alpamayo-train/payloads')
labels = json.load(open('/mnt/nvme-500/comma5k/labels.json'))
chosen = sorted(k for k, v in labels.items() if v.get('device') == 'mici' and (DATA/k/'localizer.safetensors').exists())
split = lambda seg: 'heldout' if zlib.crc32(seg.encode()) % 7 == 0 else 'train'
here = str(paths.HERE)
def one(seg):
  done = ROOT/'done'/seg
  if done.exists(): return seg, 'skip'
  pay = TMP/f'comma-{seg}'
  shutil.rmtree(pay, ignore_errors=True)
  r = subprocess.run([sys.executable, 'comma_payload.py', str(pay), str(DATA/seg), '60'], cwd=here, capture_output=True, text=True)
  if r.returncode: shutil.rmtree(pay, ignore_errors=True); return seg, 'payload failed: '+r.stderr.strip()[-160:]
  if '"moments": []' in (pay/'fixture.json').read_text():
    shutil.rmtree(pay, ignore_errors=True); done.parent.mkdir(parents=True, exist_ok=True); done.write_text('no human moments'); return seg, 'no human moments'
  for fov, sub in (('180,60', 'samples'), ('120,30', 'samples-120')):
    r = subprocess.run([sys.executable, 'samples.py', str(pay), str(ROOT/sub/split(seg)), '--recovery', '0', '--fov', fov], cwd=here, capture_output=True, text=True)
    if r.returncode: break
  shutil.rmtree(pay, ignore_errors=True)
  if r.returncode: return seg, 'samples failed: '+r.stderr.strip()[-160:]
  done.parent.mkdir(parents=True, exist_ok=True); done.write_text(r.stdout[-200:])
  return seg, r.stdout.strip().splitlines()[-1]
if __name__ == '__main__':
  print('segments', len(chosen), collections.Counter(split(s) for s in chosen), flush=True)
  with ThreadPoolExecutor(6) as pool:
    for seg, status in pool.map(one, chosen):
      print(seg, status, flush=True)
