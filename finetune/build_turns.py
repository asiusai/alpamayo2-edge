# Turn dataset: every human-driven moment of every segment with a turn (approaches included) and of straight segments.
# Navigation text is always the default 'Follow the road.' (no turn-signal or lane-change intents).
# Views use the given output field of view (default 180,60: the lenses' native coverage). No recovery samples.
# Usage: build_turns.py OUT_ROOT [FOV] [--also-120 heldout]   payload -> samples -> delete payload, 6 CPU workers.
import json, os, shutil, subprocess, sys, zlib, collections
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import paths
ROOT = Path(sys.argv[1]); FOV = sys.argv[2] if len(sys.argv) > 2 and not sys.argv[2].startswith('--') else '180,60'
ALSO_120 = '--also-120' in sys.argv  # heldout segments also get the stock 120/30 views, for zero-shot comparison
FOOT = paths.FOOTAGE; TMP = Path('/mnt/nvme-500/alpamayo-train/payloads')
scan = json.load(open('/home/john/alpamayo-work/alpamayo-inputsweep-20261121/scan.json'))
# Turning segments, plus straight-only segments with at least 5 human-driven moments, so straight driving trains as much.
chosen = sorted(k for k, v in scan.items() if v and 'turns' in v and (v['turns'] > 0 or v['straight'] >= 5))
split = lambda seg: 'heldout' if zlib.crc32(seg.rsplit('--', 1)[0].encode()) % 7 == 0 else 'train'
here = str(paths.HERE)
def one(seg):
  done = ROOT/'done'/seg
  if done.exists(): return seg, 'skip'
  pay = TMP/f'turns-{seg}'
  shutil.rmtree(pay, ignore_errors=True)
  r = subprocess.run([sys.executable, 'payload.py', str(pay), str(FOOT/seg), '60'], cwd=here, capture_output=True, text=True)
  if r.returncode: return seg, 'payload failed: '+r.stderr[-200:]
  runs = [(FOV, ROOT/'samples'/split(seg))] + ([('120,30', ROOT/'samples-120'/'heldout')] if ALSO_120 and split(seg) == 'heldout' else [])
  for fov, out in runs:
    r = subprocess.run([sys.executable, 'samples.py', str(pay), str(out), '--recovery', '0', '--fov', fov], cwd=here, capture_output=True, text=True)
    if r.returncode: break
  shutil.rmtree(pay, ignore_errors=True)
  if r.returncode: return seg, 'samples failed: '+r.stderr[-200:]
  done.parent.mkdir(parents=True, exist_ok=True); done.write_text(r.stdout[-200:])
  return seg, r.stdout.strip().splitlines()[-1]
if __name__ == '__main__':
  print('segments', len(chosen), collections.Counter(split(s) for s in chosen), 'fov', FOV, flush=True)
  with ThreadPoolExecutor(6) as pool:
    for seg, status in pool.map(one, chosen):
      print(seg, status, flush=True)
