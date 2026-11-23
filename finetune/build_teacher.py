# Recovery data: comma's logged plan (modelV2) as the target on real Karel frames, including Alpamayo-engaged drift.
# payload.py teacher mode -> samples.py (120/30) -> delete payload, 6 CPU workers. Routes 9a+ (Alpamayo era), 25 moments per
# segment, no intersection turns; about 1 route in 7 goes to heldout (the recovery test set).
# Usage: build_teacher.py OUT_ROOT
import shutil, subprocess, sys, zlib, collections
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import paths
ROOT = Path(sys.argv[1]); FOOT = paths.FOOTAGE; TMP = Path('/mnt/nvme-500/alpamayo-train/payloads')
chosen = sorted(d.name for d in FOOT.iterdir() if d.name.split('--')[0] >= '0000009a' and all((d/f).exists() for f in ('rlog.zst', 'fcamera.mp4', 'ecamera.mp4')))
split = lambda seg: 'heldout' if zlib.crc32(seg.rsplit('--', 1)[0].encode()) % 7 == 0 else 'train'
here = str(paths.HERE)
def one(seg):
  done = ROOT/'done'/seg
  if done.exists(): return seg, 'skip'
  pay = TMP/f'teacher-{seg}'
  shutil.rmtree(pay, ignore_errors=True)
  r = subprocess.run([sys.executable, 'payload.py', str(pay), str(FOOT/seg), '25', 'teacher'], cwd=here, capture_output=True, text=True)
  if r.returncode: shutil.rmtree(pay, ignore_errors=True); return seg, 'payload failed: '+r.stderr.strip()[-160:]
  r = subprocess.run([sys.executable, 'samples.py', str(pay), str(ROOT/'samples'/split(seg)), '--recovery', '0', '--fov', '120,30'],
                     cwd=here, capture_output=True, text=True)
  shutil.rmtree(pay, ignore_errors=True)
  if r.returncode: return seg, 'samples failed: '+r.stderr.strip()[-160:]
  done.parent.mkdir(parents=True, exist_ok=True); done.write_text(r.stdout[-200:])
  return seg, r.stdout.strip().splitlines()[-1]
if __name__ == '__main__':
  print('segments', len(chosen), collections.Counter(split(s) for s in chosen), flush=True)
  with ThreadPoolExecutor(6) as pool:
    for seg, status in pool.map(one, chosen):
      print(seg, status, flush=True)
