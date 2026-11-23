# Heldout samples at other output fields of view (camera zoom test): payload -> samples (--fov) -> delete payload.
# Usage: build_fov.py OUT_ROOT KIND FOV [FOV ...]   KIND: turns (human moments of the turn heldout) or teacher (recovery heldout)
import shutil, subprocess, sys, zlib, json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import paths
ROOT, KIND, FOVS = Path(sys.argv[1]), sys.argv[2], sys.argv[3:]
FOOT = paths.FOOTAGE; TMP = Path('/mnt/nvme-500/alpamayo-train/payloads')
heldout = lambda seg: zlib.crc32(seg.rsplit('--', 1)[0].encode()) % 7 == 0
if KIND == 'turns':
  scan = json.load(open('/home/john/alpamayo-work/alpamayo-inputsweep-20261121/scan.json'))
  segs = sorted(k for k, v in scan.items() if v and 'turns' in v and (v['turns'] > 0 or v['straight'] >= 5) and heldout(k))
  extra, moments = [], '60'
else:
  segs = sorted(d.name for d in FOOT.iterdir() if d.name.split('--')[0] >= '0000009a' and heldout(d.name)
                and all((d/f).exists() for f in ('rlog.zst', 'fcamera.mp4', 'ecamera.mp4')))
  extra, moments = ['teacher'], '25'
here = str(paths.HERE)
def one(seg):
  pay = TMP/f'fov-{KIND}-{seg}'
  shutil.rmtree(pay, ignore_errors=True)
  r = subprocess.run([sys.executable, 'payload.py', str(pay), str(FOOT/seg), moments, *extra], cwd=here, capture_output=True, text=True)
  if r.returncode or '"moments": []' in (pay/'fixture.json').read_text() if (pay/'fixture.json').exists() else True:
    shutil.rmtree(pay, ignore_errors=True); return seg, 'no moments'
  for fov in FOVS:
    subprocess.run([sys.executable, 'samples.py', str(pay), str(ROOT/KIND/fov.replace(',', '-')), '--recovery', '0', '--fov', fov],
                   cwd=here, capture_output=True, text=True)
  shutil.rmtree(pay, ignore_errors=True)
  return seg, 'ok'
print(KIND, len(segs), 'heldout segments', FOVS, flush=True)
with ThreadPoolExecutor(4) as pool:
  for seg, st in pool.map(one, segs):
    print(seg, st, flush=True)
