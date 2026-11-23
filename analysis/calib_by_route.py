# Per-route extrinsics calibration (rpyCalib, degrees) from each route's last and first segment.
# Usage (opendbc-coop venv, PYTHONPATH=op-coop): calib_by_route.py ROUTE [...]
import math, sys, types
from pathlib import Path
import numpy as np, zstandard
sys.modules.setdefault('setproctitle', types.SimpleNamespace(getproctitle=lambda: '', setproctitle=lambda _: None))
from openpilot.cereal import log
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'finetune'))
from paths import FOOTAGE as FOOT  # car footage root: ALPAMAYO_FOOTAGE or finetune/paths.local.json
for r in sys.argv[1:]:
  segs = sorted(FOOT.glob(r+'--*--*'), key=lambda s: int(s.name.rsplit('--', 1)[1]))
  out = []
  for seg in (segs[0], segs[-1]) if len(segs) > 1 else segs:
    try:
      raw = zstandard.ZstdDecompressor().stream_reader((seg/'rlog.zst').open('rb')).read()
    except Exception as e:
      out.append(f'{seg.name[-3:]} unreadable'); continue
    rows = []
    it = log.Event.read_multiple_bytes(raw)
    while True:
      try: m = next(it)
      except Exception: break
      if m.which() == 'extrinsicsCalibration':
        c = m.extrinsicsCalibration
        rows.append((*[math.degrees(v) for v in c.rpyCalib], int(c.calPerc), c.calStatus.raw, *[math.degrees(v) for v in c.wideFromDeviceEuler]))
    if rows:
      a = np.array(rows)
      out.append(f'seg {seg.name.rsplit("--",1)[1]:>3s}: pitch {np.median(a[:,1]):+.2f} yaw {np.median(a[:,2]):+.3f} (first {a[0,2]:+.3f} last {a[-1,2]:+.3f}) perc {int(a[-1,3])} status {int(a[-1,4])} wide yaw {np.median(a[:,7]):+.3f}')
  print(r, len(segs), '|', ' | '.join(out), flush=True)
