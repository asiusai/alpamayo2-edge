# Lane geometry per modelV2 frame with absolute log time, for scoring offline plans against the lane centre.
# Usage (opendbc-coop venv, PYTHONPATH=op-coop): lane_table.py OUT.npz SEGMENT_DIR [...]
# Lateral values are right positive (modelV2.laneLines). Columns: see COLS.
import sys, types
from pathlib import Path
import numpy as np, zstandard
sys.modules.setdefault('setproctitle', types.SimpleNamespace(getproctitle=lambda: '', setproctitle=lambda _: None))
from openpilot.cereal import log

XS = (0., 10., 20., 30., 40., 60.)
COLS = ['t_ns', 'sof_ns', 'v', 'lat', 'pressed', 'angle', 'prob', 'width', 'blink'] + [f'c{int(x)}' for x in XS] + [f'm{int(x)}' for x in XS]
rows, names = [], []
for seg in sys.argv[2:]:
  raw = zstandard.ZstdDecompressor().stream_reader((Path(seg)/'rlog.zst').open('rb')).read()
  cs = cc = None
  it = log.Event.read_multiple_bytes(raw)
  while True:
    try:
      m = next(it)
    except StopIteration:
      break
    except Exception:  # truncated tail
      break
    w = m.which()
    if w == 'carState':
      cs = m.carState
    elif w == 'carControl':
      cc = m.carControl
    elif w == 'modelV2' and cs is not None and cc is not None:
      mv = m.modelV2
      ll, pr = mv.laneLines, mv.laneLineProbs
      if len(ll) < 4 or len(ll[1].x) == 0:
        continue
      at = lambda line, x: float(np.interp(x, line.x, line.y))
      rows.append([m.logMonoTime, mv.timestampEof, cs.vEgo, cc.latActive, cs.steeringPressed, cs.steeringAngleDeg, min(pr[1], pr[2]),
                   at(ll[2], 0)-at(ll[1], 0), cs.leftBlinker or cs.rightBlinker] + [(at(ll[1], x)+at(ll[2], x))/2 for x in XS] +
                  [float(np.interp(x, mv.position.x, mv.position.y)) if len(mv.position.x) else np.nan for x in XS])
      names.append(Path(seg).name)
  print(seg, len(rows), flush=True)
np.savez(sys.argv[1], a=np.array(rows, dtype=np.float64), seg=np.array(names), cols=np.array(COLS))
