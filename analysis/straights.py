# Per-frame lane geometry vs. the car and the newest Alpamayo plan (mapped to the current pose by odometry).
# Usage (opendbc-coop venv, PYTHONPATH=op-coop): straights.py OUT.npz SEGMENT_DIR [...]
# All lateral values are right positive, in the calibrated frame of comma's lane lines (modelV2.laneLines).
import math, sys, types
from pathlib import Path
import numpy as np, zstandard
sys.modules.setdefault('setproctitle', types.SimpleNamespace(getproctitle=lambda: '', setproctitle=lambda _: None))
from openpilot.cereal import log
from openpilot.selfdrive.alpamayo.planner import PlanSelector

XS = (0., 10., 20., 30., 40., 60.)
COLS = ['t', 'v', 'lat', 'pressed', 'torque', 'angle', 'p1', 'p2', 'width', 'edgeL', 'edgeR', 'age', 'k_cmd', 'k_meas',
        'blinkL', 'blinkR', 'seg'] + [f'c{int(x)}' for x in XS] + [f'p{int(x)}' for x in XS] + [f'm{int(x)}' for x in XS]


def events(seg):
  raw = zstandard.ZstdDecompressor().stream_reader((Path(seg)/'rlog.zst').open('rb')).read()
  try:
    evs = list(log.Event.read_multiple_bytes(raw))
  except Exception:  # truncated tail
    evs = []
    import capnp  # noqa: F401
    for e in log.Event.read_multiple_bytes(raw):
      evs.append(e)
  return sorted(evs, key=lambda m: m.logMonoTime)


def plan_in_current(ps, now):
  """Newest plan's path in the current vehicle frame: (forward, left) arrays, or None."""
  q, c = ps.pose_at(ps.query), ps.pose_at(now)
  if q is None or c is None or ps.path is None:
    return None
  x, y, _ = ps.path
  qx, qy, qh = q
  wx, wy = qx+math.cos(qh)*x-math.sin(qh)*y, qy+math.sin(qh)*x+math.cos(qh)*y
  px, py, ph = c
  fx = math.cos(ph)*(wx-px)+math.sin(ph)*(wy-py)
  fy = -math.sin(ph)*(wx-px)+math.cos(ph)*(wy-py)
  return fx, fy


def main():
  out, segs = sys.argv[1], sys.argv[2:]
  rows = []
  t0 = None
  for si, seg in enumerate(segs):
    try:
      evs = events(seg)
    except Exception as e:  # noqa: BLE001
      print(seg, 'unreadable', e); continue
    ps = PlanSelector()
    cs = cc = None
    for m in evs:
      w, t = m.which(), m.logMonoTime
      t0 = t0 or t
      if w == 'carState':
        cs = m.carState
      elif w == 'carControl':
        cc = m.carControl
      elif w == 'deviceMotion':
        stamp = m.deviceMotion.timestamp if m.deviceMotion.timestamp > 0 else t
        ps.motion(stamp, cs.vEgo if cs else 0., -m.deviceMotion.orientationNED.z)
      elif w == 'alpamayoPlan' and m.valid:
        ps.receive(m.alpamayoPlan, t+20_000_000, t)
      elif w == 'modelV2' and cs is not None and cc is not None:
        mv = m.modelV2
        ll, pr, re = mv.laneLines, mv.laneLineProbs, mv.roadEdges
        if len(ll) < 4 or len(re) < 2 or len(ll[1].x) == 0:
          continue
        at = lambda line, x: float(np.interp(x, line.x, line.y))
        cen = [(at(ll[1], x)+at(ll[2], x))/2 for x in XS]
        pl = plan_in_current(ps, t)
        if pl is not None and pl[0][-1] > 60:
          pp = [-float(np.interp(x, pl[0], pl[1])) for x in XS]
        else:
          pp = [np.nan]*len(XS)
        # modelV2's own plan position (comma model), for reference only
        mp = [float(np.interp(x, mv.position.x, mv.position.y)) if len(mv.position.x) else np.nan for x in XS]
        age = (t-ps.query)/1e9 if ps.query else np.nan
        k_meas = ps.measured_curvature(t, cs.vEgo)
        rows.append([(t-t0)/1e9, cs.vEgo, cc.latActive, cs.steeringPressed, cs.steeringTorque, cs.steeringAngleDeg,
                     pr[1], pr[2], at(ll[2], 0)-at(ll[1], 0), at(re[0], 0), at(re[1], 0), age,
                     mv.action.desiredCurvature, -k_meas, cs.leftBlinker, cs.rightBlinker, si] + cen + pp + mp)
    print(seg, len(rows), flush=True)
  np.savez(out, a=np.array(rows, dtype=float), cols=np.array(COLS))


if __name__ == '__main__':
  main()
