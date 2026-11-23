# Replay logged drives through the live PlanSelector with and without lane centring.
# Usage (opendbc-coop venv, PYTHONPATH=op-coop): centering_replay.py OUT.npz WEIGHT SEGMENT_DIR [...]
# Per modelV2 frame: t, v, latActive, pressed, blinker, logged curvature, replayed curvature (no centring),
# replayed curvature (centring WEIGHT), lane centre at 0/20 m (right+), lane heading error, lane probs.
import sys, types
from pathlib import Path
import numpy as np, zstandard
sys.modules.setdefault('setproctitle', types.SimpleNamespace(getproctitle=lambda: '', setproctitle=lambda _: None))
from openpilot.cereal import log
import importlib.util, os
_spec = importlib.util.spec_from_file_location('planner_under_test', os.environ['PLANNER_PATH'])
_mod = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_mod)
PlanSelector = _mod.PlanSelector

LAT, LONG = .35, .3


def main():
  out, weight, segs = sys.argv[1], float(sys.argv[2]), sys.argv[3:]
  rows, t0 = [], None
  for si, seg in enumerate(segs):
    try:
      raw = zstandard.ZstdDecompressor().stream_reader((Path(seg)/'rlog.zst').open('rb')).read()
      evs = sorted(log.Event.read_multiple_bytes(raw), key=lambda m: m.logMonoTime)
    except Exception as e:  # noqa: BLE001
      print(seg, 'unreadable', e); continue
    base, cen = PlanSelector(), PlanSelector(lane_centering=weight, plan_curvature_bias=float(os.environ.get('BIAS', '0')))
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
        for p in (base, cen):
          p.motion(stamp, cs.vEgo if cs else 0., -m.deviceMotion.orientationNED.z)
      elif w == 'alpamayoPlan' and m.valid:
        for p in (base, cen):
          p.receive(m.alpamayoPlan, t+20_000_000, t)
      elif w == 'modelV2' and cs is not None and cc is not None:
        mv = m.modelV2
        blink = cs.leftBlinker or cs.rightBlinker
        cen.observe_lanes(mv, t, blink)
        kb = base.candidate_action(t, cs.vEgo, LAT, LONG)
        kc = cen.candidate_action(t, cs.vEgo, LAT, LONG)
        ll, pr = mv.laneLines, mv.laneLineProbs
        if len(ll) < 4 or len(ll[1].x) == 0:
          continue
        at = lambda line, x: float(np.interp(x, line.x, line.y))
        c0, c20 = (at(ll[1], 0)+at(ll[2], 0))/2, (at(ll[1], 20)+at(ll[2], 20))/2
        rows.append([(t-t0)/1e9, cs.vEgo, cc.latActive, cs.steeringPressed, blink, mv.action.desiredCurvature,
                     kb['desiredCurvature'] if kb else np.nan, kc['desiredCurvature'] if kc else np.nan,
                     c0, c20, (c20-c0)/20, pr[1], pr[2], si, cs.steeringAngleDeg])
    print(seg, len(rows), flush=True)
  np.savez(out, a=np.array(rows, dtype=float),
           cols=np.array(['t', 'v', 'lat', 'pressed', 'blink', 'k_log', 'k_base', 'k_cen', 'c0', 'c20', 'head', 'p1', 'p2', 'seg', 'angle']))


if __name__ == '__main__':
  main()
