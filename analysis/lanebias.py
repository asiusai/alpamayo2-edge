# Measurement only: car and Alpamayo path relative to the small model's lane centre (right positive), straight engaged driving.
import sys, glob, numpy as np
from openpilot.tools.lib.logreader import LogReader
route = sys.argv[1]
segs = sorted(glob.glob(f'/data/media/0/realdata/{route}--*'), key=lambda p: int(p.rsplit('--', 1)[1]))
cs = cc = ap = None; rows = []; t0 = None
for s in segs:
  f = glob.glob(s+'/rlog*')
  if not f: continue
  for m in LogReader(f[0]):
    w = m.which(); t = m.logMonoTime; t0 = t0 or t
    if w == 'carState': cs = m.carState
    elif w == 'carControl': cc = m.carControl
    elif w == 'alpamayoPlan' and m.valid and len(m.alpamayoPlan.y) == 64: ap = m.alpamayoPlan
    elif w == 'modelV2' and cs is not None and cc is not None and ap is not None:
      mv = m.modelV2
      if not cc.latActive or cs.vEgo < 8 or cs.steeringPressed or abs(cs.steeringAngleDeg) > 6: continue
      ll, pr = mv.laneLines, mv.laneLineProbs
      if len(ll) < 4 or pr[1] < .6 or pr[2] < .6: continue
      at = lambda line, x: float(np.interp(x, line.x, line.y))
      if not 2.5 < at(ll[2], 0)-at(ll[1], 0) < 4.5: continue
      ax, ay = np.array(ap.x), -np.array(ap.y)
      vals = [(t-t0)/1e9, cs.vEgo]
      for d in (0, 20, 40, 60):
        vals += [(at(ll[1], d)+at(ll[2], d))/2, float(np.interp(d, ax, ay)) if ax[-1] > d else np.nan]
      rows.append(vals)
a = np.array(rows)
print('frames', len(a), f'({len(a)/20:.0f} s straight engaged with lanes)')
off = -a[:, 2]
print(f'car offset from lane centre (right+): mean {off.mean():+.2f} m, |offset| median {np.median(np.abs(off)):.2f} m, p90 {np.percentile(np.abs(off), 90):.2f} m')
r = np.convolve(np.r_[0, -np.diff(a[:, 2])/np.maximum(np.diff(a[:, 0]), .05)], np.ones(10)/10, 'same')
for d, ci, pi in ((20, 4, 5), (40, 6, 7), (60, 8, 9)):
  m = np.isfinite(a[:, pi])
  coef = np.linalg.lstsq(np.c_[off[m], r[m], np.ones(m.sum())], a[m, pi]-a[m, ci], rcond=None)[0]
  print(f'path-centre at {d} m = {coef[0]:+.2f}*car_offset {coef[1]:+.2f}*drift_rate {coef[2]:+.2f}  (n {m.sum()})')
