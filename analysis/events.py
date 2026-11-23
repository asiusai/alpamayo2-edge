# Disengagements and steering overrides on a route, with the context before each one.
import sys, glob, numpy as np
from openpilot.tools.lib.logreader import LogReader
route = sys.argv[1]
segs = sorted(glob.glob(f'/data/media/0/realdata/{route}--*'), key=lambda p: int(p.rsplit('--', 1)[1]))
T, events = [], []; cs = ss = dm = mv = None; t0 = None; ctl = np.nan
for s in segs:
  f = glob.glob(s+'/rlog*')
  if not f: continue
  for m in LogReader(f[0]):
    w = m.which(); t = m.logMonoTime
    if t0 is None: t0 = t
    if w == 'carState': cs = m.carState
    elif w == 'selfdriveState': ss = m.selfdriveState
    elif w == 'deviceMotion': dm = m.deviceMotion
    elif w == 'modelV2': mv = (m.modelV2.alpamayoQueryMonoTime, t)
    elif w == 'onroadEvents':
      names = [str(e.name) for e in m.onroadEvents]
      if names: events.append(((t-t0)/1e9, names))
    elif w == 'carControl' and cs is not None and ss is not None and dm is not None:
      cc = m.carControl; v = max(cs.vEgo, .1)
      age = (t-mv[0])/1e6 if mv and mv[0] else np.nan
      T.append(((t-t0)/1e9, cs.vEgo, ss.enabled, cc.latActive, cs.steeringPressed, cs.steeringTorque, cs.gasPressed, cs.brakePressed,
                cc.actuators.curvature, -dm.angularVelocityDevice.z/v, cs.steeringAngleDeg, age, cs.leftBlinker, cs.rightBlinker))
a = np.array(T, dtype=float)
t, v, en, lat, sp = a[:,0], a[:,1], a[:,2] > 0, a[:,3] > 0, a[:,4] > 0
print(f'duration {t[-1]:.0f} s, enabled {en.mean()*t[-1]:.0f} s, latActive {lat.mean()*t[-1]:.0f} s')
for i in np.nonzero(en[:-1] & ~en[1:])[0]:
  w = (t > t[i]-3) & (t <= t[i])
  cause = sorted({n for (te, names) in events if t[i]-.5 <= te <= t[i]+.5 for n in names} - {'noGps'})
  print(f'DISENGAGE t={t[i]:6.1f} v={v[i]:5.1f} gas={a[i,6]:.0f} brake={a[i+1,7]:.0f} press={sp[i]:d} | 3s: cmd {a[w,8].mean():+.4f} act {a[w,9].mean():+.4f} '
        f'angle {a[w,10].mean():+.1f}->{a[i,10]:+.1f} torque {a[w,5][np.argmax(np.abs(a[w,5]))]:+.2f} age {np.nanmedian(a[w,11]):.0f} blink {int(a[w,12].max())}{int(a[w,13].max())} | {cause[:4]}')
press = lat & sp
edges = np.diff(np.r_[0, press.astype(int), 0]); starts, ends = np.nonzero(edges == 1)[0], np.nonzero(edges == -1)[0]
for s0, e0 in zip(starts, ends):
  if t[e0-1]-t[s0] < .3: continue
  w = (t > t[s0]-2) & (t <= t[s0])
  print(f'OVERRIDE t={t[s0]:6.1f} dur {t[e0-1]-t[s0]:4.1f}s v={v[s0]:5.1f} torque {a[s0:e0,5].mean():+.2f} | before: cmd {a[w,8].mean():+.4f} act {a[w,9].mean():+.4f} '
        f'angle {a[w,10].mean():+.1f} age {np.nanmedian(a[w,11]):.0f} blink {int(a[w,12].max())}{int(a[w,13].max())}')
