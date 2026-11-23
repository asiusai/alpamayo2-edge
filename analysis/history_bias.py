# Ego-history curvature source check: steering-angle curvature (as selfdrive/alpamayo/history.py builds it) vs the gyro.
# Usage (opendbc-coop venv, PYTHONPATH=op-coop): history_bias.py ROUTE_PREFIX [...]   (segment dirs are ROUTE_PREFIX--N)
# Left positive curvature in 1/m. Straights: |gyro curvature| < 5e-4 smoothed over 2 s and speed > 12 m/s.
import math, sys, types
from pathlib import Path
from types import SimpleNamespace
import numpy as np, zstandard
sys.modules.setdefault('setproctitle', types.SimpleNamespace(getproctitle=lambda: '', setproctitle=lambda _: None))
from openpilot.cereal import log
from opendbc.car.vehicle_model import VehicleModel


def run(prefix):
  segs = sorted(Path(prefix).parent.glob(Path(prefix).name+'--*'), key=lambda s: int(s.name.rsplit('--', 1)[1]))
  vm, lp, gyro, rows = None, None, None, []
  for seg in segs:
    raw = zstandard.ZstdDecompressor().stream_reader((seg/'rlog.zst').open('rb')).read()
    it = log.Event.read_multiple_bytes(raw)
    while True:
      try:
        m = next(it)
      except StopIteration:
        break
      except Exception:
        break
      w = m.which()
      if w == 'carParams' and vm is None:
        vm = VehicleModel(m.carParams)
      elif w == 'vehicleParameters':
        lp = m.vehicleParameters
      elif w == 'deviceMotion':
        gyro = (m.logMonoTime, m.deviceMotion.angularVelocityDevice.z, m.deviceMotion.angularVelocityDevice.valid)
      elif w == 'liveCalibration':
        pass
      elif w == 'carState' and vm is not None and lp is not None and gyro is not None:
        cs = m.carState
        if m.logMonoTime-gyro[0] > 1e8 or cs.vEgo < 5:
          continue
        vm.update_params(max(lp.stiffnessFactor, .1), max(lp.steerRatio, .1))
        k = vm.calc_curvature(math.radians(cs.steeringAngleDeg-lp.angleOffsetDeg), cs.vEgo, lp.roll)
        k0 = vm.calc_curvature(math.radians(cs.steeringAngleDeg-lp.angleOffsetDeg), cs.vEgo, 0.)
        rows.append((m.logMonoTime*1e-9, cs.vEgo, k, k0, -gyro[1]/cs.vEgo, lp.roll, lp.angleOffsetDeg, lp.angleOffsetAverageDeg, lp.steerRatio,
                     cs.steeringPressed, cs.cruiseState.enabled))
  a = np.array(rows)
  n = 200  # 2 s at 100 Hz
  ker = np.ones(n)/n
  sm = lambda x: np.convolve(x, ker, 'same')
  kg, ks, k0 = sm(a[:, 4]), sm(a[:, 2]), sm(a[:, 3])
  st = (np.abs(kg) < 5e-4) & (a[:, 1] > 12)
  d = (ks-kg)[st][::n]
  print(f'{Path(prefix).name}: {len(a)/100:.0f} s moving, straights {st.sum()/100:.0f} s; SR {a[:, 8].mean():.2f} offset {a[:, 6].mean():+.2f} deg avg {a[:, 7].mean():+.2f} roll {a[:, 5].mean():+.4f} rad')
  print(f'  steer-gyro curvature on straights: mean {d.mean():+.2e} median {np.median(d):+.2e} p10 {np.percentile(d, 10):+.2e} p90 {np.percentile(d, 90):+.2e} (2 s windows {len(d)})')
  print(f'  steer(no roll)-gyro: mean {(k0-kg)[st].mean():+.2e}; gyro mean {kg[st].mean():+.2e}; steer mean {ks[st].mean():+.2e}')
  for name, sel in (('engaged', st & (a[:, 10] > 0)), ('manual', st & (a[:, 10] == 0))):
    if sel.sum() > 500:
      print(f'  {name}: {sel.sum()/100:.0f} s  steer-gyro {(ks-kg)[sel].mean():+.2e}  steer {ks[sel].mean():+.2e}  gyro {kg[sel].mean():+.2e}')
  return a


for p in sys.argv[1:]:
  run(p)
