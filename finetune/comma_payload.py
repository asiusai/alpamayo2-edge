"""Payload (payload.py format) from one comma5k segment recorded on a comma four (mici, same OS04C10 cameras as the Dragon).

Usage: comma_payload.py OUT SEGMENT_DIR [MAX_MOMENTS]
comma5k ships a stripped rlog (carState, carParams, wide camera states), raw HEVC, `frame_info.safetensors` (frame times and
HEVC index) and `localizer.safetensors` (100 Hz locationd state: ECEF position, ECEF<-device quaternion w,x,y,z, ECEF
velocity; mount calibration `rpy` and `wide_from_device_euler`). This builds the states the car pipeline reads:
- deviceMotion at 20 Hz from the localizer (NED orientation of the device, device-frame velocity), as locationd publishes it;
- extrinsicsCalibration from the localizer calibration;
- vehicleParameters: platform spec steer ratio, no offset (liveParameters is not in comma5k; unused for these samples);
- localizer.npz: the driven path. samples.py takes the target and the ego history's x/y and heading from it, which is what
  the car's vehicle-model history gives with well-learned parameters.
Moments: 1 s grid, at least 4 m/s, cruise disengaged (human driving) from -2.1 s to +6.5 s. Frames follow payload.py.
"""
import json
import math
import sys
from pathlib import Path

import av
import numpy as np
import zstandard
from safetensors.numpy import load_file
from scipy.spatial.transform import Rotation

from openpilot.cereal import log
from openpilot.selfdrive.alpamayo.capture import state_value

OFFSETS = (-2., -1., -.5, -.1, 0.)
FUTURE_NS = 6_500_000_000
VIEWS = {6: 'fcamera', 1: 'ecamera'}


def ecef_to_ned_matrix(ecef):
  x, y, z = ecef
  lon = math.atan2(y, x)
  lat = math.atan2(z, math.hypot(x, y)*(1-6.69437999014e-3))  # geodetic latitude, one fixed-point step is enough here
  sl, cl, so, co = math.sin(lat), math.cos(lat), math.sin(lon), math.cos(lon)
  return np.array([[-sl*co, -sl*so, cl], [-so, co, 0.], [-cl*co, -cl*so, -sl]])  # rows: north, east, down in ECEF


def car_params(fingerprint):
  """comma5k strips carParams to the fingerprint; the vehicle-model fields come from opendbc's platform specs."""
  from opendbc.car.interfaces import scale_rot_inertia_per_mass, scale_tire_stiffness_per_mass
  from opendbc.car.values import PLATFORMS
  if fingerprint not in PLATFORMS:
    raise ValueError(f'unknown platform {fingerprint}')
  specs = PLATFORMS[fingerprint].config.specs
  front, rear = scale_tire_stiffness_per_mass(specs.wheelbase, specs.centerToFrontRatio*specs.wheelbase, specs.tireStiffnessFactor)
  return dict(rotationalInertiaPerMass=float(scale_rot_inertia_per_mass(specs.wheelbase)), wheelbase=float(specs.wheelbase),
              centerToFront=float(specs.centerToFrontRatio*specs.wheelbase), tireStiffnessFrontPerMass=float(front),
              tireStiffnessRearPerMass=float(rear), steerRatio=float(specs.steerRatio))


def main(out, seg, max_moments=60):
  raw = zstandard.ZstdDecompressor().stream_reader((seg/'rlog.zst').open('rb')).read()
  loc, info = load_file(str(seg/'localizer.safetensors')), load_file(str(seg/'frame_info.safetensors'))
  events, cruise, cs_rows, cp = [], [], [], None
  for m in log.Event.read_multiple_bytes(raw):
    kind = m.which()
    if kind == 'carState':
      value, valid = state_value(kind, m.carState)
      events.append(dict(k=kind, stamp=int(m.logMonoTime), received=int(m.logMonoTime), valid=bool(m.valid and valid), value=value))
      cruise.append((m.logMonoTime, m.carState.cruiseState.enabled))
      cs_rows.append((m.logMonoTime, value['vEgo'], value['steeringAngleDeg']))
    elif kind == 'carParams' and cp is None:
      cp = car_params(m.carParams.carFingerprint)
  if cp is None or not cs_rows:
    raise ValueError('no carParams or carState')
  first_ns = min(e['stamp'] for e in events)
  events.append(dict(k='carParams', stamp=first_ns, received=first_ns, valid=True, value=cp))

  # Localizer -> deviceMotion (20 Hz) and the path.
  t_ns = (loc['t']*1e9).astype(np.int64)
  s = loc['states']
  pos, quat, vel = s[:, 0:3], s[:, 3:7], s[:, 7:10]
  ned = ecef_to_ned_matrix(pos[len(pos)//2])
  r_ecef_dev = Rotation.from_quat(quat[:, [1, 2, 3, 0]]).as_matrix()
  r_ned_dev = np.einsum('ij,njk->nik', ned, r_ecef_dev)
  euler = Rotation.from_matrix(r_ned_dev).as_euler('xyz')
  v_dev = np.einsum('nji,nj->ni', r_ecef_dev, vel)
  for i in range(0, len(t_ns), 5):
    events.append(dict(k='deviceMotion', stamp=int(t_ns[i]), received=int(t_ns[i])+5_000_000, valid=True,
                       value=[euler[i].tolist(), v_dev[i].tolist()]))
  p_ned, v_ned = (pos-pos[0]) @ ned.T, vel @ ned.T

  # liveParameters is not in comma5k. The history and target come from the localizer (samples.py), so these spec values only
  # satisfy the state lookups.
  steer_ratio, offset = cp['steerRatio'], 0.
  cs = np.array(cs_rows, dtype=np.float64)
  vp = dict(stiffnessFactor=1., steerRatio=steer_ratio, angleOffsetDeg=offset, roll=0.)
  for stamp in range(int(first_ns), int(cs[-1, 0])+1, 50_000_000):
    events.append(dict(k='vehicleParameters', stamp=stamp, received=stamp, valid=True, value=vp))
  cal = dict(rpyCalib=[float(x) for x in loc['rpy']], wideFromDeviceEuler=[float(x) for x in loc['wide_from_device_euler']])
  for stamp in range(int(first_ns), int(cs[-1, 0])+1, 250_000_000):  # the car requires calibration at most 1 s old
    events.append(dict(k='extrinsicsCalibration', stamp=stamp, received=stamp, valid=True, value=cal))

  # Moments: human driving (cruise off) over the history and target horizon.
  sofs = {v: (info[f'{name}/t']*1e9).astype(np.int64) for v, name in VIEWS.items()}
  first = max(int(sofs[v][0]) for v in VIEWS)
  last = min(int(t_ns[-1]), int(cs[-1, 0]), int(sofs[6][-1]))-FUTURE_NS
  ct, ce = np.array([c[0] for c in cruise]), np.array([c[1] for c in cruise], float)
  moments = []
  for t in np.arange(math.ceil((first+2.6e9)/1e9)*1e9, last, 1e9).astype(np.int64):
    if np.interp(t, cs[:, 0], cs[:, 1]) < 4:
      continue
    w = (ct >= t-2.1e9) & (ct <= t+6.5e9)
    if not w.any() or ce[w].max() > 0:
      continue
    moments.append(int(t))
  moments = moments[:max_moments]
  need, queries = {v: set() for v in VIEWS}, []
  for t in moments:
    qi = int(np.searchsorted(sofs[6], t, side='right'))-1
    if qi < 0:
      continue
    q = int(sofs[6][qi]); queries.append(q)
    for v in VIEWS:
      for o in OFFSETS:
        i = int(np.searchsorted(sofs[v], min(q+int(o*1e9)+10_000_000, q), side='right'))-1
        if i >= 0:
          need[v].add(i)
  out.mkdir(parents=True, exist_ok=False)
  np.savez(out/'localizer.npz', t_ns=t_ns, north=p_ned[:, 0], east=p_ned[:, 1], vn=v_ned[:, 0], ve=v_ned[:, 1])
  counts = {v: 0 for v in VIEWS}
  for v, name in VIEWS.items():
    with (out/f'drive-{v}.nv12').open('wb') as bank, av.open(str(seg/f'{name}.hevc'), format='hevc') as container:
      for position, frame in enumerate(container.decode(video=0)):
        if position not in need[v] or position >= len(sofs[v]):
          continue
        sof = int(sofs[v][position])
        bank.write(frame.reformat(format='nv12').to_ndarray().reshape(-1).tobytes())
        events.append(dict(k='frame', v=v, id=position, index=position, sof=sof, eof=sof+20_000_000,
                           received=sof+50_000_000, offset=counts[v]*1344*1140))
        counts[v] += 1
  if any(counts[v] < len(need[v]) for v in VIEWS):
    raise ValueError(f'decoded {counts}, needed {[len(need[v]) for v in VIEWS]}')
  events.sort(key=lambda e: e['received'])
  (out/'drive-events.json').write_text(json.dumps(events, allow_nan=False))
  (out/'fixture.json').write_text(json.dumps(dict(source=[str(seg)], moments=queries, localizer='localizer.npz',
      vehicle_parameters=vp, media={'drive': {str(v): {'file': f'drive-{v}.nv12', 'frames': counts[v]} for v in VIEWS}}), indent=1)+'\n')
  print(seg.name, 'moments', len(queries), 'frames', counts, flush=True)


if __name__ == '__main__':
  main(Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3]) if len(sys.argv) > 3 else 60)
