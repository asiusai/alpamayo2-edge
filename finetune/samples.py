"""Training samples from a payload (payload.py): live-identical model inputs, optionally seen from a shifted and turned pose,
with a target trajectory that returns to the path the human drove.

Usage: samples.py PAYLOAD OUT_DIR [--recovery K] [--seed N] [--fov WIDE,NARROW]
Per query moment: one clean sample (the human's own path as target) and K recovery samples with random dy/dpsi.
"""
import argparse
import json
import math
import sys
from bisect import bisect_right
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parent))
import render  # noqa: E402
from openpilot.selfdrive.alpamayo.frontend import (FRAME_SLACK_NS, OFFSETS_NS, STATE_LIMITS, State, Unavailable,  # noqa: E402
                                                    navigation, required)
from openpilot.selfdrive.alpamayo.history import ego_history  # noqa: E402

FRAME_BYTES = 1344*1140
GEOMETRY = json.load(open('/home/john/alpamayo-chestnut-pc/live-karel.json'))['geometry']
CACHE = '/home/john/alpamayo-work/camera-views/cache'
KINDS = ('carState', 'carParams', 'vehicleParameters', 'extrinsicsCalibration', 'deviceMotion', 'navInstruction')
RECOVERY_S = 3.0  # s, the target returns to the human path over this time
STEP_NS, HORIZON = 50_000_000, 128  # future integration grid; the model's 0.1 s grid is every second step


def load(payload):
  events = json.loads((payload/'drive-events.json').read_bytes())
  fixture = json.loads((payload/'fixture.json').read_bytes())
  banks = {int(k): np.memmap(payload/m['file'], dtype=np.uint8, mode='r') for k, m in fixture['media']['drive'].items()}
  rows = {k: {} for k in KINDS}
  for e in events:
    if e['k'] in rows:
      old = rows[e['k']].get(e['stamp'])
      if old is None or e['received'] > old.received:
        rows[e['k']][e['stamp']] = State(e['stamp'], e['received'], e['value'] if e['valid'] else None, e['valid'])
  states = {k: tuple(v[s] for s in sorted(v)) for k, v in rows.items()}
  if states['carParams']:
    # carParams is logged once per segment; in the car it carries over from earlier segments. Constant per drive.
    first = min(e['received'] for e in events)
    cp = states['carParams'][0]
    states['carParams'] = (State(first, first, cp.value, cp.valid),)
  frames = {1: [], 6: []}
  for e in events:
    if e['k'] == 'frame':
      frames[e['v']].append(e)
  for v in frames:
    frames[v].sort(key=lambda e: e['sof'])
  if fixture.get('localizer'):
    with np.load(payload/fixture['localizer']) as z:
      fixture['_path'] = {k: z[k] for k in z.files}
  return fixture, banks, states, frames


def visible(states, now):
  return {k: tuple(r for r in rows if r.received <= now) for k, rows in states.items()}


def human_future(states, query):
  """The driven path after the query from speed and steering (the live history's vehicle model), in the query frame."""
  from opendbc.car.vehicle_model import VehicleModel
  cp, _ = required(states['carParams'], query, STATE_LIMITS['carParams'])
  vm = VehicleModel(SimpleNamespace(**cp))
  speed, rate = [], []
  for k in range(HORIZON+1):
    stamp = query+k*STEP_NS
    cs, _ = required(states['carState'], stamp, STATE_LIMITS['carState'])
    lp, _ = required(states['vehicleParameters'], stamp, STATE_LIMITS['vehicleParameters'])
    vm.update_params(max(lp['stiffnessFactor'], .1), max(lp['steerRatio'], .1))
    v = max(cs['vEgo'], 0.)
    speed.append(v)
    rate.append(v*vm.calc_curvature(math.radians(cs['steeringAngleDeg']-lp['angleOffsetDeg']), v, lp['roll']))
  speed, rate = np.array(speed), np.array(rate)
  dt = STEP_NS/1e9
  yaw = np.r_[0., np.cumsum((rate[:-1]+rate[1:])*dt/2)]
  vel = speed[:, None]*np.stack([np.cos(yaw), np.sin(yaw)], -1)
  xy = np.vstack([np.zeros(2), np.cumsum((vel[:-1]+vel[1:])*dt/2, axis=0)])
  return xy, yaw, speed  # 0..6.4 s every 50 ms


def teacher_future(plan):
  """comma's logged plan (modelV2.position for the query frame: t, x forward, y left) on the 50 ms target grid."""
  T, X, Y = (np.asarray(v, dtype=np.float64) for v in plan)
  t = np.arange(HORIZON+1)*STEP_NS/1e9
  x, y = np.interp(t, T, X), np.interp(t, T, Y)
  dx, dy = np.gradient(x, t), np.gradient(y, t)
  speed = np.hypot(dx, dy)
  yaw = np.unwrap(np.arctan2(dy, dx))
  for i in range(1, len(yaw)):  # hold the heading through stops
    if speed[i] < 1.:
      yaw[i] = yaw[i-1]
  return np.stack([x-x[0], y-y[0]], -1), yaw, speed


def localizer_future(path, query):
  """The driven path after the query from a localizer track (comma5k), in the query frame: x along the direction of travel
  at the query, y left. Same 50 ms grid and outputs as human_future."""
  t = query+np.arange(HORIZON+1)*STEP_NS
  if t[-1] > path['t_ns'][-1] or t[0] < path['t_ns'][0]:
    raise Unavailable('localizer track too short')
  north, east = (np.interp(t, path['t_ns'], path[k]) for k in ('north', 'east'))
  vn, ve = (np.interp(t, path['t_ns'], path[k]) for k in ('vn', 've'))
  speed = np.hypot(vn, ve)
  if speed[0] < 1.:
    raise Unavailable('too slow for a heading')
  course = np.unwrap(np.arctan2(-ve, vn))  # left positive (NED: east is right of north)
  h0 = course[0]
  dn, de = north-north[0], east-east[0]
  x = np.cos(h0)*dn-np.sin(h0)*de
  y = -(np.sin(h0)*dn+np.cos(h0)*de)
  # The course is only defined while moving; hold it below 1 m/s (stops) instead of following noise.
  yaw = course-h0
  for i in range(1, len(yaw)):
    if speed[i] < 1.:
      yaw[i] = yaw[i-1]
  return np.stack([x, y], -1), yaw, speed


def localizer_history(path, query):
  """Ego history x/y (0.1 s, -1.9..0 s) and heading from a localizer track, in the query frame of localizer_future."""
  t = query+np.arange(-19, 1)*100_000_000
  if t[0] < path['t_ns'][0]:
    raise Unavailable('localizer track starts too late')
  north, east, vn, ve = (np.interp(t, path['t_ns'], path[k]) for k in ('north', 'east', 'vn', 've'))
  speed = np.hypot(vn, ve)
  course = np.unwrap(np.arctan2(-ve, vn))
  for i in range(len(course)-2, -1, -1):  # hold the course through stops, from the query backwards
    if speed[i] < 1.:
      course[i] = course[i+1]
  h0 = course[-1]
  dn, de = north-north[-1], east-east[-1]
  return np.stack([np.cos(h0)*dn-np.sin(h0)*de, -(np.sin(h0)*dn+np.cos(h0)*de)], -1), course-h0


def recovery_target(xy, yaw, speed, dy, dpsi):
  """The human path, offset by a lateral error that decays from (dy, its rate) to zero over RECOVERY_S; in the shifted frame."""
  t = np.arange(len(xy))*STEP_NS/1e9
  v0 = max(speed[0], 1.)
  rate0 = v0*math.sin(dpsi)
  tau = np.clip(t/RECOVERY_S, 0., 1.)
  e = (2*tau**3-3*tau**2+1)*dy+(tau**3-2*tau**2+tau)*RECOVERY_S*rate0
  de = np.gradient(e, t)
  normal = np.stack([-np.sin(yaw), np.cos(yaw)], -1)
  pts = xy+normal*e[:, None]
  heading = yaw+np.arctan2(de, np.maximum(speed, 1.))
  # express in the shifted start frame: origin at (0, dy), heading dpsi
  c, s = math.cos(dpsi), math.sin(dpsi)
  rel = pts-np.array([0., dy])
  fx, fy = c*rel[:, 0]+s*rel[:, 1], -s*rel[:, 0]+c*rel[:, 1]
  return np.stack([fx, fy], -1), heading-dpsi


def shifted_history(xyz, rot, dy, dpsi, speed):
  """Past poses of a car that drifted at heading error dpsi into lateral offset dy (left), in its own current frame."""
  if not dy and not dpsi:
    return xyz, rot
  t = np.arange(-19, 1)*.1
  yaw = Rotation.from_matrix(rot[0]).as_euler('xyz')[:, 2]
  off = dy+speed*math.sin(dpsi)*t
  normal = np.stack([-np.sin(yaw), np.cos(yaw)], -1)
  pts = xyz[0, :, :2]+normal*off[:, None]
  c, s = math.cos(dpsi), math.sin(dpsi)
  rel = pts-pts[-1]
  out = xyz.copy()
  out[0, :, 0], out[0, :, 1] = c*rel[:, 0]+s*rel[:, 1], -s*rel[:, 0]+c*rel[:, 1]
  return out.astype(np.float32), rot  # constant heading error: relative yaws are unchanged


def build(payload, out, recovery, rng, geometry=GEOMETRY):
  fixture, banks, states, frames = load(payload)
  route = Path(fixture['source'][0]).name
  sofs = {v: [f['sof'] for f in frames[v]] for v in frames}
  written = 0
  for query in fixture['moments']:
    try:
      now = query+60_000_000
      vis = visible(states, now)
      chosen = []
      for v in (1, 6):
        for o in OFFSETS_NS:
          i = bisect_right(sofs[v], min(query+o+FRAME_SLACK_NS, query))-1
          if i < 0:
            raise Unavailable('missing frame')
          chosen.append(frames[v][i])
      cal, _ = required(vis['extrinsicsCalibration'], query, STATE_LIMITS['extrinsicsCalibration'])
      xyz, rot, hist = ego_history(vis, query, cal['rpyCalib'])
      if '_path' in fixture:  # comma5k: the driven path replaces the bicycle-model history (no learned parameters logged)
        hxy, hyaw = localizer_history(fixture['_path'], query)
        xyz[0, :, :2] = hxy
        tilt = Rotation.from_matrix(rot[0]).as_euler('xyz')
        tilt[:, 2] = hyaw
        rot[0] = Rotation.from_euler('xyz', tilt).as_matrix()
      if 'teacher' in fixture:
        xy, yaw, speed = teacher_future(fixture['teacher'][str(query)])
      elif '_path' in fixture:
        xy, yaw, speed = localizer_future(fixture['_path'], query)
      else:
        xy, yaw, speed = human_future(states, query)
      nav, _ = navigation(vis['navInstruction'], query, 'Follow the road.')
    except (Unavailable, KeyError, ValueError) as error:
      print(route, query, 'skipped:', error, flush=True)
      continue
    pixels = [np.asarray(banks[f['v']][f['offset']:f['offset']+FRAME_BYTES]).reshape(1140, 1344) for f in chosen]
    cals = [required(vis['extrinsicsCalibration'], f['sof'], STATE_LIMITS['extrinsicsCalibration'])[0] for f in chosen]
    perturbations = [(0., 0.)]
    for _ in range(recovery):
      kind = rng.integers(3)
      dy = rng.uniform(.3, 1.) * rng.choice((-1., 1.)) if kind != 1 else 0.
      dpsi = rng.uniform(.01, .045) * rng.choice((-1., 1.)) if kind != 0 else 0.
      perturbations.append((float(dy), float(dpsi)))
    for k, (dy, dpsi) in enumerate(perturbations):
      images = np.stack([render.render(px, f['v'], geometry, c, dy+speed[0]*math.sin(dpsi)*(OFFSETS_NS[j % 5]/1e9), dpsi, cache_dir=CACHE)
                         for j, (px, f, c) in enumerate(zip(pixels, chosen, cals, strict=True))])[:, None]
      hxyz, hrot = shifted_history(xyz, rot, dy, dpsi, speed[0])
      txy, tyaw = recovery_target(xy, yaw, speed, dy, dpsi)
      target_xyz = np.zeros((64, 3), np.float32); target_xyz[:, :2] = txy[2::2]
      target_rot = Rotation.from_euler('z', tyaw[2::2][:, None]).as_matrix().astype(np.float32)
      name = f'{route}-{query}-{k}'
      np.savez_compressed(out/f'{name}.npz', image_frames=images, camera_indices=np.array([1]*5+[6]*5, np.int64),
                          absolute_timestamps=np.array([f['sof']//1000 for f in chosen], np.int64), ego_history_xyz=hxyz.astype(np.float32),
                          ego_history_rot=hrot.astype(np.float32), nav_text=np.array(nav), target_xyz=target_xyz, target_rot=target_rot)
      (out/f'{name}.json').write_text(json.dumps(dict(route=route, query_ns=query, dy_m=dy, dpsi_rad=dpsi, kind='clean' if k == 0 else 'recovery', fov=[geometry['cameras'][str(c)].get('output_fov', 120 if c == 1 else 30) for c in (1, 6)],
                                                      speed_m_s=float(speed[0]), nav_text=nav)))
      written += 1
  print(route, 'samples', written, flush=True)


if __name__ == '__main__':
  ap = argparse.ArgumentParser()
  ap.add_argument('payload', type=Path); ap.add_argument('out', type=Path)
  ap.add_argument('--recovery', type=int, default=2); ap.add_argument('--seed', type=int, default=0)
  ap.add_argument('--fov', default=None, help='wide,narrow output field of view in degrees (default: the 120,30 model rig)')
  a = ap.parse_args()
  a.out.mkdir(parents=True, exist_ok=True)
  geometry = json.loads(json.dumps(GEOMETRY))
  if a.fov:
    for c, fov in zip(('1', '6'), a.fov.split(','), strict=True):
      geometry['cameras'][c]['output_fov'] = float(fov)
  build(a.payload, a.out, a.recovery, np.random.default_rng(a.seed), geometry)
