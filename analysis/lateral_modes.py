"""Replay logged Alpamayo plans through the live PlanSelector and compare steering conversions with the human.

Usage (openpilot PYTHONPATH): lateral_modes.py SEGMENT_DIR [SEGMENT_DIR ...]
Modes, both at gain 1:
  tracker: the current pure pursuit on the plan path (1.5 s lookahead, odometry-fixed path, 0.25 s smoothing).
  plan:    openpilot's get_curvature_from_plan on the plan's heading at the steering delay (the tracker's fallback).
Reference: the curvature the human drove LAT_DELAY later (odometry heading rate / speed), only while lateral control is
off and the driver steers. The tracker replay is checked against the logged modelV2 command while engaged.
"""
import math
import sys
from pathlib import Path

import types

import numpy as np
import zstandard

try:
  import setproctitle  # noqa: F401
except ImportError:  # offline replay env: drive_helpers imports common.realtime only for constants
  sys.modules['setproctitle'] = types.SimpleNamespace(getproctitle=lambda: '', setproctitle=lambda _: None)

from openpilot.cereal import log
from openpilot.selfdrive.alpamayo.planner import PlanSelector

LAT_DELAY = .35  # s, steering delay used for both the command and the human reference (Tesla lateralDelay + smoothing)
LONG_DELAY = .3


class PlanOnly(PlanSelector):
  def tracked_curvature(self, now, v_ego, lat_delay, gain):
    return None


def events(seg):
  raw = zstandard.ZstdDecompressor().stream_reader((Path(seg)/'rlog.zst').open('rb')).read()
  return sorted(log.Event.read_multiple_bytes(raw), key=lambda m: m.logMonoTime)


def segment(seg):
  rows, check = [], []
  for seg in [seg]:
    try:
      evs = events(seg)
    except Exception as e:  # noqa: BLE001
      print(seg, 'unreadable', e)
      continue
    modes = {'tracker': PlanSelector(None, 1.), 'plan': PlanOnly(None, 1.)}
    v, lat, pressed = 0., False, False
    heading = []  # (ns, heading) for the human reference
    cmds = []     # (t, v, lat, tracker, plan, logged)
    for m in evs:
      w, t = m.which(), m.logMonoTime
      if w == 'carState':
        v, pressed = m.carState.vEgo, m.carState.steeringPressed
      elif w == 'carControl':
        lat = m.carControl.latActive
      elif w == 'deviceMotion':
        stamp = m.deviceMotion.timestamp if m.deviceMotion.timestamp > 0 else t
        h = -m.deviceMotion.orientationNED.z
        heading.append((stamp, h))
        for p in modes.values():
          p.motion(stamp, v, h)
      elif w == 'alpamayoPlan' and m.valid:
        for p in modes.values():
          p.receive(m.alpamayoPlan, t+20_000_000, t)
      elif w == 'modelV2':
        out = {}
        for k, p in modes.items():
          a = p.candidate_action(t, v, LAT_DELAY, LONG_DELAY)
          out[k] = a['desiredCurvature'] if a else np.nan
        cmds.append((t, v, lat, out['tracker'], out['plan'], m.modelV2.action.desiredCurvature))
    if len(heading) < 10 or not cmds:
      continue
    hs, hv = np.array([h[0] for h in heading]), np.unwrap(np.array([h[1] for h in heading]))
    for t, vv, la, ktr, kpl, klog in cmds:
      ref_t = t+LAT_DELAY*1e9
      if vv < 3 or ref_t+.15e9 > hs[-1]:
        continue
      rate = (np.interp(ref_t+.15e9, hs, hv)-np.interp(ref_t-.15e9, hs, hv))/.3
      human = -rate/vv  # openpilot curvature is right positive; heading is left positive
      if la:
        check.append((ktr, klog))
      else:
        rows.append((human, ktr, kpl, vv))
  return rows, check


def main(segs):
  from multiprocessing import Pool
  rows, check = [], []
  with Pool(12) as pool:
    for r, c in pool.imap_unordered(segment, segs):
      rows += r
      check += c
  r = np.array(rows, float)
  ok = np.isfinite(r).all(1)
  r = r[ok]
  print(f'human-driven samples {len(r)} (of {len(rows)} with both commands)')
  for name, col in (('tracker', 1), ('plan', 2)):
    turn = np.abs(r[:, 0]) > .01
    straight = np.abs(r[:, 0]) < .002
    slope = np.sum(r[turn, 0]*r[turn, col])/np.sum(r[turn, 0]**2)
    rmse = math.sqrt(np.mean((r[:, col]-r[:, 0])**2))
    lat_acc_err = np.median(np.abs(r[:, col]-r[:, 0])*r[:, 3]**2)
    print(f'{name:8s} turn ratio {slope:.2f} (n {turn.sum()})  rmse {rmse:.4f} 1/m  median |lat accel err| {lat_acc_err:.2f} m/s2  '
          f'straight mean {1e3*np.mean(r[straight, col]):+.2f} mrad/m (n {straight.sum()})')
  if check:
    c = np.array(check, float)
    c = c[np.isfinite(c).all(1)]
    if len(c):
      print(f'replay check while engaged: tracker vs logged command median |diff| {np.median(np.abs(c[:, 0]-c[:, 1])):.5f} 1/m, '
            f'corr {np.corrcoef(c[:, 0], c[:, 1])[0, 1]:.3f}, n {len(c)}')


if __name__ == '__main__':
  main(sys.argv[1:])
