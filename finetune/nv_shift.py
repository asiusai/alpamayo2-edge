"""Shifted-view recovery samples from exported PhysicalAI-AV samples (NVIDIA's own six-camera rig and calibration).

Usage (NVIDIA venv, CPU): nv_shift.py SAMPLE_DIR OUT_DIR [--limit N] [--workers N]
For each exported sample (<clip>-<t0_ms>.npz from physicalai_export.py) on a straight at speed, writes variants at 960x540:
  <stem>.c.npz    unshifted, through the same warp path (identity)
  <stem>.yL.npz   car 0.5 m left of the recorded path, parallel to it; .yR 0.5 m right
  <stem>.hL.npz   car heading 1.5 deg left of the recorded path, crossing it at t0 (earlier frames were to the right); .hR
Every image is rendered from the virtual pose at its own capture time: heading as an exact rotation, lateral offset with
a road-plane warp (rays above the horizon and far points keep their direction; the car's own body stays fixed). The ego history is unchanged: the virtual
car drove the same motion relative to itself. `gt_xyz` is the recorded future path in the virtual ego frame, and `shift`
holds (dy m left, dpsi rad left) at t0.
"""
import argparse
import math
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import map_coordinates
from scipy.spatial.transform import Rotation

CAL = Path('/mnt/nvme-500/alpamayo-train/physicalai/calibration')
NAMES = {0: 'camera_cross_left_120fov', 1: 'camera_front_wide_120fov', 2: 'camera_cross_right_120fov',
         3: 'camera_rear_left_70fov', 5: 'camera_rear_right_70fov', 6: 'camera_front_tele_30fov'}
OUT_W, OUT_H, FAR = 960, 540, 200.
BODY_CENTRE, BODY_LENGTH, BODY_WIDTH = 1.327, 4.872, 2.121  # m: bbox centre ahead of the rear axle, length, width
VARIANTS = {'c': (0., 0.), 'yL': (.5, 0.), 'yR': (-.5, 0.), 'hL': (0., math.radians(1.5)), 'hR': (0., -math.radians(1.5))}
INTR = EXTR = None


def calibration():
  global INTR, EXTR
  if INTR is None:
    INTR = pd.read_parquet(CAL/'camera_intrinsics/camera_intrinsics.chunk_0000.parquet')
    EXTR = pd.read_parquet(CAL/'sensor_extrinsics/sensor_extrinsics.chunk_0000.parquet')
  return INTR, EXTR


def warp_coordinates(intr, extr, dy, dpsi):
  """Source pixel coordinates (v, u) at full resolution for each output pixel of the camera at the virtual pose."""
  w, h, cx, cy = intr['width'], intr['height'], intr['cx'], intr['cy']
  bw = [intr[f'bw_poly_{i}'] for i in range(5)]
  fw = [intr[f'fw_poly_{i}'] for i in range(5)]
  rc = Rotation.from_quat([extr['qx'], extr['qy'], extr['qz'], extr['qw']]).as_matrix()
  tc = np.array([extr['x'], extr['y'], extr['z']])
  rz = Rotation.from_euler('z', dpsi).as_matrix()
  cv, rv = rz @ tc+np.array([0., dy, 0.]), rz @ rc
  vv, uu = np.mgrid[:OUT_H, :OUT_W]
  u, v = (uu+.5)*w/OUT_W-.5, (vv+.5)*h/OUT_H-.5
  dx, dyp = u-cx, v-cy
  r = np.hypot(dx, dyp)
  theta = sum(b*r**i for i, b in enumerate(bw))
  s = np.divide(np.sin(theta), r, out=np.zeros_like(r), where=r > 1e-9)
  ray = np.stack([dx*s, dyp*s, np.cos(theta)], -1) @ rv.T  # virtual camera ray in the recorded rig frame
  t = np.full(r.shape, FAR)
  down = ray[..., 2] < -1e-6
  t[down] = np.minimum(-cv[2]/ray[..., 2][down], FAR)
  ground = cv+t[..., None]*ray
  p = (ground-tc) @ rc  # point in the recorded camera frame
  rad = np.hypot(p[..., 0], p[..., 1])
  th = np.arctan2(rad, p[..., 2])
  rr = sum(f*th**i for i, f in enumerate(fw))
  k = np.divide(rr, rad, out=np.zeros_like(rad), where=rad > 1e-9)
  sv, su = cy+k*p[..., 1], cx+k*p[..., 0]
  # Rays that end under the virtual car's own body show the car, which moves with the camera: keep those pixels.
  own = (ground-np.array([0., dy, 0.])) @ rz
  body = down & (np.abs(own[..., 0]-BODY_CENTRE) < BODY_LENGTH/2+.15) & (np.abs(own[..., 1]) < BODY_WIDTH/2+.1)
  return np.where(body, v, sv), np.where(body, u, su)


def render(image, coords):
  return np.stack([map_coordinates(c, coords, order=1, mode='nearest', prefilter=False) for c in image]).astype(np.uint8)


def one(args):
  path, out = args
  stem = path.stem
  clip, t0_ms = stem.rsplit('-', 1)
  if all((out/f'{stem}.{k}.npz').exists() for k in VARIANTS):
    return stem, 'exists'
  intr, extr = calibration()
  with np.load(path) as z:
    z = {k: z[k] for k in z.files}
  hist, gt = z['ego_history_xyz'][0], z['gt_xyz']
  v = float(np.linalg.norm(hist[-1, :2]-hist[0, :2])/1.9)
  d3 = gt[29]-gt[26]
  if v < 5. or abs(math.atan2(d3[1], d3[0])) > .03:
    return stem, 'skip'
  tau = (z['absolute_timestamps'].reshape(-1).astype(np.float64)-int(t0_ms)*1000)/1e6  # s, <= 0
  for key, (dy, dpsi) in VARIANTS.items():
    frames = np.empty((len(tau), 1, 3, OUT_H, OUT_W), np.uint8)
    for i, cam in enumerate(z['camera_indices']):
      ci, ce = intr.loc[(clip, NAMES[int(cam)])], extr.loc[(clip, NAMES[int(cam)])]
      offset = dy+math.tan(dpsi)*v*tau[i]  # lateral position of the virtual car at this frame's time
      frames[i, 0] = render(z['image_frames'][i, 0], warp_coordinates(ci, ce, offset, dpsi))
    rz = Rotation.from_euler('z', dpsi).as_matrix()
    gt_virtual = (gt-np.array([0., dy, 0.])) @ rz  # recorded future path in the virtual ego frame
    np.savez_compressed(out/f'{stem}.{key}.npz', image_frames=frames, camera_indices=z['camera_indices'],
                        absolute_timestamps=z['absolute_timestamps'], ego_history_xyz=z['ego_history_xyz'],
                        ego_history_rot=z['ego_history_rot'], gt_xyz=gt_virtual.astype(np.float32),
                        shift=np.array([dy, dpsi], np.float32), speed=np.float32(v))
  return stem, 'ok'


if __name__ == '__main__':
  ap = argparse.ArgumentParser()
  ap.add_argument('samples', type=Path); ap.add_argument('out', type=Path)
  ap.add_argument('--limit', type=int); ap.add_argument('--workers', type=int, default=8)
  a = ap.parse_args()
  a.out.mkdir(parents=True, exist_ok=True)
  files = sorted(a.samples.glob('*.npz'))[:a.limit]
  with Pool(a.workers) as pool:
    for n, (stem, status) in enumerate(pool.imap_unordered(one, [(f, a.out) for f in files])):
      print(n+1, '/', len(files), stem, status, flush=True)
