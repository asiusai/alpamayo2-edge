"""Stage A of the six-to-two camera distillation set: decode PhysicalAI-AV moments and render their variants.

Usage (NVIDIA venv, CPU): nv_distill_decode.py LOCAL_DIR SPOOL_DIR CHUNK [CHUNK ...] [--workers N] [--t0 3,5.5,8,10.5,13]
Per clip and t0 (NVIDIA's own loader: six cameras, 22 images, 20 history poses) writes spool files for stage B:
  <clip>-<t0_ms>.c.npz     the recorded moment
  <clip>-<t0_ms>.r.npz     one recovery variant when the car is moving and the past 2 s are nearly straight: the car is
                           dy (0.3-1.0 m) beside and/or dpsi (0.6-2.6 deg) across the recorded path (nv_shift.py warp)
Each file holds the six-camera frames at 960x540 for the teacher, the two front cameras at 768x432 for the student (the
car's prepared size), timestamps, ego history, the recorded future in the (virtual) ego frame, shift and speed.
Stage B (nv_distill_teacher.py) deletes spool files; this stage waits while the spool holds MAX_SPOOL files.
"""
import argparse
import contextlib
import math
import random
import time
import zlib
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy.spatial.transform import Rotation

import nv_shift
from alpamayo2_edge.load_physical_aiavdataset import load_physical_aiavdataset
import physical_ai_av

MAX_SPOOL = 60
STUDENT_W, STUDENT_H = 768, 432
A = None
AVDI = None
CAL = {}


def interface(local_dir):
  global AVDI
  if AVDI is None:
    AVDI = physical_ai_av.PhysicalAIAVDatasetInterface(local_dir=str(local_dir), confirm_download_threshold_gb=1.)
    original = AVDI.open_file

    @contextlib.contextmanager
    def open_local(filename, mode='rb', maybe_stream=False):  # 0.2.0 looks only in the HF cache
      path = Path(local_dir)/filename
      if path.is_file():
        with open(path, mode) as f:
          yield f
      else:
        with original(filename, mode, maybe_stream=maybe_stream) as f:
          yield f
    AVDI.open_file = open_local
  return AVDI


def calibration(local_dir, chunk):
  if chunk not in CAL:
    base = Path(local_dir)/'calibration'
    CAL[chunk] = (pd.read_parquet(base/f'camera_intrinsics/camera_intrinsics.chunk_{chunk:04d}.parquet'),
                  pd.read_parquet(base/f'sensor_extrinsics/sensor_extrinsics.chunk_{chunk:04d}.parquet'))
  return CAL[chunk]


def resize(frame, w, h):
  return np.asarray(Image.fromarray(frame.transpose(1, 2, 0)).resize((w, h), Image.BICUBIC)).transpose(2, 0, 1)


def variant(z, intr, extr, clip, tau, v, dy, dpsi):
  """Six-camera 960x540 frames and front 768x432 frames from the virtual pose (dy, dpsi at t0)."""
  cams = z['camera_indices']
  teacher = np.empty((len(cams), 1, 3, nv_shift.OUT_H, nv_shift.OUT_W), np.uint8)
  student = []
  for i, cam in enumerate(cams):
    src = z['image_frames'][i, 0]
    if dy == 0. and dpsi == 0.:
      teacher[i, 0] = resize(src, nv_shift.OUT_W, nv_shift.OUT_H)
    else:
      ci, ce = intr.loc[(clip, nv_shift.NAMES[int(cam)])], extr.loc[(clip, nv_shift.NAMES[int(cam)])]
      teacher[i, 0] = nv_shift.render(src, nv_shift.warp_coordinates(ci, ce, dy+math.tan(dpsi)*v*tau[i], dpsi))
    if int(cam) in (1, 6):
      student.append(resize(teacher[i, 0], STUDENT_W, STUDENT_H))
  return teacher, np.stack(student)[:, None]


def one(job):
  local_dir, spool, chunk, clip, t0 = job
  stem = f'{clip}-{int(t0*1000)}'
  done = spool.parent/'done'/f'{stem}.done'
  if done.exists():
    return stem, 'exists'
  while len(list(spool.glob('*.npz'))) >= MAX_SPOOL:
    time.sleep(5)
  avdi = interface(local_dir)
  try:
    t0_us = int(t0*1e6)
    data = load_physical_aiavdataset(clip, t0_us, nav_text='Continue straight.', avdi=avdi, maybe_stream=False)
    ego = avdi.get_clip_feature(clip, avdi.features.LABELS.EGOMOTION, maybe_stream=False)
    future = ego(t0_us+np.arange(1, 65, dtype=np.int64)*100_000)
    now = ego(np.array([t0_us], dtype=np.int64))
    gt = now.pose.rotation[0].inv().apply(np.asarray(future.pose.translation)-np.asarray(now.pose.translation)[0])
    gt_rot = (now.pose.rotation[0].inv()*future.pose.rotation).as_matrix()
  except Exception as e:  # noqa: BLE001 - clips without coverage at t0
    return stem, f'skip {type(e).__name__} {str(e)[:60]}'
  z = {'image_frames': data['image_frames'].numpy(), 'camera_indices': data['camera_indices'].numpy(),
       'absolute_timestamps': data['absolute_timestamps'].numpy().reshape(-1)}
  hist_xyz, hist_rot = data['ego_history_xyz'].numpy(), data['ego_history_rot'].numpy()
  hist = hist_xyz[0]
  v = float(np.linalg.norm(hist[-1, :2]-hist[0, :2])/1.9)
  past_yaw = abs(math.atan2(hist_rot[0, 0, 1, 0], hist_rot[0, 0, 0, 0]))  # heading 1.9 s ago in the t0 frame
  tau = (z['absolute_timestamps'].astype(np.float64)-t0_us)/1e6
  rng = random.Random(zlib.crc32(stem.encode()))
  variants = [('c', 0., 0.)]
  if v > 3. and past_yaw < .05:
    kind = rng.choice(('y', 'h', 'b'))
    dy = rng.choice((-1, 1))*rng.uniform(.3, 1.) if kind in 'yb' else 0.
    dpsi = rng.choice((-1, 1))*math.radians(rng.uniform(.6, 2.6)) if kind in 'hb' else 0.
    variants.append(('r', dy, dpsi))
  intr, extr = calibration(local_dir, chunk)
  for key, dy, dpsi in variants:
    teacher, student = variant(z, intr, extr, clip, tau, v, dy, dpsi)
    rz = Rotation.from_euler('z', dpsi).as_matrix()
    front = np.isin(z['camera_indices'], (1, 6))
    tmp = spool/f'{stem}.{key}.tmp'
    with open(tmp, 'wb') as f:
      np.savez(f, image_frames=teacher, camera_indices=z['camera_indices'], absolute_timestamps=z['absolute_timestamps'],
               student_frames=student, student_timestamps=z['absolute_timestamps'][front],
               ego_history_xyz=hist_xyz, ego_history_rot=hist_rot, gt_xyz=((gt-np.array([0., dy, 0.])) @ rz).astype(np.float32),
               gt_rot=np.einsum('ji,njk->nik', rz, gt_rot).astype(np.float32),
               shift=np.array([dy, dpsi], np.float32), speed=np.float32(v), chunk=np.int32(chunk))
    tmp.rename(spool/f'{stem}.{key}.npz')
  done.parent.mkdir(exist_ok=True)
  done.write_text(','.join(k for k, _, _ in variants))
  return stem, 'ok ' + ','.join(k for k, _, _ in variants)


if __name__ == '__main__':
  ap = argparse.ArgumentParser()
  ap.add_argument('local_dir', type=Path); ap.add_argument('spool', type=Path); ap.add_argument('chunks', type=int, nargs='+')
  ap.add_argument('--workers', type=int, default=10); ap.add_argument('--t0', default='3,5.5,8,10.5,13')
  ap.add_argument('--limit', type=int); ap.add_argument('--final', action='store_true', help='write decode.done for stage B')
  a = ap.parse_args()
  a.spool.mkdir(parents=True, exist_ok=True)
  avdi = interface(a.local_dir)
  jobs = []
  for chunk in a.chunks:
    clips = [c for c, row in avdi.clip_index.iterrows() if int(row['chunk']) == chunk and bool(row['clip_is_valid'])]
    jobs += [(a.local_dir, a.spool, chunk, c, float(t)) for c in clips for t in a.t0.split(',')]
  jobs = jobs[:a.limit]
  print(len(jobs), 'moments', flush=True)
  with Pool(a.workers) as pool:
    for n, (stem, status) in enumerate(pool.imap_unordered(one, jobs)):
      print(n+1, '/', len(jobs), stem, status, flush=True)
  if a.final:
    (a.spool.parent/'decode.done').write_text('done')
  print('decode done', flush=True)
