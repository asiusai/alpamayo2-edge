"""Predictions under ego-history variants and Euler step counts (frozen reasoning tokens, the car's sampler).

Usage (alpamayo-train env): history_sweep.py GPU SAMPLE_DIR OUT_PREFIX [--lora DIR] [--steps 1,2,10] [--history real,straight,half]
[--kinds clean] [--limit N]
History variants keep the same distances travelled (speed profile) and change only the path shape: `straight` removes all
past turning (zero curvature, zero yaw), `half` halves the heading profile. Writes OUT_PREFIX-<history>-s<steps>.npz.
"""
import argparse
import os
import sys
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument('gpu', type=int); ap.add_argument('samples', type=Path); ap.add_argument('out', type=str)
ap.add_argument('--lora', type=Path); ap.add_argument('--steps', default='1,2,10'); ap.add_argument('--history', default='real,straight,half')
ap.add_argument('--kinds', default='clean'); ap.add_argument('--limit', type=int)
a = ap.parse_args()
TRAIN = str(Path(__file__).resolve().parents[1]/'train'); sys.path.insert(0, TRAIN)
os.chdir(TRAIN)
from alpamayo_train.common import start  # noqa: E402
gpu_mod = start(a.gpu, 8)
import numpy as np  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402
from alpamayo_train.data import Preparer, discover, load_arrays  # noqa: E402
from alpamayo_train.model import ORIGINAL_CHECKPOINT, ExpertTrainer  # noqa: E402

refs, _ = discover([a.samples], kinds=set(a.kinds.split(',')) if a.kinds else None, limit=a.limit)
pre = Preparer(ORIGINAL_CHECKPOINT)
tr = ExpertTrainer(ORIGINAL_CHECKPOINT, gpu_mod.open_device(a.gpu))
if a.lora:
  tr.load_lora(a.lora); tr.merge_lora_inplace()
steps = [int(s) for s in a.steps.split(',')]
modes = a.history.split(',')


def reshape_history(xyz, rot, scale):
  """Keep the step lengths, scale the yaw profile (relative to the current pose) and re-integrate, ending at the origin."""
  p = xyz[0, :, :2].astype(np.float64)
  yaw = Rotation.from_matrix(rot[0]).as_euler('zyx')[:, 0]*scale
  step = np.hypot(*np.diff(p, axis=0).T)
  mid = (yaw[:-1]+yaw[1:])/2
  d = np.stack([step*np.cos(mid), step*np.sin(mid)], -1)
  q = np.vstack([-np.cumsum(d[::-1], axis=0)[::-1], np.zeros((1, 2))])
  out = xyz.copy(); out[0, :, :2] = q
  r = Rotation.from_matrix(rot[0]).as_euler('zyx'); r[:, 0] = yaw
  return out.astype(np.float32), Rotation.from_euler('zyx', r).as_matrix().astype(np.float32)[None]


preds = {(m, s): {} for m in modes for s in steps}
for n, ref in enumerate(refs):
  try:
    item = pre(ref)
  except ValueError:
    continue
  base = load_arrays(ref)
  for m in modes:
    arr = dict(base)
    if m != 'real':
      arr['ego_history_xyz'], arr['ego_history_rot'] = reshape_history(base['ego_history_xyz'], base['ego_history_rot'],
                                                                       0. if m == 'straight' else .5)
    prepared = pre.prompt(arr)
    prepared['reasoning_ids'] = item['reasoning_ids']
    prepared['t0_states'] = pre.action_space.estimate_t0_states(prepared['history_xyz'], prepared['history_rot'])
    prefix = tr.prefix(prepared)
    for s in steps:
      action = tr.sample_car(prefix, steps=s)
      preds[(m, s)][ref.stem] = pre.decode(tr.stager.d2h(action), prepared)[0][0].numpy().astype(np.float32)
    del prefix
  if n % 50 == 0:
    print(n+1, '/', len(refs), flush=True)
for (m, s), p in preds.items():
  np.savez(f'{a.out}-{m}-s{s}.npz', **p)
print('done', flush=True)
