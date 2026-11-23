"""How much of a fake past turn in the ego history does the plan copy? (copycat test; images unchanged)

Usage (alpamayo-train env): history_copy_test.py GPU SAMPLE_DIR [--lora DIR] [--steps 1] [--limit 60] [--curvature 0.002]
On straight clean samples, replaces the 2 s ego history with the same speed profile on an arc of curvature +/-k (the car
has been turning that way), keeps the frozen reasoning tokens, samples the car's way and reports the lateral change at 3 s
relative to the unperturbed plan, as a fraction of what continuing the arc would give (1 = fully copies the history).
"""
import argparse
import os
import sys
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument('gpu', type=int); ap.add_argument('samples', type=Path)
ap.add_argument('--lora', type=Path); ap.add_argument('--steps', type=int, default=1); ap.add_argument('--limit', type=int, default=60)
ap.add_argument('--curvature', type=float, default=.002)
a = ap.parse_args()
TRAIN = str(Path(__file__).resolve().parents[1]/'train'); sys.path.insert(0, TRAIN); sys.path.insert(0, str(Path(__file__).resolve().parent))
os.chdir(TRAIN)
from alpamayo_train.common import start  # noqa: E402
gpu_mod = start(a.gpu, 8)
import json  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402
from alpamayo_train.data import Preparer, discover  # noqa: E402
from alpamayo_train.model import ORIGINAL_CHECKPOINT, ExpertTrainer  # noqa: E402
from evaluate_turns import heading  # noqa: E402

refs, _ = discover([a.samples], kinds={'clean'})
pre = Preparer(ORIGINAL_CHECKPOINT)
dev = gpu_mod.open_device(a.gpu)
tr = ExpertTrainer(ORIGINAL_CHECKPOINT, dev)
if a.lora:
  tr.load_lora(a.lora); tr.merge_lora_inplace()


def arc_history(xyz, k):
  """Same distances along the path, bent onto a constant-curvature arc that ends at the origin heading forward."""
  x = xyz[0, :, 0].astype(np.float64)
  s = -np.r_[np.cumsum(np.hypot(np.diff(xyz[0, ::-1, 0]), np.diff(xyz[0, ::-1, 1])))[::-1], 0.]  # signed arc length <= 0
  th = k*s
  nx = np.where(abs(k) > 1e-9, np.sin(th)/k if k else s, s)
  ny = np.where(abs(k) > 1e-9, (1-np.cos(th))/k if k else 0*s, 0*s)
  out = xyz.copy(); out[0, :, 0], out[0, :, 1] = nx, ny
  rot = Rotation.from_euler('z', th[:, None]).as_matrix().astype(np.float32)[None]
  return out.astype(np.float32), rot


def plan(arrays):
  item = pre(refs_by[arrays['_stem']]) if False else None
  return item


refs_by = {r.stem: r for r in refs}
rows = []
for r in refs:
  if len(rows) >= a.limit:
    break
  item = pre(r)
  t = item['target_xyz'][:, :2]
  h = heading(t, 63)
  if h is None or abs(h) > .03 or item['meta'].get('speed_m_s', 10) < 6:
    continue
  from alpamayo_train.data import load_arrays
  base = load_arrays(r)
  res = {}
  for k in (0., a.curvature, -a.curvature):
    arr = dict(base)
    if k:
      arr['ego_history_xyz'], arr['ego_history_rot'] = arc_history(base['ego_history_xyz'], k)
    prepared = pre.prompt(arr)
    prepared.update(reasoning_ids=item['reasoning_ids'], t0_states=None)
    prepared['history_xyz'], prepared['history_rot'] = prepared['history_xyz'], prepared['history_rot']
    hx, hr = prepared['history_xyz'], prepared['history_rot']
    prepared['t0_states'] = pre.action_space.estimate_t0_states(hx, hr)
    prefix = tr.prefix(prepared)
    action = tr.sample_car(prefix, steps=a.steps)
    xyz = pre.decode(tr.stager.d2h(action), prepared)[0][0].numpy()
    res[k] = xyz[29, 1]
  v = float(item['meta'].get('speed_m_s', 10))
  copy_y = a.curvature*(3*v)**2/2  # lateral at 3 s if the arc continued
  rows.append(((res[a.curvature]-res[0.])/copy_y, (res[0.]-res[-a.curvature])/copy_y))
r = np.array(rows)
print(json.dumps({'samples': len(r), 'lora': str(a.lora), 'steps': a.steps, 'curvature': a.curvature,
                  'copied_fraction_left_turn': round(float(np.median(r[:, 0])), 2), 'copied_fraction_right_turn': round(float(np.median(r[:, 1])), 2)}))
