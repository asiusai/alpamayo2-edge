"""Score nv_shift_eval.py predictions: fraction of a lateral or heading error kept at 20/40/60 m, by camera set.

Usage: nv_shift_score.py SHIFT_DIR PRED.npz [PRED.npz ...]
Reference for each sample and camera set: that set's own prediction on the unshifted variant (.c), moved into the virtual
ego frame. Lateral: kept = remaining lateral error / 0.5 m. Heading: kept = remaining lateral error / (tan(1.5 deg) * x),
the error of a car that continues straight in its own frame. 0 = back on the unshifted plan, 1 = no correction.
Also: the unshifted prediction's lateral error to the recorded path, by camera set.
"""
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

XS = (20., 40., 60.)


def at(path, x):
  """Lateral position where the path reaches x m forward, or nan when it ends earlier."""
  return float(np.interp(x, path[:, 0], path[:, 1])) if path[-1, 0] >= x and (np.diff(path[:, 0]) > 0).all() else math.nan


def main(shift_dir, pred_files):
  preds = {}
  for f in pred_files:
    with np.load(f) as z:
      preds.update({k: z[k] for k in z.files})
  sets = sorted({k.split('/')[1] for k in preds})
  kept, clean = defaultdict(list), defaultdict(list)
  for key, p in preds.items():
    name, cset = key.split('/')
    stem, var = name.rsplit('.', 1)
    with np.load(Path(shift_dir)/f'{name}.npz') as z:
      dy, dpsi, gt = float(z['shift'][0]), float(z['shift'][1]), z['gt_xyz']
    if var == 'c':
      clean[cset].append([at(p, x)-at(gt, x) for x in XS])
      continue
    ref = preds.get(f'{stem}.c/{cset}')
    if ref is None:
      continue
    c, s = math.cos(dpsi), math.sin(dpsi)
    rx, ry = ref[:, 0], ref[:, 1]-dy
    ref_v = np.c_[c*rx+s*ry, -s*rx+c*ry]  # unshifted prediction in the virtual ego frame
    row = []
    for x in XS:
      err = at(p, x)-at(ref_v, x)  # 0 when back on the unshifted plan
      start = dy if dy else math.tan(dpsi)*x  # error with no correction: keep the offset / continue straight
      row.append(err/start)
    kept[(cset, 'lateral' if dy else 'heading', var)].append(row)
  print(f'{"set":4s} {"error":8s} {"kept at 20 m":>14s} {"40 m":>12s} {"60 m":>12s}   (median; n)')
  for cset in sets:
    for kind in ('lateral', 'heading'):
      rows = np.array([r for (s_, k_, _), v in kept.items() if s_ == cset and k_ == kind for r in v])
      if not len(rows):
        continue
      cells = [f'{np.nanmedian(rows[:, i]):+.2f} ({np.isfinite(rows[:, i]).sum()})' for i in range(3)]
      print(f'{cset:4s} {kind:8s} {cells[0]:>14s} {cells[1]:>12s} {cells[2]:>12s}')
    for var in ('yL', 'yR', 'hL', 'hR'):
      rows = np.array(kept.get((cset, 'lateral' if var[0] == 'y' else 'heading', var), []))
      if len(rows):
        print(f'       {var:5s} ' + '  '.join(f'{np.nanmedian(rows[:, i]):+.2f}' for i in range(3)))
  print('\nunshifted: prediction minus recorded path (left positive), median |error| and mean at 20/40/60 m')
  for cset in sets:
    rows = np.array(clean[cset])
    if len(rows):
      print(f'{cset:4s} ' + '  '.join(f'{np.nanmedian(np.abs(rows[:, i])):.2f} / {np.nanmean(rows[:, i]):+.2f}' for i in range(3)), f' n {len(rows)}')


if __name__ == '__main__':
  main(sys.argv[1], sys.argv[2:])
