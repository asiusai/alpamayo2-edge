"""Score camera sets on the Asius six-camera replay at zero noise against the recorded path.

Usage: demo_zero_score.py SAMPLE_DIR PRED.npz [PRED.npz ...]   (predictions from nv_shift_eval.py, keys '<time_us>.c/<set>')
Turn ratio: predicted / driven heading change at 3 s (least squares over turns). Lean: prediction minus the recorded path
(left positive), mean on all moving moments and on straights, at 3 s and 6.4 s.
"""
import math
import sys
from pathlib import Path

import numpy as np


def heading(p, i):
  d = p[i]-p[i-3]
  return math.atan2(d[1], d[0])


def main(samples, files):
  preds = {}
  for f in files:
    with np.load(f) as z:
      preds.update({k: z[k] for k in z.files})
  gt, speed = {}, {}
  for f in Path(samples).glob('*.npz'):
    with np.load(f) as z:
      gt[f.stem], speed[f.stem] = z['gt_xyz'], float(z['speed'])
  print(f'{"set":4s} {"n":>4s} {"turns":>5s} {"ratio":>6s} {"lean 3s":>8s} {"6.4s":>6s} {"straights n":>11s} {"3s":>6s} {"6.4s":>6s} {"|lat| 3s":>8s}')
  for cset in sorted({k.split('/')[1] for k in preds}):
    rows = np.array([(speed[k.split('/')[0]], heading(gt[k.split('/')[0]], 29), heading(p, 29), p[29, 1]-gt[k.split('/')[0]][29, 1],
                      p[63, 1]-gt[k.split('/')[0]][63, 1]) for k, p in preds.items() if k.split('/')[1] == cset])
    mv = rows[:, 0] > 3
    turn, straight = mv & (np.abs(rows[:, 1]) > .15), mv & (np.abs(rows[:, 1]) < .03)
    ratio = np.sum(rows[turn, 2]*rows[turn, 1])/np.sum(rows[turn, 1]**2)
    print(f'{cset:4s} {mv.sum():4d} {turn.sum():5d} {ratio:6.2f} {rows[mv, 3].mean():+8.2f} {rows[mv, 4].mean():+6.2f} {straight.sum():11d} '
          f'{rows[straight, 3].mean():+6.2f} {rows[straight, 4].mean():+6.2f} {np.median(np.abs(rows[mv, 3])):8.2f}')


if __name__ == '__main__':
  main(sys.argv[1], sys.argv[2:])
