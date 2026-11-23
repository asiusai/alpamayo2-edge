"""Random initial noise vs zero noise on unshifted straights: lean against the recorded path and spread between draws.

Usage: noise_score.py SHIFT_DIR ZERO.npz [ZERO.npz ...] -- DRAWS.npz [DRAWS.npz ...]
Zero files hold '<stem>.c/<set>', draw files '<stem>.c/<set>#k' (nv_shift_eval.py --temperature 1 --draws N).
Per camera set at 20/40/60 m (left positive): zero-noise mean lean; mean lean of single draws; lean of the draw average;
standard deviation between draws of one sample (median over samples); mean |draw - zero|.
"""
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

XS = (20., 40., 60.)


def at(path, x):
  return float(np.interp(x, path[:, 0], path[:, 1])) if path[-1, 0] >= x and (np.diff(path[:, 0]) > 0).all() else math.nan


def load(files):
  out = {}
  for f in files:
    with np.load(f) as z:
      out.update({k: z[k] for k in z.files})
  return out


def main(shift_dir, zero_files, draw_files):
  zero, draws = load(zero_files), load(draw_files)
  by = defaultdict(list)
  for key, p in draws.items():
    by[key.split('#')[0]].append(p)
  print(f'{"set":4s} {"n":>4s} | zero lean 20/40/60 | draw lean | mean-of-draws lean | sd between draws | mean |draw-zero|')
  for cset in sorted({k.split('/')[1] for k in by}):
    z_, d_, m_, sd_, dz_ = [], [], [], [], []
    for key, plist in by.items():
      name, s = key.split('/')
      if s != cset or key not in zero or not name.endswith('.c'):
        continue
      with np.load(Path(shift_dir)/f'{name}.npz') as f:
        gt = f['gt_xyz']
      g = np.array([at(gt, x) for x in XS])
      zz = np.array([at(zero[key], x) for x in XS])
      dd = np.array([[at(p, x) for x in XS] for p in plist])
      z_.append(zz-g); d_.append((dd-g).mean(0)); m_.append(np.array([at(np.mean(plist, 0), x) for x in XS])-g)
      sd_.append(dd.std(0)); dz_.append(np.abs(dd-zz).mean(0))
    f3 = lambda rows, fn=np.nanmean: ' '.join(f'{v:+.2f}' for v in fn(np.array(rows), 0))
    print(f'{cset:4s} {len(z_):4d} | {f3(z_)} | {f3(d_)} | {f3(m_)} | {f3(sd_, np.nanmedian)} | {f3(dz_)}')


if __name__ == '__main__':
  cut = sys.argv.index('--')
  main(sys.argv[1], sys.argv[2:cut], sys.argv[cut+1:])
