"""Score two-camera predictions on the PhysicalAI distillation heldout set against the human path and the six-camera teacher.

Usage: nv_distill_score.py HELDOUT_DIR [NAME=PREDICTIONS.npz ...]
Always scores 'teacher6' (the six-camera stock plan, target_xyz) and 'stock2' (the frozen two-camera car recipe,
baseline_xyz from the frozen side file). Each PREDICTIONS.npz holds '<sample stem>' -> (64,3) xyz.
Clean samples: turn ratio (predicted / driven heading change at 3 s), lateral error to the human path at 3 s and 6.4 s,
mean lateral bias on straights (left positive) and the distance to the teacher plan.
Recovery samples: fraction of the starting error kept at 20/40/60 m, relative to the same source's clean plan.
"""
import json
import math
import sys
from pathlib import Path

import numpy as np

FROZEN = Path('/mnt/nvme-500/alpamayo-train/frozen-cs')
DATA = Path('/mnt/nvme-500/alpamayo-train')


def heading(p, i):
  d = p[i]-p[i-3]
  return math.atan2(d[1], d[0])


def at(path, x):
  return float(np.interp(x, path[:, 0], path[:, 1])) if path[-1, 0] >= x and (np.diff(path[:, 0]) > 0).all() else math.nan


def main(directory, named):
  directory = Path(directory)
  sources = {'teacher6': {}, 'stock2': {}}
  meta, gt = {}, {}
  for f in sorted(directory.glob('*.npz')):
    stem = f.name[:-4]
    with np.load(f) as z:
      sources['teacher6'][stem], gt[stem] = z['target_xyz'], z['gt_xyz']
    side = FROZEN/directory.resolve().relative_to(DATA)/f'{stem}.frozen.npz'
    if side.exists():
      with np.load(side) as z:
        sources['stock2'][stem] = z['baseline_xyz']
    meta[stem] = json.loads(f.with_suffix('.json').read_text())
  for item in named:
    name, path = item.split('=', 1)
    with np.load(path) as z:
      sources[name] = {k: z[k] for k in z.files}
  print(f'{len(meta)} samples; clean {sum(m["kind"] == "clean" for m in meta.values())}')
  print(f'{"source":10s} {"n":>4s} {"turn":>5s} {"|lat|3s":>8s} {"6.4s":>6s} {"bias3s":>7s} {"6.4s":>6s} {"to teacher 3s":>13s} | kept 20/40/60 m (n)')
  for name, preds in sources.items():
    rows, kept = [], []
    for stem, m in meta.items():
      p = preds.get(stem)
      if p is None:
        continue
      if m['kind'] == 'clean':
        g = gt[stem]
        rows.append((m['speed_m_s'], heading(g, 29), heading(p, 29), p[29, 1]-g[29, 1], p[63, 1]-g[63, 1],
                     float(np.linalg.norm(p[29, :2]-sources['teacher6'][stem][29, :2]))))
      else:
        ref = preds.get(stem[:-2]+'-c')
        if ref is None:
          continue
        dy, dpsi = m['dy_m'], m['dpsi_rad']
        c, s = math.cos(dpsi), math.sin(dpsi)
        rx, ry = ref[:, 0], ref[:, 1]-dy
        ref_v = np.c_[c*rx+s*ry, -s*rx+c*ry]
        row = []
        for x in (20., 40., 60.):
          start = dy+math.tan(dpsi)*x
          row.append((at(p, x)-at(ref_v, x))/start if abs(start) > .2 else math.nan)
        kept.append(row)
    a, k = np.array(rows), np.array(kept)
    if not len(a):
      continue
    mv = a[:, 0] > 3
    turn = mv & (np.abs(a[:, 1]) > .15)
    straight = mv & (np.abs(a[:, 1]) < .03)
    ratio = np.sum(a[turn, 2]*a[turn, 1])/np.sum(a[turn, 1]**2) if turn.any() else math.nan
    kcells = ' '.join(f'{np.nanmedian(k[:, i]):+.2f}' for i in range(3)) + f' ({len(k)})' if len(k) else '-'
    print(f'{name:10s} {mv.sum():4d} {ratio:5.2f} {np.median(np.abs(a[mv, 3])):8.2f} {np.median(np.abs(a[mv, 4])):6.2f} '
          f'{a[straight, 3].mean():+7.2f} {a[straight, 4].mean():+6.2f} {np.median(a[mv, 5]):13.2f} | {kcells}')


if __name__ == '__main__':
  main(sys.argv[1], sys.argv[2:])
