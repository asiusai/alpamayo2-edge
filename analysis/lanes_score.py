"""Score offline plans on Karel's 2026-11-22 drive against the lane centre (comma's lane lines).

Usage: lanes_score.py LANE_TABLE.npz SAMPLE_DIR NAME=PREDICTIONS.npz [...]
Straight, well-marked moments only. For each source and distance x: fit (plan - lane centre at x) = keep * (car offset now) + bias.
keep: share of the car's current offset that the plan still has at x (0 = plan returns to the centre). bias: plan lean at x
for a centred car (right positive, so a left lean is negative). equilibrium = bias / (1 - keep): the offset where a car
that follows the plan settles. Also the mean and RMS distance of the plan from the lane centre.
"""
import json
import math
import sys
from pathlib import Path

import numpy as np

XS = (20, 40, 60)
tab = np.load(sys.argv[1]); a, segs, cols = tab['a'], tab['seg'], list(tab['cols'])
C = lambda n: a[:, cols.index(n)]
samples = Path(sys.argv[2])
rows, stems = [], []
for f in sorted(samples.glob('*.json')):
  m = json.loads(f.read_text())
  idx = np.flatnonzero(segs == m['route'])
  if not len(idx):
    continue
  j = idx[np.argmin(np.abs(C('sof_ns')[idx]-m['query_ns']))]
  if abs(C('sof_ns')[j]-m['query_ns']) > 80e6:
    continue
  rows.append(a[j]); stems.append(f.stem)
r = np.array(rows); R = lambda n: r[:, cols.index(n)]
bend = np.abs(R('c40')-2*R('c20')+R('c0'))
good = (R('v') > 8) & (R('prob') > .6) & (R('width') > 2.5) & (R('width') < 4.5) & (R('blink') == 0) & (bend < .5) & (np.abs(R('angle')) < 8)
off = -R('c0')
print(f'{len(r)} matched samples, {good.sum()} straight with good lanes (engaged {int((good & (R("lat") > 0)).sum())}); '
      f'car offset mean {off[good].mean():+.2f} m, sd {off[good].std():.2f} m')


def at(path, x):
  return -float(np.interp(x, path[:, 0], path[:, 1])) if path[-1, 0] >= x and (np.diff(path[:, 0]) > 0).all() else math.nan


sources = {'comma': None}
for item in sys.argv[3:]:
  name, path = item.split('=', 1)
  with np.load(path) as z:
    sources[name] = {k: z[k] for k in z.files}
print(f'{"source":10s} {"n":>4s} | ' + ' | '.join(f'{x} m: keep  bias  equil   rms' for x in XS))
for name, preds in sources.items():
  cells, n = [], 0
  for x in XS:
    e = R(f'm{x}')-R(f'c{x}') if preds is None else np.array([at(preds[s], x) if s in preds else math.nan for s in stems])-R(f'c{x}')
    ok = good & np.isfinite(e)
    keep, bias = np.linalg.lstsq(np.c_[off[ok], np.ones(ok.sum())], e[ok], rcond=None)[0]
    cells.append(f'     {keep:+.2f} {bias:+.2f} {bias/(1-keep) if keep < .97 else math.inf:+6.2f} {math.sqrt(np.mean(e[ok]**2)):5.2f}')
    n = ok.sum()
  print(f'{name:10s} {n:4d} | ' + ' | '.join(cells))
