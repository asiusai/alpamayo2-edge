"""Recovery score on teacher samples (target = comma's logged plan on real Karel frames, many while Alpamayo drifted).

Usage: evaluate_recovery.py SAMPLE_DIR [PREDICTIONS.npz ...]
For each prediction set (default: the frozen baseline_xyz side files): on samples where comma's plan moves at least
MIN_CORRECTION sideways by 3 s, the least-squares ratio of predicted to comma lateral at 1.5 s and 3 s (1 = corrects like
comma, 0 = keeps going straight) and the correlation; plus the median lateral difference to comma's plan on all samples.
"""
import sys
from pathlib import Path

import numpy as np

MIN_CORRECTION = .15  # m at 3 s


def load(directory, predictions=None):
  preds = dict(np.load(predictions)) if predictions else {}
  rows = []
  for f in sorted(Path(directory).glob('*.npz')):
    if f.name.endswith(('.frozen.npz', '.tmp.npz')):
      continue
    if predictions:
      p = preds.get(f.stem)
    else:
      side = f.with_suffix('.frozen.npz')
      p = np.load(side)['baseline_xyz'] if side.exists() else None
    if p is None:
      continue
    with np.load(f) as z:
      t = z['target_xyz']
    rows.append((t[[14, 29], 1], np.asarray(p)[[14, 29], 1]))
  return np.array([r[0] for r in rows]), np.array([r[1] for r in rows])


def main(directory, sets):
  for name in sets or [None]:
    t, p = load(directory, name)
    if not len(t):
      print(name, 'no samples'); continue
    sel = np.abs(t[:, 1]) >= MIN_CORRECTION
    out = [f'{Path(name).stem if name else "frozen baseline":24s} n {len(t):4d} correcting {sel.sum():4d}']
    for i, lab in ((0, '1.5 s'), (1, '3 s')):
      a, b = t[sel, i], p[sel, i]
      out.append(f'{lab}: ratio {np.sum(a*b)/np.sum(a*a):.2f} corr {np.corrcoef(a, b)[0, 1]:.2f}')
    out.append(f'median |pred-comma| at 3 s {np.median(np.abs(p[:, 1]-t[:, 1])):.2f} m')
    print('  '.join(out))


if __name__ == '__main__':
  main(sys.argv[1], sys.argv[2:])
