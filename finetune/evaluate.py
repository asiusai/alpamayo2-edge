"""Recovery and accuracy metrics for frozen (baseline_*) or fine-tuned predictions.

Usage: evaluate.py SAMPLE_DIR [PREDICTIONS.npz]
PREDICTIONS.npz (optional): arrays named '<sample stem>' with a (64,3) xyz prediction; otherwise each sample's baseline_xyz.
Recovery samples: the fraction of the starting lateral error the predicted path keeps at 20/40/60 m (target ~0 beyond 3 s).
Clean samples: lateral and heading error to the human path and the turn ratio (predicted / driven heading change at 3 s).
"""
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

DIST = (20., 40., 60.)


def lateral_at(xy, d):
  """y of a path at forward distance d (None when the path is shorter)."""
  x = np.r_[0., xy[:, 0]]; y = np.r_[0., xy[:, 1]]
  if x[-1] < d or np.any(np.diff(x) <= 0):
    return None
  return float(np.interp(d, x, y))


def human_in_shifted_frame(clean_xyz, dy, dpsi):
  c, s = math.cos(dpsi), math.sin(dpsi)
  hx, hy = clean_xyz[:, 0], clean_xyz[:, 1]-dy
  return np.stack([c*hx+s*hy, -s*hx+c*hy], -1)


def main(directory, predictions=None):
  directory = Path(directory)
  preds = dict(np.load(predictions)) if predictions else {}
  samples = {}
  for f in sorted(directory.glob('*.npz')):
    if f.name.endswith('.frozen.npz'):
      continue
    side = f.with_suffix('.frozen.npz')
    baseline = None
    with np.load(f) as z:
      target = z['target_xyz']
      if 'baseline_xyz' in z.files:
        baseline = z['baseline_xyz']
    if baseline is None and side.exists():
      with np.load(side) as z:
        baseline = z['baseline_xyz']
    pred = preds.get(f.stem, baseline)
    if pred is None:
      continue
    samples[f.stem] = (json.loads(f.with_suffix('.json').read_text()), target, pred)
  clean = {(m['route'], m['query_ns']): t for m, t, _ in samples.values() if m['kind'] == 'clean'}
  kept = defaultdict(list); target_kept = defaultdict(list); clean_err = []; turns = []
  for stem, (m, target, pred) in samples.items():
    if pred is None:
      continue
    pred = np.asarray(pred)[:, :2]
    if m['kind'] == 'clean':
      clean_err.append(np.abs(pred[:30, 1]-target[:30, 1]).mean())
      dpsi_h = math.atan2(target[30, 1]-target[28, 1], target[30, 0]-target[28, 0]); dpsi_p = math.atan2(pred[30, 1]-pred[28, 1], pred[30, 0]-pred[28, 0])
      if abs(dpsi_h) > .1:
        turns.append((dpsi_p, dpsi_h))
      continue
    human = clean.get((m['route'], m['query_ns']))
    if human is None:
      continue
    h = human_in_shifted_frame(human, m['dy_m'], m['dpsi_rad'])
    kind = 'shift' if m['dy_m'] and not m['dpsi_rad'] else 'heading' if m['dpsi_rad'] and not m['dy_m'] else 'both'
    for d in DIST:
      hp, pp, tp = lateral_at(h, d), lateral_at(pred, d), lateral_at(target[:, :2], d)
      if hp is None or pp is None or tp is None:
        continue
      start = m['dy_m']+d*math.sin(m['dpsi_rad'])  # the error at d if the car kept its pose (no correction)
      if abs(start) < .2:
        continue
      kept[(kind, d)].append((pp-hp)/start)
      target_kept[(kind, d)].append((tp-hp)/start)
  print(f'samples {len(samples)}  clean {len(clean)}')
  for kind in ('shift', 'heading', 'both'):
    row = [f'{kind:8s}']
    for d in DIST:
      v = kept.get((kind, d), [])
      row.append(f'{d:.0f} m kept {np.median(v):+.2f} (target {np.median(target_kept[(kind, d)]):+.2f}, n {len(v)})' if v else f'{d:.0f} m -')
    print('  '.join(row))
  if clean_err:
    print(f'clean: mean lateral error over 3 s {np.median(clean_err):.2f} m median')
  if turns:
    p, h = np.array(turns).T
    print(f'clean turns (n {len(turns)}): heading ratio at 3 s {float(np.sum(p*h)/np.sum(h*h)):.2f}')


if __name__ == '__main__':
  main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
