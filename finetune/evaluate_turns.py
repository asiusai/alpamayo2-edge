"""Turn metrics for clean samples: frozen baseline (<sample>.frozen.npz) or PREDICTIONS.npz ('<stem>' -> (64,3) xyz).

Usage: evaluate_turns.py SAMPLE_DIR [PREDICTIONS.npz] [--rows] [--all]
--all also counts samples labeled 'ambiguous' (route-choice turns; default: clean only).
Horizon points are 0.1 s apart (index 29 = 3 s, 63 = 6.4 s). Headings come from the path direction (atan2 of a segment).
- turn ratio: least-squares slope of predicted on human heading change, for moments whose human change exceeds 0.1 rad.
- approach: the human goes straight for 3 s (|change| < 0.05 rad) and then turns (> 0.3 rad by 6.4 s). Early turning shows
  as a predicted 3 s heading toward the coming turn; reported as the mean signed predicted 3 s heading toward the turn.
- turn path error: mean lateral distance from the predicted to the human path over 6.4 s, on turning moments.
"""
import json
import math
import sys
from pathlib import Path

import numpy as np


def heading(xy, i, back=3):
  """Path direction around index i, from points far enough apart to be stable at low speed."""
  j = i
  while j > 0 and np.hypot(*(xy[i]-xy[j-1])) < 1.0 and i-j < 12:
    j -= 1
  a, b = (xy[j-1] if j > 0 else np.zeros(2)), xy[i]
  if np.hypot(*(b-a)) < .3:
    return None
  return math.atan2(b[1]-a[1], b[0]-a[0])


def path_error(pred, human):
  """Mean distance from each human point to the predicted polyline (dense)."""
  p = np.r_[np.zeros((1, 2)), pred]
  seg = np.linspace(0, 1, 8)
  dense = (p[:-1, None]*(1-seg[None, :, None])+p[1:, None]*seg[None, :, None]).reshape(-1, 2)
  d = np.linalg.norm(human[:, None]-dense[None], axis=-1).min(1)
  return float(d.mean())


def load(directory, predictions=None, kinds=('clean',)):
  preds = dict(np.load(predictions)) if predictions else {}
  rows = []
  for f in sorted(Path(directory).glob('*.npz')):
    if f.name.endswith(('.frozen.npz', '.tmp.npz')):
      continue
    meta = json.loads(f.with_suffix('.json').read_text())
    if meta['kind'] not in kinds:
      continue
    pred = preds.get(f.stem)
    if pred is None and not predictions and f.with_suffix('.frozen.npz').exists():
      with np.load(f.with_suffix('.frozen.npz')) as z:
        pred = z['baseline_xyz']
    if pred is None:
      continue
    with np.load(f) as z:
      target = z['target_xyz'][:, :2]
    rows.append((f.stem, meta, target, np.asarray(pred)[:, :2]))
  return rows


def main(directory, predictions=None, show_rows=False, kinds=('clean',)):
  rows = load(directory, predictions, kinds)
  turns3, turns6, approach, perr_turn, perr_straight, lat3 = [], [], [], [], [], []
  by_nav = {}
  for stem, meta, t, p in rows:
    h3, p3, h6, p6 = heading(t, 29), heading(p, 29), heading(t, 63), heading(p, 63)
    lat3.append(abs(p[29, 1]-t[29, 1]))
    if None in (h3, p3, h6, p6):
      continue
    if abs(h3) > .1:
      turns3.append((p3, h3))
    if abs(h6) > .1:
      turns6.append((p6, h6))
      e = path_error(p, t); perr_turn.append(e)
      by_nav.setdefault(meta.get('nav_text', '?'), []).append((p6, h6))
    elif abs(h6) < .03:
      perr_straight.append(path_error(p, t))
    if abs(h3) < .05 and abs(h6) > .3:
      approach.append(p3*math.copysign(1, h6))
    if show_rows:
      print(f'{stem} nav {meta.get("nav_text")!r} human {math.degrees(h3):+6.1f}/{math.degrees(h6):+6.1f} deg  '
            f'pred {math.degrees(p3):+6.1f}/{math.degrees(p6):+6.1f} deg')
  ratio = lambda v: float(np.sum(np.array(v)[:, 0]*np.array(v)[:, 1])/np.sum(np.array(v)[:, 1]**2)) if v else float('nan')
  out = dict(samples=len(rows), turn_ratio_3s=round(ratio(turns3), 3), n3=len(turns3), turn_ratio_6s=round(ratio(turns6), 3), n6=len(turns6),
             approach_early_deg=round(math.degrees(float(np.mean(approach))), 2) if approach else None, n_approach=len(approach),
             turn_path_err_m=round(float(np.median(perr_turn)), 3) if perr_turn else None,
             straight_path_err_m=round(float(np.median(perr_straight)), 3) if perr_straight else None,
             lat_err_3s_m=round(float(np.median(lat3)), 3) if lat3 else None,
             by_nav={k: (round(ratio(v), 2), len(v)) for k, v in by_nav.items()})
  print(json.dumps(out))
  return out


if __name__ == '__main__':
  args = [a for a in sys.argv[1:] if not a.startswith('--')]
  main(args[0], args[1] if len(args) > 1 else None, '--rows' in sys.argv, ('clean', 'ambiguous') if '--all' in sys.argv else ('clean',))
