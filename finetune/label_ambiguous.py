"""Mark route-choice turns as kind 'ambiguous' so training (--kinds clean) skips them; nothing is deleted.

Usage: label_ambiguous.py DIR [DIR ...] [--dry]
With "Follow the road." a human turn is a route choice when the frozen model (its trajectory and its reasoning, from
<sample>.frozen.npz and the meta 'reasoning') neither turns that way nor names that side. Such labels contradict what the
inputs show. Kept: straight and gentle curves (|human heading change by 6.4 s| < 0.3 rad), turns the model already takes
the same way (same sign and at least 15% of the human's change) or names, and human straight where the model turned.
"""
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_turns import heading  # noqa: E402

TURN_RAD, SAME_FRACTION = .3, .15


def side_named(text):
  t = (text or '').lower()
  left, right = bool(re.search(r'\b(turn|veer|bear|keep|curve|merge|steer)\w*\b[^.]*\bleft\b', t)), bool(re.search(r'\b(turn|veer|bear|keep|curve|merge|steer)\w*\b[^.]*\bright\b', t))
  return {(True, False): 1, (False, True): -1}.get((left, right), 0)


def main(dirs, dry=False):
  counts = Counter()
  for d in dirs:
    for f in sorted(Path(d).glob('*.npz')):
      if f.name.endswith(('.frozen.npz', '.tmp.npz')):
        continue
      meta_path, side = f.with_suffix('.json'), f.with_suffix('.frozen.npz')
      if not meta_path.exists() or not side.exists():
        counts['no frozen pass'] += 1
        continue
      meta = json.loads(meta_path.read_text())
      if meta['kind'] not in ('clean', 'ambiguous'):
        continue
      with np.load(f) as z:
        target = z['target_xyz'][:, :2]
      with np.load(side) as z:
        pred = z['baseline_xyz'][:, :2]
      h, p = heading(target, 63), heading(pred, 63)
      named = side_named(meta.get('reasoning'))
      if h is None or abs(h) < TURN_RAD:
        kind, why = 'clean', 'straight or gentle'
      elif (p is not None and p*h > 0 and abs(p) >= SAME_FRACTION*abs(h)) or named == math.copysign(1, h):
        kind, why = 'clean', 'turn the model takes'
      else:
        kind, why = 'ambiguous', 'route choice'
      counts[why] += 1
      if kind != meta['kind'] and not dry:
        meta['kind'] = kind
        meta['ambiguous_reason'] = why if kind == 'ambiguous' else None
        meta_path.write_text(json.dumps(meta))
  print(dict(counts))


if __name__ == '__main__':
  main([a for a in sys.argv[1:] if not a.startswith('--')], '--dry' in sys.argv)
