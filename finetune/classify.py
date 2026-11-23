"""Per-sample stats for manifest building: kind, speed, human heading change at 3 s / 6.4 s, lateral at 6.4 s.

Usage: classify.py OUT.json DIR [DIR ...]   (CPU only; reads target_xyz and meta)
"""
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from evaluate_turns import heading


def one(f):
  meta = json.loads(f.with_suffix('.json').read_text())
  with np.load(f) as z:
    t = z['target_xyz'][:, :2]
    hist = z['ego_history_xyz'][0, :, :2]
  h3, h6 = heading(t, 29), heading(t, 63)
  hh = heading(-hist[::-1] + hist[-1], 19) if len(hist) >= 20 else None
  return str(f), {'kind': meta.get('kind'), 'speed': meta.get('speed_m_s'), 'route': meta.get('route'),
                  'h3': h3, 'h6': h6, 'y6': float(t[63, 1]), 'dist': float(np.hypot(*t[63]))}


files = [f for d in sys.argv[2:] for f in sorted(Path(d).glob('*.npz'))
         if not f.name.endswith(('.frozen.npz', '.tmp.npz')) and f.with_suffix('.json').exists()]
with ThreadPoolExecutor(12) as pool:
  out = dict(pool.map(one, files))
Path(sys.argv[1]).write_text(json.dumps(out))
print(len(out))
