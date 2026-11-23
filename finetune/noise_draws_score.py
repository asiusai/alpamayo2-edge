"""Score noise_draws.py output against the human path of car-format samples (model frame, left positive).

Usage: noise_draws_score.py SAMPLE_DIR OUT_PREFIX
Per source (zero, each draw, the average of the draws): turn ratio at 3 s, median |lateral error| at 3 s on turns and on
straights, mean signed lean on straights at 3 s and 6.4 s, share of straight plans left of the human path at 6.4 s.
Also the spread between draws of one sample and the distance from the zero-noise plan.
"""
import glob
import json
import math
import sys
from pathlib import Path

import numpy as np


def heading(p, i):
  d = p[i]-p[i-3]
  return math.atan2(d[1], d[0])


samples, prefix = Path(sys.argv[1]), sys.argv[2]
src = {}
for f in sorted(glob.glob(prefix+'-*.npz')):
  with np.load(f) as z:
    src[f[len(prefix)+1:-4]] = {k: z[k] for k in z.files}
draws = [n for n in src if n != 'zero']
src['mean'] = {k: np.mean([src[d][k] for d in draws], 0) for k in src['zero']}
gt, speed = {}, {}
for k in src['zero']:
  with np.load(samples/f'{k}.npz') as z:
    gt[k] = z['target_xyz']
  speed[k] = json.loads((samples/f'{k}.json').read_text())['speed_m_s']
print(f'{"source":6s} {"turns":>5s} {"ratio":>6s} {"|lat|3s":>8s} {"straights":>9s} {"|lat|3s":>8s} {"lean 3s":>8s} {"6.4s":>6s} {"left share":>10s}')
for name, p in src.items():
  r = np.array([(speed[k], heading(gt[k], 29), heading(p[k], 29), p[k][29, 1]-gt[k][29, 1], p[k][63, 1]-gt[k][63, 1]) for k in p])
  mv = r[:, 0] > 3
  t, s = mv & (np.abs(r[:, 1]) > .15), mv & (np.abs(r[:, 1]) < .03)
  print(f'{name:6s} {t.sum():5d} {np.sum(r[t, 2]*r[t, 1])/np.sum(r[t, 1]**2):6.2f} {np.median(np.abs(r[t, 3])):8.2f} {s.sum():9d} '
        f'{np.median(np.abs(r[s, 3])):8.2f} {r[s, 3].mean():+8.2f} {r[s, 4].mean():+6.2f} {(r[s, 4] > 0).mean():10.2f}')
d3 = np.array([[src[d][k][29, 1] for d in draws] for k in src['zero']]); d6 = np.array([[src[d][k][63, 1] for d in draws] for k in src['zero']])
z3 = np.array([src['zero'][k][29, 1] for k in src['zero']]); z6 = np.array([src['zero'][k][63, 1] for k in src['zero']])
print(f'spread between draws (median sd): {np.median(d3.std(1)):.2f} m at 3 s, {np.median(d6.std(1)):.2f} m at 6.4 s; '
      f'draw - zero mean: {(d3-z3[:, None]).mean():+.2f} / {(d6-z6[:, None]).mean():+.2f} m; mean |draw - zero|: {np.abs(d3-z3[:, None]).mean():.2f} / {np.abs(d6-z6[:, None]).mean():.2f} m')
