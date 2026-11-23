"""Balanced training manifest from classify.py stats (clean samples only, shuffled, one .npz path per line).

Usage: manifest.py STATS.json OUT.txt [--teacher-repeat N] [--straight-ratio R] [--comma-share S] [--seed N]
Turning moments (|human heading change at 6.4 s| > 0.1 rad) and slow moments are all kept once.
Straight and mild moments (< 0.1 rad) from turns120 and comma are drawn so that straights (incl. repeated teacher
samples) = R x turning moments; S is comma's share of that draw. Teacher samples (comma plans on Karel frames) repeat N times.
"""
import argparse
import json
import random
from collections import Counter

ap = argparse.ArgumentParser()
ap.add_argument('stats'); ap.add_argument('out')
ap.add_argument('--teacher-repeat', type=int, default=2); ap.add_argument('--straight-ratio', type=float, default=1.0)
ap.add_argument('--comma-share', type=float, default=0.5); ap.add_argument('--seed', type=int, default=0)
a = ap.parse_args()
rng = random.Random(a.seed)
stats = json.load(open(a.stats))
src = lambda p: 'teacher' if '/teacher/' in p else 'comma' if '/comma/' in p else 'turns'
group = {}
for p, v in stats.items():
  if v['kind'] != 'clean':
    continue
  h = v['h6']
  g = 'slow' if h is None else 'turn' if abs(h) > .1 else 'straight'
  group.setdefault((src(p), g), []).append(p)
for v in group.values():
  v.sort()
out = []
turning = group[('turns', 'turn')] + group[('comma', 'turn')]
out += turning + group.get(('turns', 'slow'), []) + group.get(('comma', 'slow'), [])
teacher = [p for g in ('turn', 'straight', 'slow') for p in group.get(('teacher', g), [])]
out += teacher * a.teacher_repeat
need = int(a.straight_ratio * (len(turning) + len(group.get(('teacher', 'turn'), [])) * a.teacher_repeat)) \
  - len(group.get(('teacher', 'straight'), [])) * a.teacher_repeat
n_comma = min(len(group[('comma', 'straight')]), int(need * a.comma_share))
n_turns = min(len(group[('turns', 'straight')]), need - n_comma)
out += rng.sample(group[('comma', 'straight')], n_comma) + rng.sample(group[('turns', 'straight')], n_turns)
rng.shuffle(out)
open(a.out, 'w').write('\n'.join(out) + '\n')
c = Counter((src(p), 'turn' if stats[p]['h6'] is not None and abs(stats[p]['h6']) > .1 else 'straight') for p in out)
print(len(out), dict(c))
