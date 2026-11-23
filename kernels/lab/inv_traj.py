sl = lab_import("stamplib")
from tinygrad.helpers import PROFILE
PROFILE.value = 0
from tinygrad.uop.ops import UOp
m = rt.model
INV = globals().setdefault('INV', {})
a, k = ARGS['traj']
actions, ac, ass, length, _, offset = a
STEPS = 2
m.pack_expert()
ST = globals().setdefault('ST', {})
s = ST['traj'] = sl.Stamped(lambda a, c, s_, n, p: m._trajectory(a, c, s_, n, STEPS, p))
ln = UOp.variable('cache_length', 1, m.cache[0].shape[3]).bind(length)
of = UOp.variable('action_position', 0, ac.shape[2]-64).bind(offset)
t0 = time.monotonic()
names, us, raw = s.run((actions, ac, ass, ln, of), m.synchronize, n=7)
print(f'capture+runs {time.monotonic()-t0:.0f} s; run-to-run total spread {raw.sum(axis=1).min()/1000:.3f}..{raw.sum(axis=1).max()/1000:.3f} ms')
INV['traj'] = (names, us)
TA = (actions, ac, ass, ln, of)
sl.table(names, us, top=30, per=STEPS, label=f'trajectory {STEPS} steps (per step)')
print('stamped wall', timeit(lambda: s(actions, ac, ass, ln, of), 10))
f1 = lambda st: m.trajectory(actions, ac, ass, length, st, offset).realize()
for st in (2, 10):
  print(f'production graph {st} steps', timeit(lambda: f1(st), 10))
