# A/B of expert_block variants in stamped two-step trajectory graphs: bit-exact actions and per-step GPU time.
import types, gc
sl = lab_import('stamplib'); en = lab_import('expert_new')
from tinygrad.helpers import PROFILE
from tinygrad.uop.ops import UOp
PROFILE.value = 0
m = rt.model
a, k = ARGS['traj']; actions, ac, ass, length, _, offset = a
m.pack_expert()
targs = (actions, ac, ass, UOp.variable('cache_length', 1, m.cache[0].shape[3]).bind(length), UOp.variable('action_position', 0, ac.shape[2]-64).bind(offset))
MODES = globals().get('MODES', ('current', 'signed', 'signed_staged', 'staged'))
G, OUT = {}, {}
for mode in MODES:
  if mode != 'current': m.expert_block = types.MethodType(en.make(mode), m)
  try:
    s = sl.Stamped(lambda a, c, s_, n, p: m._trajectory(a, c, s_, n, 2, p))
    for _ in range(3): out = s(*targs)
    OUT[mode] = out.numpy().copy(); G[mode] = s
  finally:
    if 'expert_block' in vars(m): del m.expert_block
m.synchronize()
res = {n: [] for n in G}
for r in range(6):
  for n, s in (list(G.items()) if r % 2 == 0 else list(G.items())[::-1]):
    for _ in range(4): s(*targs)
    m.synchronize(); s.zero()
    for _ in range(3): s(*targs)
    cnt, st = s.read(m.synchronize); per = cnt//3
    res[n].append(np.median(np.diff(st.reshape(3, per), axis=1).sum(axis=1)[1:])/100.0/2)
base = np.median(res['current'])
for n, s in G.items():
  names = s.kernel_names()
  print(f'{n:<14} {len(names)//2:4d} launches per step, {np.median(res[n])/1000:.3f} ms per step ({100*(np.median(res[n])/base-1):+.1f}%), bit-exact actions: {np.array_equal(OUT[n].view(np.uint32), OUT["current"].view(np.uint32))}')
for s in G.values(): s.jit.reset(); s.zero.reset()
G.clear(); gc.collect()
