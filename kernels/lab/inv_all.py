sl = lab_import("stamplib")
from tinygrad.helpers import PROFILE
from tinygrad.uop.ops import UOp
from tinygrad import Device
PROFILE.value = 0
m = rt.model
INV = globals().setdefault('INV', {}); ST = globals().setdefault('ST', {}); SA = globals().setdefault('SA', {})
WHICH = globals().get('WHICH', ('traj', 'prefill', 'part0', 'part1', 'vision'))
from openpilot.selfdrive.alpamayo.selected.model import PREFILL_SPLITS
def build(name):
  if name == 'traj':
    a, k = ARGS['traj']; actions, ac, ass, length, _, offset = a
    m.pack_expert()
    ST[name] = sl.Stamped(lambda a, c, s_, n, p: m._trajectory(a, c, s_, n, 2, p))
    SA[name] = (actions, ac, ass, UOp.variable('cache_length', 1, m.cache[0].shape[3]).bind(length), UOp.variable('action_position', 0, ac.shape[2]-64).bind(offset))
  elif name == 'prefill':
    x, cos, sin, _, length = ARGS['prefill']
    ST[name] = sl.Stamped(lambda x, c, s_, n: rt.prefill.language(x, c, s_, n, 32))
    SA[name] = (x, cos, sin, UOp.variable('valid_prefill_heads32_length', PREFILL_SPLITS[-1]+1, x.shape[1]-32+1).bind(length))
  elif name in ('part0', 'part1'):
    part = int(name[-1]); x, cos, sin, _, length = ARGS['prefill']
    start = PREFILL_SPLITS[part]
    rows = [t.clone().realize() for t in rt.prefill.part_rows(part, x, cos, sin)]
    ST[name] = sl.Stamped(lambda x, c, s_: rt.prefill.part_language(x, c, s_, start))
    SA[name] = tuple(rows)
  elif name == 'vision':
    from openpilot.selfdrive.alpamayo.selected.vision_pair import PairCapture
    from openpilot.selfdrive.alpamayo.selected.vision_pair.programs import PairPrograms
    cap = PairCapture(rt.vision.pair_weights.model, PairPrograms(Device[Device.DEFAULT].renderer))
    ST[name] = sl.Stamped(cap.encode_group)
    ST[name].jit.cnt = 1
    SA[name] = tuple(ARGS['pairs'])
for name in WHICH:
  t0 = time.monotonic()
  if name not in ST or name not in SA: build(name)
  s, args = ST[name], SA[name]
  names, us, raw = s.run(args, m.synchronize, n=6)
  _, cold, _ = s.run(args, m.synchronize, idle=0.3)
  INV[name] = (names, us, cold)
  per = 2 if name == 'traj' else 1
  print(f'[{name}] {time.monotonic()-t0:.0f} s; back-to-back run totals ms {np.round(raw.sum(axis=1)/1000, 2).tolist()}; after 0.3 s idle {cold.sum()/1000:.2f} ms')
  sl.table(names, us, top=24, per=per, label=name + (' (per step)' if per == 2 else ''))
  print('stamped graph wall', timeit(lambda: s(*args), 8))
print('temp', max(int(open(f).read()) for f in __import__('glob').glob('/sys/class/thermal/thermal_zone*/temp')))
