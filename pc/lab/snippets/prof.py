# Per-kernel GPU time (DEBUG=2, uncaptured) for the last prefill part, one vision pair and diffusion.
import io, re, contextlib, collections, samples
from tinygrad.helpers import Context
from tinygrad.uop.ops import UOp
m = rt.model
ARGS = globals().setdefault('ARGS', {})
if not ARGS:
  cls = type(rt.prefill); orig_call = cls.__call__
  def spy_prefill(self, *a, **k):
    ARGS['prefill'] = (a, k); return orig_call(self, *a, **k)
  orig_traj = m.trajectory
  def spy_traj(*a, **k):
    ARGS['traj'] = (a, k); return orig_traj(*a, **k)
  orig_enc = m.encode_group
  def spy_enc(*a):
    if len(a) == 4: ARGS['enc'] = a
    return orig_enc(*a)
  cls.__call__ = spy_prefill; m.trajectory = spy_traj; m.encode_group = spy_enc
  try:
    S = [samples.load(p) for p in samples.turn_samples()[:3]]
    for s in S:
      try: rt.predict(s)
      except Exception as e: print("predict failed", type(e).__name__)
  finally:
    cls.__call__ = orig_call; del m.trajectory; del m.encode_group
ANSI = re.compile(r'\x1b\[[0-9;]*m')
LINE = re.compile(r'\*\*\*\s+\S+\s+\d+\s+(\S+).*?tm\s+([\d.]+)(us|ms)')
def profile(label, fn, runs=2, top=25):
  fn(); m.synchronize()
  buf = io.StringIO()
  with contextlib.redirect_stdout(buf), Context(DEBUG=2):
    for _ in range(runs): fn()
    m.synchronize()
  t, cnt = collections.defaultdict(float), collections.Counter()
  for line in buf.getvalue().splitlines():
    mt = LINE.search(ANSI.sub('', line))
    if not mt: continue
    name, us = mt.group(1), float(mt.group(2))*(1000 if mt.group(3) == 'ms' else 1)
    t[name] += us/runs; cnt[name] += 1/runs
  print(f'== {label}: {sum(cnt.values()):.0f} kernels, {sum(t.values())/1000:.2f} ms GPU')
  for k, us in sorted(t.items(), key=lambda x: -x[1])[:top]:
    print(f'{us/1000:7.3f} {cnt[k]:6.0f} {us/cnt[k]:8.1f}  {k[:90]}')
  if not t: print(buf.getvalue()[:3000])
(x, cos, sin, max_tokens, length), _ = ARGS['prefill']
p = rt.prefill
def run_last():
  valid = UOp.variable(f'valid_prefill_heads32_length', 769, x.shape[1]-31).bind(length)
  Tensor.realize(*p.language(x, cos, sin, valid, 32))
def run_part0():
  xx, cc, ss = p.part_rows(0, x, cos, sin)
  p.part_language(xx, cc, ss, 0).realize()
profile('prefill part 0 (576 rows)', run_part0)
profile('prefill last part (1344 rows + heads)', run_last)
a, k = ARGS['traj']

profile('vision pair', lambda: Tensor.realize(*m.encode_group(*ARGS['enc'])), runs=1, top=30)
