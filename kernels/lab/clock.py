sl = lab_import("stamplib")
from tinygrad.helpers import PROFILE
PROFILE.value = 0
x = Tensor.randn(64, dtype=dtypes.bfloat16).contiguous().realize()
s = sl.Stamped(lambda x: (x.float()+1).cast(dtypes.bfloat16).contiguous().realize())
for _ in range(3): s(x)
rt.model.synchronize()
for wait in (0.5, 1.0, 2.0):
  s.zero(); s(x); rt.model.synchronize(); t0 = time.perf_counter(); time.sleep(wait); s(x); rt.model.synchronize(); t1 = time.perf_counter()
  n, st = s.read(rt.model.synchronize)
  print(n, wait, 'ticks', int(st[-1]-st[0]), 'host s', t1-t0, 'Hz', (st[-1]-st[0])/(t1-t0))
