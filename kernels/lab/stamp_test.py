sl = lab_import('stamplib')
def chain(x):
  for _ in range(4):
    mx = x.float().abs().max().realize()
    x = (x.float()/(mx+1)).cast(dtypes.bfloat16).contiguous().realize()
  return x
x = Tensor.randn(1344, 2048, dtype=dtypes.bfloat16).contiguous().realize()
s = sl.Stamped(chain)
names, us, raw = s.run((x,), rt.model.synchronize, n=5)
for n,u in zip(names, us): print(f'{u:8.1f} {n}')
print(raw[:, :6])
# reference: wall for unstamped chain
from tinygrad import TinyJit
j = TinyJit(chain)
for _ in range(3): j(x)
print('unstamped', timeit(lambda: j(x), 10)); print('stamped', timeit(lambda: s(x), 10))
