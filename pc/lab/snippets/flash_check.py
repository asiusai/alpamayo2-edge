# Native gfx11 flash prefill attention vs the shimmed frozen object: error and GPU time per call.
from tinygrad import TinyJit
from openpilot.selfdrive.alpamayo.selected.kernels import frozen_ops
kv = (Tensor.randn(2,1,8,2304,128)*1.5).cast(dtypes.bfloat16).contiguous().realize()
for rows, off in ((576, 0), (192, 576), (1344, 768)):
  q = (Tensor.randn(1,16,rows,128)*1.5).cast(dtypes.bfloat16).contiguous().realize()
  fa = TinyJit(lambda q, kv: frozen_ops.prefill_attention(q, kv, off).realize())
  fb = TinyJit(lambda q, kv: frozen_ops.frozen_prefill_attention(q, kv, off).realize())
  a = fa(q, kv).float().numpy(); b = fb(q, kv).float().numpy()
  for _ in range(2): fa(q, kv); fb(q, kv)
  d = np.abs(a-b); rel = d/(np.abs(b)+1e-3)
  print(rows, off, 'max abs', float(d.max()), 'mean abs', float(d.mean()), 'max |b|', float(np.abs(b).max()), 'equal frac', float((a == b).mean()))
  print('  flash', timeit(lambda: fa(q, kv), n=10), '| frozen', timeit(lambda: fb(q, kv), n=10))
