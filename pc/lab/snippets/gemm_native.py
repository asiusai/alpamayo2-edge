# Native gfx11 packed BF16 GEMM vs the tinygrad _gemm: bit agreement and in-graph GPU time.
from functools import partial
from tinygrad import TinyJit
import alpamayo_pc.port as port
from openpilot.selfdrive.alpamayo.selected import amd_gemm
for rows, n, k, res in ((1344, 9216, 2048, False), (1344, 2048, 9216, True), (1344, 2048, 2048, True), (1344, 1024, 2048, False), (576, 9216, 2048, False), (192, 2048, 9216, True)):
  x = amd_gemm.pack((Tensor.randn(rows, k)*2).cast(dtypes.fp8e4m3)).cast(dtypes.bfloat16).contiguous().realize()
  w = amd_gemm.pack((Tensor.randn(n, k)*2).cast(dtypes.fp8e4m3)).cast(dtypes.bfloat16).contiguous().realize()
  xs, ws = Tensor([.03]).realize(), Tensor([.02]).realize()
  r = (Tensor.randn(rows, n)).cast(dtypes.bfloat16).contiguous().realize()
  ref = TinyJit(lambda x, w: Tensor.custom_kernel(Tensor.empty(rows, n, dtype=dtypes.bfloat16), x, w, xs, ws, *((r,) if res else ()),
                                                  fxn=partial(amd_gemm._gemm, packed_input=True, residual=r if res else None) if False else
                                                  (amd_gemm.prefill_residual if res else partial(amd_gemm._gemm, packed_input=True)))[0].realize())
  new = TinyJit(lambda x, w: port.native_gemm(x, w, xs, ws, r if res else None).realize())
  a, b = ref(x, w).float().numpy(), new(x, w).float().numpy()
  for _ in range(2): ref(x, w); new(x, w)
  d = np.abs(a-b)
  tr = float(timeit(lambda: ref(x, w), n=10).split()[1]); tn = float(timeit(lambda: new(x, w), n=10).split()[1])
  f = 2*rows*n*k
  print(rows, n, k, 'res' if res else '', f'equal {float((a == b).mean()):.4f} maxdiff {float(d.max()):.4g} | tinygrad {tr:.3f} ms {f/tr/1e9:.0f} TF | native {tn:.3f} ms {f/tn/1e9:.0f} TF', flush=True)
