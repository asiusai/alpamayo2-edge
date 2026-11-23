from functools import partial
from tinygrad import TinyJit
from openpilot.selfdrive.alpamayo.selected import amd_gemm
from openpilot.selfdrive.alpamayo.selected.kernels import frozen_ops
dev = 'AMD'
for rows, n, k in ((1344, 9216, 2048), (1344, 2048, 9216), (1344, 2048, 2048)):
  xf = (Tensor.randn(rows, k)*.5).cast(dtypes.fp8e4m3).contiguous().realize()
  wf = (Tensor.randn(n, k)*.5).cast(dtypes.fp8e4m3).contiguous().realize()
  xp, wp = amd_gemm.pack(xf).realize(), amd_gemm.pack(wf).realize()
  xb, wb = xp.cast(dtypes.bfloat16).contiguous().realize(), wp.cast(dtypes.bfloat16).contiguous().realize()
  xh, wh = xp.cast(dtypes.half).contiguous().realize(), wp.cast(dtypes.half).contiguous().realize()
  s = Tensor([.01]).realize()
  out = Tensor.empty(rows, n, dtype=dtypes.bfloat16)
  flops = 2*rows*n*k
  for label, a, b in (('fp8', xp, wp), ('bf16', xb, wb), ('f16', xh, wh)):
    f = TinyJit(lambda a, b: Tensor.custom_kernel(out, a, b, s, s, fxn=partial(amd_gemm._gemm, packed_input=True))[0].realize())
    f = partial(f, a, b)
    r = timeit(f, n=10)
    med = float(r.split()[1])
    print(rows, n, k, label, r, f'{flops/med/1e9:.1f} TFLOPS', flush=True)
  if n == 9216:
    f = partial(TinyJit(lambda a, b: frozen_ops.fused_mlp(a, b, s, s, s, rows).realize()), xp, wp)
    r = timeit(f, n=10); med = float(r.split()[1]); print('frozen mlp shim', r, f'{flops/med/1e9:.1f} TFLOPS')
