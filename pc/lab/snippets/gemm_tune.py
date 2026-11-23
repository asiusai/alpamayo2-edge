# bf16 tiled GEMM configurations for the prefill shapes (packed layout, same epilogue).
from functools import partial
from tinygrad import TinyJit
from openpilot.selfdrive.alpamayo.selected import amd_gemm
from openpilot.selfdrive.alpamayo.selected.kernels import tiled_gemm
SHAPES = globals().get('SHAPES', [(1344, 9216, 2048), (1344, 2048, 9216), (1344, 2048, 2048), (1344, 1024, 2048)])
CONFIGS = globals().get('CONFIGS', [dict(tile=(4,4)), dict(tile=(2,4)), dict(tile=(4,2)), dict(tile=(4,4), local=(2,2)), dict(tile=(4,4), local=(1,2)),
  dict(tile=(4,4), local=(2,1)), dict(tile=(2,4), local=(2,2)), dict(tile=(4,4), group_m=4), dict(tile=(4,4), unroll_k=2), dict(tile=(4,4), local=(2,2), unroll_k=2),
  dict(tile=(3,4)), dict(tile=(4,4), swap=True), dict(tile=(2,2), local=(2,2))])
for rows, n, k in SHAPES:
  xp = (Tensor.randn(rows, k)*.5).cast(dtypes.bfloat16).contiguous().realize(); wp = (Tensor.randn(n, k)*.5).cast(dtypes.bfloat16).contiguous().realize()
  s = Tensor([.01]).realize(); out = Tensor.empty(rows, n, dtype=dtypes.bfloat16)
  base = TinyJit(lambda a, b: Tensor.custom_kernel(out, a, b, s, s, fxn=partial(amd_gemm._gemm, packed_input=True))[0].realize())
  r = timeit(lambda: base(xp, wp), n=10); print(rows, n, k, 'amd_gemm', r.split()[1], f'{2*rows*n*k/float(r.split()[1])/1e9:.0f} TF', flush=True)
  for c in CONFIGS:
    o = dict(tile=(4,4), local=(1,1), swap=False, unroll_k=1, group_m=1); o.update(c)
    try:
      f = TinyJit(lambda a, b, o=o: Tensor.custom_kernel(out, a, b, s, s, fxn=partial(tiled_gemm.kernel, tile_m=o['tile'][0], tile_n=o['tile'][1],
            local_m=o['local'][0], local_n=o['local'][1], swap=o['swap'], unroll_k=o['unroll_k'], group_m=o['group_m']))[0].realize())
      r = timeit(lambda: f(xp, wp), n=10); med = float(r.split()[1])
      print(rows, n, k, c, f'{med:.3f} ms {2*rows*n*k/med/1e9:.0f} TF', flush=True)
    except Exception as e:
      print(rows, n, k, c, 'failed', type(e).__name__, str(e)[:80], flush=True)
