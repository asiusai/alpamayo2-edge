# In-graph time of GEMM variants: 20 back-to-back kernels per captured graph.
from functools import partial
from tinygrad import TinyJit
import alpamayo_pc.port as port
from openpilot.selfdrive.alpamayo.selected import amd_gemm
REPS = 20
SHAPES = globals().get('SHAPES', [(1344, 9216, 2048, False), (1344, 2048, 9216, True), (1344, 2048, 2048, False), (1344, 1024, 2048, False)])
TILES = globals().get('TILES', [(2, 4), (4, 4), (4, 2), (2, 2), (4, 8), (2, 8)])
for rows, n, k, res in SHAPES:
  x = amd_gemm.pack((Tensor.randn(rows, k)*2).cast(dtypes.fp8e4m3)).cast(dtypes.bfloat16).contiguous().realize()
  ws = [amd_gemm.pack((Tensor.randn(n, k)*2).cast(dtypes.fp8e4m3)).cast(dtypes.bfloat16).contiguous().realize() for _ in range(4)]
  xs = Tensor([.03]).realize(); r = Tensor.randn(rows, n).cast(dtypes.bfloat16).contiguous().realize()
  def run_ref(x, *w):
    outs = [Tensor.custom_kernel(Tensor.empty(rows, n, dtype=dtypes.bfloat16), x, w[i % 4], xs, xs, *((r,) if res else ()),
            fxn=amd_gemm.prefill_residual if res else partial(amd_gemm._gemm, packed_input=True))[0] for i in range(REPS)]
    return Tensor.realize(*outs)
  f = TinyJit(run_ref); f(x, *ws); f(x, *ws)
  med = float(timeit(lambda: f(x, *ws), n=5).split()[1])/REPS
  line = [f'{rows}x{n}x{k}{" res" if res else ""}: tinygrad {med*1e3:.0f} us {2*rows*n*k/med/1e9:.0f} TF']
  for tile in TILES:
    if n % (32*tile[1]): continue
    port.GEMM_TILE = tile
    def run_new(x, *w):
      return Tensor.realize(*[port.native_gemm(x, w[i % 4], xs, xs, r if res else None) for i in range(REPS)])
    g = TinyJit(run_new)
    try:
      g(x, *ws); g(x, *ws)
      med = float(timeit(lambda: g(x, *ws), n=5).split()[1])/REPS
      line.append(f'{tile} {med*1e3:.0f} us {2*rows*n*k/med/1e9:.0f} TF')
    except Exception as e:
      line.append(f'{tile} failed {type(e).__name__}')
  print(' | '.join(line), flush=True)
port.GEMM_TILE = (2, 4)
