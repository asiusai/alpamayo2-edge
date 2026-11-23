import alpamayo_pc.port as port
from openpilot.selfdrive.alpamayo.selected import amd_gemm
x = amd_gemm.pack((Tensor.randn(1344, 2048)).cast(dtypes.fp8e4m3)).cast(dtypes.bfloat16).contiguous().realize()
w = amd_gemm.pack((Tensor.randn(2048, 2048)).cast(dtypes.fp8e4m3)).cast(dtypes.bfloat16).contiguous().realize()
s = Tensor([.01]).realize()
seen = []
orig = port._gemm_program
def spy(out, x, w, *a):
  seen.append(('native', x.dtype, x.dtype.name))
  return orig(out, x, w, *a)
port._gemm_program = spy
def dbg(out, x, w, xs, ws, *extra, residual=False):
  print('fxn args', out.shape, x.dtype, getattr(x.dtype, 'name', None), w.shape, port.NATIVE_GEMM, port._gemm_program is not None)
  return port.gemm_fxn(out, x, w, xs, ws, *extra, residual=residual)
Tensor.custom_kernel(Tensor.empty(1344, 2048, dtype=dtypes.bfloat16), x, w, s, s, fxn=dbg)[0].realize()
print(seen)
port._gemm_program = orig
