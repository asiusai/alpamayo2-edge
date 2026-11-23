kl = lab_import('kloop'); lab_import('stamplib')
from tinygrad.helpers import PROFILE
from tinygrad.uop.ops import Ops, UOp, KernelInfo
PROFILE.value = 0
SRC = '/data/alpamayo-bench/lab/src/'
# 1. exhaustive GELU equivalence on the GPU
BIN = open(SRC+'gelu_check.hsaco','rb').read(); TXT = open(SRC+'gelu_check.cpp').read()
def gprog(out, dep):
  sink = UOp.sink(out.flatten()[0].store(dep.flatten()[0].load()), UOp.special(256,'gidx0'), UOp.special(256,'lidx0'), arg=KernelInfo(name='gelu_check'))
  return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.LINEAR, src=tuple(sink.toposort())), UOp(Ops.SOURCE, arg=TXT), UOp(Ops.BINARY, arg=BIN)))
dep = Tensor.zeros(4, dtype=dtypes.uint32).contiguous().realize()
x = Tensor.custom_kernel(Tensor.empty(65536, dtype=dtypes.uint32), dep, fxn=gprog)[0].realize().numpy()
print('GELU old vs new over all 65536 BF16 inputs: differing results', int((x != 0).sum()))
# 2. prefill fused MLP (final part) loop unrolling
if 'mlp' not in REC:
  x_, cos, sin, _, length = ARGS['prefill']
  t0 = time.monotonic()
  REC['mlp'] = kl.record('mlp', lambda: rt.prefill.language(x_, cos, sin, SA['prefill'][3], 32), limit=14)
  print('recorded', len(REC['mlp']), 'mlp calls in', round(time.monotonic()-t0, 1), 's', [(tuple(t.shape), t.dtype.name) for t in REC['mlp'][0][0]], REC['mlp'][0][2])
variants = {v: (open(SRC+v+'.cpp').read(), open(SRC+v+'.hsaco', 'rb').read()) for v in ('mlp_unroll1', 'mlp_unroll4')}
RES_MLP = kl.ab(REC['mlp'], variants, kind='mlp', sync=rt.model.synchronize)
