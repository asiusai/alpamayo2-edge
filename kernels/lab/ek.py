import re
ANSI = re.compile(r'\x1b\[[0-9;]*m')
from tinygrad.uop.ops import Ops
lin = ST['prefill'].jit.captured._linear
seen = {}
for call in lin.src:
  c = call.src[0] if call.op is Ops.AFTER else call
  aux = getattr(c.arg, 'aux', None)
  if aux is None or not hasattr(aux, 'kernels'): continue
  ks = aux.kernels
  for i, k in enumerate(ks):
    name = ANSI.sub('', str(k[1]))
    if i < 60 and name != 'lab_stamp': print(i//2, name, k[2], 'bufs', k[5], 'io', k[6])
  break
