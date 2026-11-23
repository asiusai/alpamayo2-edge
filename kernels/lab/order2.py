import hashlib, re
from tinygrad.uop.ops import Ops
import openpilot.selfdrive.alpamayo.selected.vision_pair as vpi
ANSI = re.compile(r'\x1b\[[0-9;]*m')
keys = []
for call in ST['vision'].jit.captured._linear.src:
  c = call.src[0] if call.op is Ops.AFTER else call
  aux = getattr(c.arg, 'aux', None)
  if aux is not None and hasattr(aux, 'kernels'):
    keys += [k[4] for k in aux.kernels if ANSI.sub('', str(k[1])) != 'lab_stamp']
print(len(keys), type(keys[0]))
old = hashlib.sha256(b''.join(keys)).hexdigest()
new = hashlib.sha256(b''.join(bytes.fromhex(remap.get(k.hex(), k.hex())) for k in keys)).hexdigest()
print('changed calls', sum(remap.get(k.hex(), k.hex()) != k.hex() for k in keys), 'remap changed', {a[:12]: b[:12] for a, b in remap.items() if a != b})
print('ORDERED_PROGRAMS current', old, old == vpi.ORDERED_PROGRAMS)
print('ORDERED_PROGRAMS new', new)
