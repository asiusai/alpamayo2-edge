# ORDERED_PROGRAMS digest of the vision pair capture for the swapped objects (remap comes from refreeze_objects.py).
import hashlib
from tinygrad import TinyJit, Device
import openpilot.selfdrive.alpamayo.selected.vision_pair as vpi
from openpilot.selfdrive.alpamayo.selected.vision_pair.programs import PairPrograms
keys = []
class Rec(vpi.PairCapture):
  def encode_group(self, *pairs):
    parent = vpi.capturing[0]
    def compile_group(linear):
      compiled = self.compiler(linear)
      for call in compiled.src: keys.append(call.body.key)
      return compiled
    vpi.capturing[0] = vpi.CompileAtRealize(parent, compile_group)
    try: return vpi.SelectedModel.encode_group(self, *pairs)
    finally: vpi.capturing[0] = parent
cap = Rec(rt.vision.pair_weights.model, PairPrograms(Device[Device.DEFAULT].renderer))
jit = TinyJit(cap.encode_group); jit.cnt = 1
jit(*ARGS['pairs']); rt.model.synchronize()
old = hashlib.sha256(b''.join(keys)).hexdigest()
new = hashlib.sha256(b''.join(bytes.fromhex(remap.get(k.hex(), k.hex())) for k in keys)).hexdigest()
print(len(keys), 'calls; changed calls', sum(k.hex() in remap and remap[k.hex()] != k.hex() for k in keys))
print('ORDERED_PROGRAMS current', old, old == vpi.ORDERED_PROGRAMS)
print('ORDERED_PROGRAMS new', new)
del jit, cap
import gc; gc.collect()
