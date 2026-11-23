from tinygrad import TinyJit, Device
from tinygrad.helpers import PROFILE
from tinygrad.device import Compiled
dev = Device[Device.DEFAULT]
print('PROFILE', PROFILE.value, 'divider', dev.timestamp_divider, 'is_usb', getattr(dev,'is_usb',None))
if 'PROFD' in globals():
  ev = PROFD['ev']
  print(len(ev), ev[:6]); print(ev[-4:])
PROFILE.value = 1
def chain(x):
  for _ in range(6):
    mx = x.float().abs().max().realize()
    x = (x.float()/(mx+1)).cast(dtypes.bfloat16).contiguous().realize()
  return x
j = TinyJit(chain)
x = Tensor.randn(1344, 2048, dtype=dtypes.bfloat16).contiguous().realize()
for _ in range(3): j(x)
rt.model.synchronize()
Compiled.profile_events.clear()
for r in range(3):
  j(x); rt.model.synchronize()
  print('prof_ents', len(dev.prof_ents))
  dev.collect_prof() if hasattr(dev,'collect_prof') else None
  for e in Compiled.profile_events:
    if type(e).__name__ == 'ProfileGraphEvent':
      print(len(e.ents), [str(s) for s in e.sigs[:40]])
      for ent in e.ents[:12]: print(ent.name, ent.st_id, ent.en_id, float(e.sigs[ent.en_id]-e.sigs[ent.st_id]))
  Compiled.profile_events.clear()
