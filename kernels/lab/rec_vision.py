# Record real inputs of one vision-pair kernel kind (first LAYERS FP8 layers) for the short loop.
kl = lab_import('kloop'); lab_import('stamplib')
from tinygrad.helpers import PROFILE
PROFILE.value = 0
m = rt.model
REC = globals().setdefault('REC', {})
KINDS = globals().get('KINDS', ('vision_fc1_gelu_max',))
pairs = ARGS['pairs']
for kind in KINDS:
  t0 = time.monotonic()
  REC[kind] = kl.record(kind, lambda: Tensor.realize(*m.encode_group(*pairs)), limit=globals().get('LIMIT', 25))
  print(kind, len(REC[kind]), 'calls recorded in', round(time.monotonic()-t0, 1), 's; buffers', [(tuple(t.shape), t.dtype.name) for t in REC[kind][0][0]], 'outputs', REC[kind][0][2])
