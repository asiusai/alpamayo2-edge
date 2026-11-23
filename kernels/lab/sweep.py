# Sweep define variants of a frozen vision GEMM kind. Set SWEEP = (kind, json name) first.
kl = lab_import('kloop'); lab_import('stamplib')
import json, gc
from tinygrad.helpers import PROFILE
PROFILE.value = 0
SRC = '/data/alpamayo-bench/lab/src/'
m = rt.model
kind, setname = SWEEP
if kind not in REC:
  REC[kind] = kl.record(kind, lambda: Tensor.realize(*m.encode_group(*ARGS['pairs'])), limit=12)
  print(kind, len(REC[kind]), 'calls recorded; outputs', REC[kind][0][2])
spec = json.load(open(SRC+setname+'.json'))
names = list(spec)
for i in range(0, len(names), 5):
  variants = {n: {'pair': (open(SRC+n+'.cpp').read(), open(SRC+n+'.hsaco', 'rb').read()), 'grid': spec[n]['grid'], 'local': spec[n]['local']} for n in names[i:i+5]}
  kl.ab(REC[kind], variants, kind=kind, sync=m.synchronize, rounds=4)
