kl = lab_import('kloop')
S = '/data/alpamayo-bench/lab/src/'
m = rt.model
calls = []
for scale in (6.0, 40.0, 0.01):
  for bufs, fxn, empties in REC['vision_attention_max'][:3]:
    q = (bufs[1].float()*scale).cast(dtypes.bfloat16).contiguous().realize()
    calls.append(((bufs[0], q, bufs[2], bufs[3], bufs[4]), fxn, empties))
variants = {v: (open(S+v+'.cpp').read(), open(S+v+'.hsaco', 'rb').read()) for v in ('vattn_direct_nowpe', 'vattn_d4b')}
print('synthetic attention inputs: query scaled by 6, 40 and 0.01')
kl.ab(calls, variants, kind='vision_attention_max', sync=m.synchronize, rounds=2)
