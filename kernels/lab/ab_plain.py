kl = lab_import('kloop'); lab_import('stamplib')
SRC = '/data/alpamayo-bench/lab/src/'
m = rt.model
KIND = 'vision_attention'
if KIND not in REC:
  REC[KIND] = kl.record(KIND, lambda: Tensor.realize(*m.encode_group(*ARGS['pairs'])), limit=4)
  print(KIND, len(REC[KIND]), 'calls recorded; buffers', [(tuple(t.shape), t.dtype.name) for t in REC[KIND][0][0]], REC[KIND][0][2])
variants = {v: (open(SRC+v+'.cpp').read(), open(SRC+v+'.hsaco', 'rb').read()) for v in ('vattn_plain_direct',)}
kl.ab(REC[KIND], variants, kind=KIND, sync=m.synchronize, rounds=6)
