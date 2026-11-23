kl = lab_import('kloop')
SRC = '/data/alpamayo-bench/lab/src/'
for KIND, VARS in (('vision_fc1_gelu_max', ('fc1_nosmall',)), ('vision_attention_max', ('vattn_direct_nowpe',))):
  variants = {v: (open(SRC+v+'.cpp').read(), open(SRC+v+'.hsaco', 'rb').read()) for v in VARS}
  RES = kl.ab(REC[KIND], variants, kind=KIND, sync=rt.model.synchronize, rounds=10)
  for n, r in RES.items(): print('  ', n, 'warm per round', np.round(r['warm'], 1).tolist())
