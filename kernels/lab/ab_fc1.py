kl = lab_import('kloop')
SRC = '/data/alpamayo-bench/lab/src/'
VARS = globals().get('VARS', ('fc1_nosmall', 'fc1_pf2', 'fc1_nosmall_pf2'))
KIND = globals().get('KIND', 'vision_fc1_gelu_max')
variants = {v: (open(SRC+v+'.cpp').read(), open(SRC+v+'.hsaco', 'rb').read()) for v in VARS}
RES = kl.ab(REC[KIND], variants, kind=KIND, sync=rt.model.synchronize)
