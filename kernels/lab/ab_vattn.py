KINDS = tuple(k for k in ('vision_attention_max', 'vision_fc1_gelu_max') if k not in globals().get('REC', {}))
exec(open('/data/alpamayo-bench/lab/src/rec_vision.py').read())
KIND = 'vision_attention_max'
VARS = globals().get('VVARS', ('vattn_direct', 'vattn_direct_nowpe'))
exec(open('/data/alpamayo-bench/lab/src/ab_fc1.py').read())
