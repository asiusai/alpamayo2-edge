# Capture real call arguments of every stage into ARGS.
exec(open('/data/alpamayo-bench/lab/src/parity.py').read())
m = rt.model
ARGS = globals().setdefault('ARGS', {})
pcls = type(rt.prefill); o_call, o_part = pcls.__call__, pcls.part
def spy_call(self, *a, **k):
  ARGS['prefill'] = a; return o_call(self, *a, **k)
def spy_part(self, part, x, cos, sin):
  ARGS[f'part{part}'] = (x, cos, sin); return o_part(self, part, x, cos, sin)
o_traj = m.trajectory
def spy_traj(*a, **k):
  ARGS['traj'] = (a, k); return o_traj(*a, **k)
vcls = type(rt.vision); o_enc = vcls.encoder
def spy_enc(self, pairs):
  ARGS['pairs'] = list(pairs); return o_enc(self, pairs)
dcls = type(rt.decoder); o_ver, o_dec = dcls.verify, dcls.decode
pcls.__call__ = spy_call; pcls.part = spy_part; m.trajectory = spy_traj; vcls.encoder = spy_enc
try:
  run_sequence('prep', n=3)
finally:
  pcls.__call__ = o_call; pcls.part = o_part; del m.trajectory; vcls.encoder = o_enc
print({k: [getattr(t, 'shape', t) for t in (v if not isinstance(v, tuple) or k != 'traj' else v[0])] for k, v in ARGS.items()})
print(PARITY['prep'])
print('stats', rt.decoder.statistics)
print(rt.metrics.stages_ms)
