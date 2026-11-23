print(sorted(k for k in globals() if not k.startswith('__')))
m = rt.model
print('traj jits', list(m.trajectory_jits.keys()))
print('prefill jits', list(rt.prefill.jits.keys()), list(rt.prefill.part_jits.keys()))
print('encode jits', list(m.encode_jits.keys())[:5])
print('PARITY', globals().get('PARITY'))
import os
print(os.environ.get('PROFILE'), os.environ.get('DEBUG'))
from tinygrad.helpers import PROFILE
print('PROFILE', PROFILE.value)
print(type(rt.vision), [a for a in dir(rt.vision) if not a.startswith('__')])
print([a for a in dir(rt) if not a.startswith('__')])
