import gc
from tinygrad.helpers import GlobalCounters
print('before', GlobalCounters.mem_used/1e9, 'GB; gc enabled', gc.isenabled())
for k in ('RES', 'variants'):
  globals().pop(k, None)
print('collected', gc.collect(), gc.collect())
print('after', GlobalCounters.mem_used/1e9, 'GB')
