# Merge noise_draws.py shards on the drive-test straights and add the per-sample mean of the draws.
# Usage: lanes_noise_merge.py PREFIX_G0 PREFIX_G1 OUT_PREFIX DRAWS
import sys, numpy as np
a, b, out, n = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
names = ['zero']+[f'd{k}' for k in range(n)]
m = {k: {**dict(np.load(f'{a}-{k}.npz')), **dict(np.load(f'{b}-{k}.npz'))} for k in names}
for k, v in m.items():
  np.savez(f'{out}-{k}.npz', **v)
np.savez(f'{out}-mean.npz', **{s: np.mean([m[f'd{k}'][s] for k in range(n)], 0) for s in m['d0']})
np.savez(f'{out}-mean4.npz', **{s: np.mean([m[f'd{k}'][s] for k in range(4)], 0) for s in m['d0']})
sp = np.array([np.std([m[f'd{k}'][s][[19, 39, 63], 1] for k in range(n)], 0) for s in m['d0']])
print('samples', len(m['d0']), 'draw spread (sd of lateral, m) at 2 / 4 / 6.4 s:', np.round(np.median(sp, 0), 3))
