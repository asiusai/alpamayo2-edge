# Lateral oscillation of the car in its lane while engaged: spread, dominant period and amplitude per route.
# Usage: oscillation.py LANE_TABLE.npz [ROUTE_PREFIX ...]
import sys
import numpy as np
tab = np.load(sys.argv[1]); a, cols, segs = tab['a'], list(tab['cols']), tab['seg']
C = lambda n: a[:, cols.index(n)]
routes = np.array([str(s).rsplit('--', 1)[0][:8] for s in segs])
good = (C('v') > 8) & (C('prob') > .6) & (C('width') > 2.5) & (C('width') < 4.5) & (C('blink') == 0)
off, t, v, lat, ang, pressed = -C('c0'), C('t_ns')/1e9, C('v'), C('lat') > 0, C('angle'), C('pressed') > 0
want = sys.argv[2:] or sorted(set(routes))
print('route  windows  minutes  speed  offset mean / sd   detrended sd   period s (median, IQR)   peak-to-peak m   zero crossings per min   steer sd deg   hands on')
for r in want:
  k = np.flatnonzero((routes == r) & good & lat)
  if len(k) < 600: continue
  k = k[np.argsort(t[k])]
  # contiguous engaged windows (gaps under 0.5 s), at least 30 s long
  cut = np.flatnonzero(np.diff(t[k]) > .5)
  wins = [w for w in np.split(k, cut+1) if t[w[-1]]-t[w[0]] >= 30]
  if not wins: continue
  per, p2p, zc, dsd, n = [], [], [], [], 0
  for w in wins:
    tt, y = t[w], off[w]
    g = np.arange(tt[0], tt[-1], .1); yi = np.interp(g, tt, y)
    yi = yi-np.convolve(yi, np.ones(200)/200, 'same')  # remove slow drift over 20 s
    yi = yi[100:-100]
    if len(yi) < 200: continue
    n += len(yi)
    dsd.append(yi.std())
    f = np.fft.rfftfreq(len(yi), .1); P = np.abs(np.fft.rfft(yi*np.hanning(len(yi))))**2
    band = (f > 1/20) & (f < 1/1.5)
    per.append(1/f[band][np.argmax(P[band])])
    s = np.sign(yi); crossings = np.sum(s[1:]*s[:-1] < 0); zc.append(crossings/(len(yi)*.1/60))
    # peak-to-peak per cycle: range inside sliding windows of one dominant period
    m = max(int(per[-1]/.1), 10); p2p.append(np.median([np.ptp(yi[i:i+m]) for i in range(0, len(yi)-m, m)]))
  kk = np.concatenate(wins)
  print(f'{r} {len(wins):5d} {n*.1/60:8.1f} {v[kk].mean():6.0f}   {off[kk].mean():+.2f} / {off[kk].std():.2f}      {np.mean(dsd):.3f}        {np.median(per):5.1f} ({np.percentile(per,25):.1f}-{np.percentile(per,75):.1f})          {np.median(p2p):.2f}              {np.median(zc):5.1f}             {np.std(ang[kk]):.2f}        {pressed[kk].mean():.2f}')
