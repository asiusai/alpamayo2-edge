# Summarise straights.py output: car offset, plan offset relative to lane centre, recovery, bias, human baseline.
import sys, numpy as np
for f in sys.argv[1:]:
  d = np.load(f); a = d['a']; c = list(d['cols']); C = lambda n: a[:, c.index(n)]
  v, lat, pr, prob = C('v'), C('lat') > 0, C('pressed') > 0, np.minimum(C('p1'), C('p2'))
  curv = np.abs(C('c40')-2*C('c20')+C('c0'))  # lane bend over 40 m
  good = (v > 8) & (prob > .6) & (C('width') > 2.5) & (C('width') < 4.5) & (C('blinkL') == 0) & (C('blinkR') == 0)
  straight = good & (curv < .5) & (np.abs(C('angle')) < 8)
  print(f'== {f.split("/")[-1]}  frames {len(a)}  engaged {lat.mean()*100:.0f}%')
  for name, m in (('engaged straight', straight & lat & ~pr), ('human straight', straight & ~lat), ('engaged all lanes', good & lat & ~pr)):
    if m.sum() < 20: print(name, 'n', m.sum()); continue
    off = -C('c0')[m]  # car offset from lane centre, right positive
    print(f'{name:18s} n {m.sum():5d} ({m.sum()/20:.0f} s) car offset mean {off.mean():+.2f} median|.| {np.median(np.abs(off)):.2f} p90|.| {np.percentile(np.abs(off),90):.2f} >0.5m {np.mean(np.abs(off)>.5)*100:.0f}% >0.8m {np.mean(np.abs(off)>.8)*100:.0f}%')
  m = straight & lat & ~pr & np.isfinite(C('p40'))
  if m.sum() > 50:
    off = -C('c0')[m]
    for x in (20, 40, 60):
      e = C(f'p{x}')[m]-C(f'c{x}')[m]  # plan relative to lane centre at x
      me = C(f'm{x}')[m]-C(f'c{x}')[m]  # comma plan relative to lane centre
      A = np.c_[off, np.ones_like(off)]
      ka, kb = np.linalg.lstsq(A, e, rcond=None)[0]
      ma, mb = np.linalg.lstsq(A, me, rcond=None)[0]
      print(f'  at {x} m: alpamayo plan-lane = {ka:+.2f}*offset {kb:+.2f} (keeps {ka*100:.0f}% of offset)  | comma plan = {ma:+.2f}*offset {mb:+.2f}')
    print(f'  plan age median {np.nanmedian(C("age")[m]):.2f} s')
  # drift episodes: engaged, unpressed, straight; offset change rate
  mm = straight & lat & ~pr
  t = C('t')
  idx = np.nonzero(mm)[0]
  if len(idx) > 40:
    off = -C('c0')
    rates = []
    for i in idx[::10]:
      j = np.searchsorted(t, t[i]+2.)
      if j < len(t) and mm[j] and np.all(mm[i:j]):
        rates.append((off[i], (off[j]-off[i])/2.))
    r = np.array(rates)
    if len(r) > 10:
      k = np.polyfit(r[:, 0], r[:, 1], 1)
      print(f'  closed loop: d(offset)/dt = {k[0]:+.3f}*offset {k[1]:+.3f} m/s (negative slope = returns to centre; time const {(-1/k[0]) if k[0]<0 else float("inf"):.1f} s), n {len(r)}')
