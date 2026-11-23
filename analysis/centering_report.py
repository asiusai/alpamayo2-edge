# Open-loop replay summary: how lane centring changes the tracked command on straights, curves and turns.
import sys, numpy as np
for f in sys.argv[1:]:
  d = np.load(f); a = d['a']; c = list(d['cols']); C = lambda n: a[:, c.index(n)]
  ok = np.isfinite(C('k_base')) & np.isfinite(C('k_cen')) & (C('v') > 3)
  kb, kc, off, head = C('k_base'), C('k_cen'), -C('c0'), C('head')  # off right+, head: lane heading rel. car right+
  ang = np.abs(C('angle'))
  dk = kc-kb  # right positive
  print('==', f.split('/')[-1][:23], 'frames', ok.sum())
  for name, m in (('straight |angle|<5', ok & (ang < 5) & (C('v') > 8)), ('bend 5-20 deg', ok & (ang >= 5) & (ang < 20)),
                  ('turn >=20 deg', ok & (ang >= 20)), ('blinker', ok & (C('blink') > 0))):
    if m.sum() < 10: print(f'  {name:20s} n {m.sum()}'); continue
    changed = np.mean(np.abs(dk[m]) > 2e-4)
    rel = np.median(np.abs(dk[m]))/max(np.median(np.abs(kb[m])), 1e-6)
    # Centring should steer toward the centre: car right of centre (off>0) -> more left (negative right+ curvature)
    X = np.c_[off[m], head[m], np.ones(m.sum())]
    g = np.linalg.lstsq(X, dk[m], rcond=None)[0]
    print(f'  {name:20s} n {m.sum()/20:5.0f}s changed>2e-4 {changed*100:3.0f}%  median|dk|/median|k| {rel:.2f}  dk = {g[0]*1e3:+.2f}e-3*offset {g[1]*1e3:+.2f}e-3*head {g[2]*1e4:+.2f}e-4')
  # turns: largest absolute differences
  m = ok & (ang >= 20)
  if m.sum():
    print(f'  turns: p95 |dk| {np.percentile(np.abs(dk[m]),95):.4f}  p95 |k| {np.percentile(np.abs(kb[m]),95):.4f}')
