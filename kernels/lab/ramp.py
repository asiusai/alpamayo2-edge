names, us = INV['traj']
s = ST['traj']; m = rt.model; actions, ac, ass, ln, of = TA
n = np.array(names); u = np.array(us)
idx = np.where(n == 'alpamayo_expert_attention')[0]
print('attention us by layer, step1:', np.round(u[idx[:28]]).astype(int).tolist())
print('attention us by layer, step2:', np.round(u[idx[28:]]).astype(int).tolist())
half = len(u)//2
print('step1 total', u[:half].sum()/1000, 'step2 total', u[half:].sum()/1000)
# back-to-back runs without readback gaps: run the stamped graph 4 times, then read the last
for gap in (0, 0.02, 0.2):
  s.zero(); m.synchronize()
  for _ in range(4): s(actions, ac, ass, ln, of)
  if gap: m.synchronize(); time.sleep(gap); s.zero(); s(actions, ac, ass, ln, of)
  cnt, st = s.read(m.synchronize)
  per = len(names)+1
  runs = st[:cnt].reshape(-1, per)
  d = np.diff(runs, axis=1)/100
  print(f'gap {gap}: run totals ms', np.round(d.sum(axis=1)/1000, 3).tolist(), 'attention mean us', [round(float(x[idx].mean()),1) for x in d])
