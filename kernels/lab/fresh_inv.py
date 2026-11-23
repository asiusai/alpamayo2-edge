WHICH = ('traj', 'prefill', 'part0', 'part1', 'vision')
exec(open('/data/alpamayo-bench/lab/src/inv_all.py').read())
def after_idle(name, idle, runs=5):
  s, a = ST[name], SA[name]
  m.synchronize(); time.sleep(idle); s.zero()
  for _ in range(runs): s(*a)
  cnt, st = s.read(m.synchronize); per = len(s.kernel_names())
  d = np.diff(st.reshape(runs, per), axis=1).sum(axis=1)/1e5
  return np.round(d, 2).tolist()
print('SIGNATURE fresh lab: traj after idle', after_idle('traj', 0.3), after_idle('traj', 0.3), 'part0', after_idle('part0', 0.3, 3), 'vision', after_idle('vision', 0.3, 3))
CYCLES = 12
exec(open('/data/alpamayo-bench/lab/src/cycle.py').read().split("for k, per in (('vision'")[0])
