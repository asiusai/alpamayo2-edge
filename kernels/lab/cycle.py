# Production-like duty cycle with in-graph stamps: [vision, prefill, 10 expert steps] | idle | part0 | idle | vision, part1 | idle.
sl = lab_import("stamplib")
m = rt.model
PERIOD = globals().get('PERIOD', 0.300)
CYCLES = globals().get('CYCLES', 12)
def read(name, runs):
  s = ST[name]; cnt, st = s.read(m.synchronize)
  per = len(s.kernel_names())
  assert cnt == runs*per, (name, cnt, runs, per)
  return np.diff(st.reshape(runs, per), axis=1)/100.0, st.reshape(runs, per)
acc = {k: [] for k in ('vision', 'prefill', 'traj', 'part0', 'vision2', 'part1')}
span = []
for n in ST: ST[n].zero()
m.synchronize()
for c in range(CYCLES):
  t0 = time.perf_counter()
  ST['vision'](*SA['vision']); ST['prefill'](*SA['prefill'])
  for _ in range(5): ST['traj'](*SA['traj'])
  m.synchronize(); t1 = time.perf_counter()
  v, vs = read('vision', 1); p, ps = read('prefill', 1); t, ts_ = read('traj', 5)
  span.append(((ts_[-1, -1]-vs[0, 0])/1e5, (t1-t0)*1e3))
  acc['vision'].append(v[0]); acc['prefill'].append(p[0]); acc['traj'].append(t.reshape(-1))
  ST['vision'].zero(); 
  time.sleep(0.015)
  ST['part0'].zero(); ST['part0'](*SA['part0']); d, _ = read('part0', 1); acc['part0'].append(d[0])
  time.sleep(0.05)
  ST['part1'].zero(); ST['vision'](*SA['vision']); ST['part1'](*SA['part1'])
  v, _ = read('vision', 1); d, _ = read('part1', 1); acc['vision2'].append(v[0]); acc['part1'].append(d[0])
  for n in ('vision', 'prefill', 'traj'): ST[n].zero()
  m.synchronize()
  rest = PERIOD-(time.perf_counter()-t0)
  if rest > 0: time.sleep(rest)
  if c == CYCLES-1: print('gpu metrics at end', rt.gpu_metrics(), 'cycle busy ms', round((PERIOD-rest)*1e3))
CYC = {k: np.median(np.array(v[3:]), axis=0) for k, v in acc.items()}
print(f'period {PERIOD*1e3:.0f} ms; critical path GPU span / wall ms (median): {np.median([s[0] for s in span[3:]]):.1f} / {np.median([s[1] for s in span[3:]]):.1f}')
tot = {k: v.sum()/1000 for k, v in CYC.items()}
print('stage GPU ms in cycle:', {k: round(v, 2) for k, v in tot.items()}, '| traj per step', round(tot['traj']/10, 3))
print('stage GPU ms warm back-to-back:', {k: round(INV[k][1].sum()/1000*(5 if k == 'traj' else 1), 2) for k in INV})
for k, per in (('vision', 1), ('prefill', 1), ('traj', 10), ('part0', 1), ('part1', 1)):
  names = ST[k].kernel_names()[1:]
  if k == 'traj': names = (names + ['(first)'])*5; names = names[:len(CYC[k])]
  sl.table(names, CYC[k], top=12, per=per, label=f'{k} in cycle' + (' (per step)' if per > 1 else ''))
