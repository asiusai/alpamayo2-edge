# SMU workload profile (DPM ramp behaviour). PPT stays 100 W. Reverts to the default profile bit at the end.
from tinygrad import Device
m = rt.model
smu = Device['AMD'].iface.dev_impl.smu; sm = smu.smu_mod
def workload(bit):
  with rt.lock:
    try:
      smu._send_msg(sm.PPSMC_MSG_SetWorkloadMask, 1 << bit, timeout=200); return 'ok'
    except TimeoutError: return 'timeout'
def after_idle(name, idle, runs=5):
  s, a = ST[name], SA[name]
  m.synchronize(); time.sleep(idle); s.zero()
  for _ in range(runs): s(*a)
  cnt, st = s.read(m.synchronize); per = len(s.kernel_names())
  d = np.diff(st.reshape(runs, per), axis=1).sum(axis=1)/1e5
  return np.round(d, 2).tolist()
def sustained(name, seconds=1.5):
  s, a = ST[name], SA[name]
  t0 = time.perf_counter(); n = 0; peak = 0
  while time.perf_counter()-t0 < seconds:
    s(*a); m.synchronize(); n += 1; s.zero()
    if n % 6 == 0: peak = max(peak, rt.gpu_metrics()['power_w'])
  g = rt.gpu_metrics()
  return f"{g['power_w']} W (peak sample {max(peak, g['power_w'])}) {g['gfx_mhz']} MHz {seconds/n*1e3:.1f} ms/call"
print('baseline (boot profile): traj after idle', after_idle('traj', 0.3), 'part0', after_idle('part0', 0.3, 3), 'vision', after_idle('vision', 0.3, 3))
try:
  for bit, label in ((sm.WORKLOAD_PPLIB_COMPUTE_BIT, 'compute'), (sm.WORKLOAD_PPLIB_DEFAULT_BIT, 'default')):
    r = workload(bit); time.sleep(0.5)
    print(f'profile {label} ({r}): traj after idle', after_idle('traj', 0.3), 'part0', after_idle('part0', 0.3, 3), 'vision', after_idle('vision', 0.3, 3))
    print('   sustained prefill', sustained('prefill'), '| idle', rt.gpu_metrics()['power_w'], 'W')
finally:
  print('revert to default profile', workload(sm.WORKLOAD_PPLIB_DEFAULT_BIT))
