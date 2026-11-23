import contextlib
from tinygrad import Device
m = rt.model
smu = Device['AMD'].iface.dev_impl.smu; sm = smu.smu_mod
def set_softmin(mhz):
  with rt.lock:
    try:
      smu._send_msg(sm.PPSMC_MSG_SetSoftMinByFreq, (sm.PPCLK_GFXCLK << 16) | mhz, timeout=200); return 'ok'
    except TimeoutError: return 'timeout'
def after_idle(name, idle, runs=5):
  s, a = ST[name], SA[name]
  m.synchronize(); time.sleep(idle); s.zero()
  for _ in range(runs): s(*a)
  cnt, st = s.read(m.synchronize); per = len(s.kernel_names())
  d = np.diff(st.reshape(runs, per), axis=1).sum(axis=1)/1e5
  return np.round(d, 2).tolist()
try:
  for mhz in (0, 2300, 0):
    r = set_softmin(mhz); time.sleep(0.5)
    print(f'soft min {mhz} ({r}): traj 2-step totals after 0.3 s idle, back-to-back: {after_idle("traj", 0.3)}; again {after_idle("traj", 0.3)}; power {rt.gpu_metrics()["power_w"]} W')
finally:
  print('revert', set_softmin(0))
