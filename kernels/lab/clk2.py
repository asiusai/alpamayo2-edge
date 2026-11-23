# Soft minimum GFXCLK experiment (same SMU message tinygrad sends at init). PPT stays 100 W; revert with SOFTMIN = 0.
import contextlib
from tinygrad import Device
m = rt.model
smu = Device['AMD'].iface.dev_impl.smu; sm = smu.smu_mod
def set_softmin(mhz):
  with rt.lock:
    with contextlib.suppress(TimeoutError): smu._send_msg(sm.PPSMC_MSG_SetSoftMinByFreq, (sm.PPCLK_GFXCLK << 16) | mhz, timeout=20)
def metrics():
  g = rt.gpu_metrics(); return g
def sustained(name, seconds=1.5):
  s, a = ST[name], SA[name]
  t0 = time.perf_counter(); n = 0; peak = 0
  while time.perf_counter()-t0 < seconds:
    s(*a); m.synchronize(); n += 1; s.zero()
    if n % 8 == 0: peak = max(peak, metrics()['power_w'])
  g = metrics()
  return f"{g['power_w']} W (peak sample {max(peak, g['power_w'])}) {g['gfx_mhz']} MHz, {seconds/n*1e3:.1f} ms/call"
SOFTMIN = globals().get('SOFTMIN', 1800)
set_softmin(SOFTMIN)
time.sleep(1.0)
g = metrics(); print(f'soft min {SOFTMIN} MHz: idle {g["power_w"]} W {g["gfx_mhz"]} MHz {g["hotspot_c"]} C')
for name in ('prefill', 'vision', 'traj'):
  print('  sustained', name, sustained(name))
  if metrics()['power_w'] > 103:
    set_softmin(0); print('POWER ABOVE LIMIT: reverted soft min to 0'); break
time.sleep(0.5)
