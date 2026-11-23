# GPU clock and power under sustained load and in the duty cycle (AM_PERF_LEVEL=auto, 100 W PPT).
m = rt.model
def metrics(): 
  g = rt.gpu_metrics(); return f"{g['power_w']} W {g['gfx_mhz']} MHz {g['hotspot_c']} C"
def sustained(name, seconds=2.0, reps=1):
  s, a = ST[name], SA[name]
  t0 = time.perf_counter(); n = 0
  while time.perf_counter()-t0 < seconds:
    for _ in range(reps): s(*a)
    m.synchronize(); n += reps
    s.zero()
  return metrics(), n
print('idle', metrics())
for name in ('prefill', 'vision', 'traj'):
  r = sustained(name); print('sustained', name, r)
  time.sleep(1.0); print('  after 1 s idle', metrics())
