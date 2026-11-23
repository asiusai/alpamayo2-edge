# Live pipeline (prime + predict on the clock, early prefill parts off the clock): stage medians and GPU kernels on the clock.
import collections, re, sys
sys.path.insert(0, '/home/john/alpamayo2-edge/pc/tools')
import bench, samples
from tinygrad import Device
from tinygrad.device import Compiled, ProfileGraphEvent
d = Device['AMD']
SLOTS = globals().get('SLOTS', '/home/john/alpamayo2-edge/pc/data/slots/00000091--c1af5805a9--0.npz')
it = bench.slot_stream(SLOTS)
def strip(s): return {k: v for k, v in s.items() if k not in ('frame_objects', 'anchors')}
def oncall(s, sync):
  rt.stage_sync = sync
  t = time.perf_counter(); rt.prime(s['anchors'], time.monotonic_ns()+600_000_000_000, wait=sync); tp = (time.perf_counter()-t)*1e3
  p = rt.predict(strip(s)); rt.synchronize(); return (time.perf_counter()-t)*1e3, tp, p
pending = next(it)
for _ in range(40):
  s, pending = pending, next(it); oncall(s, False); rt.prefill_ahead(pending['frame_objects'][:4]); rt.synchronize()
stages, totals, primes = [], [], []
for _ in range(20):
  s, pending = pending, next(it); tot, tp, p = oncall(s, True); stages.append(p['timings_ms']); totals.append(tot); primes.append(tp)
  rt.prefill_ahead(pending['frame_objects'][:4]); rt.synchronize()
print('synchronized: total', round(float(np.median(totals)), 1), 'prime', round(float(np.median(primes)), 1), {k: round(float(np.median([x[k] for x in stages])), 1) for k in stages[0]})
tt = []
for _ in range(20):
  s, pending = pending, next(it); tot, tp, p = oncall(s, False); tt.append(tot); rt.prefill_ahead(pending['frame_objects'][:4]); rt.synchronize()
print('unsynchronized total p50', round(float(np.median(tt)), 1), 'mean', round(float(np.mean(tt)), 1))
if globals().get('PROFILE_KERNELS', True):
  t, cnt = collections.defaultdict(float), collections.Counter(); n = 12
  for _ in range(n):
    s, pending = pending, next(it)
    d.synchronize(); d.collect_prof(); Compiled.profile_events.clear()
    oncall(s, False); d.synchronize(); d.collect_prof()
    for ev in Compiled.profile_events:
      if isinstance(ev, ProfileGraphEvent):
        for e in ev.ents:
          name = re.sub(r'\x1b\[[0-9;]*m', '', e.name if isinstance(e.name, str) else getattr(e.name, 'display_name', str(e.name)))
          t[name] += float(ev.sigs[e.en_id]-ev.sigs[e.st_id]); cnt[name] += 1
    rt.prefill_ahead(pending['frame_objects'][:4]); rt.synchronize()
  print(f'GPU kernel time on the clock: {sum(t.values())/n/1000:.1f} ms per call')
  for k, us in sorted(t.items(), key=lambda x: -x[1])[:25]:
    print(f'{us/n/1000:7.2f} ms {cnt[k]/n:6.1f}/call {us/cnt[k]:8.1f} us  {k[:70]}')
