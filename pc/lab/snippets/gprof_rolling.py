# Rolling mode (2 fresh images per call): stage medians (synchronized) and per-kernel GPU time from graph timestamps.
import collections, sys
sys.path.insert(0, '/home/john/alpamayo2-edge/pc/tools')
import bench, samples
from tinygrad import Device
from tinygrad.device import Compiled, ProfileGraphEvent
d = Device['AMD']
MODE = globals().get('MODE', 'rolling')
data = [samples.load(p) for p in samples.karel_samples()]
it = bench.stream(MODE, data)
rt.stage_sync = True
for _ in range(20): rt.predict(next(it))
stages, total = [], []
for _ in range(12):
  t = time.perf_counter(); p = rt.predict(next(it)); total.append((time.perf_counter()-t)*1e3); stages.append(p['timings_ms'])
print('synchronized total median', round(float(np.median(total)), 1), {k: round(float(np.median([s[k] for s in stages])), 1) for k in stages[0]})
rt.stage_sync = False
t0 = time.perf_counter(); n = 12
for _ in range(n): rt.predict(next(it))
rt.synchronize(); print('unsynchronized per call', round((time.perf_counter()-t0)*1e3/n, 1), 'ms')
d.synchronize(); d.collect_prof(); Compiled.profile_events.clear()
for _ in range(n): rt.predict(next(it))
d.synchronize(); d.collect_prof()
t, cnt = collections.defaultdict(float), collections.Counter()
for ev in Compiled.profile_events:
  if isinstance(ev, ProfileGraphEvent):
    for e in ev.ents:
      name = e.name if isinstance(e.name, str) else getattr(e.name, 'display_name', str(e.name))
      import re; name = re.sub(r'\x1b\[[0-9;]*m', '', name)
      t[name] += float(ev.sigs[e.en_id]-ev.sigs[e.st_id]); cnt[name] += 1
print(f'GPU kernel time per call: {sum(t.values())/n/1000:.1f} ms')
for k, us in sorted(t.items(), key=lambda x: -x[1])[:28]:
  print(f'{us/n/1000:7.2f} ms {cnt[k]/n:6.1f}/call {us/cnt[k]:8.1f} us  {k[:70]}')
