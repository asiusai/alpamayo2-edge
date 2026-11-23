# GPU timeline per kernel inside captured graphs (needs a lab started with ALPAMAYO_PC_PROFILE=1).
import collections, samples
from tinygrad import Device
from tinygrad.device import Compiled, ProfileGraphEvent
d = Device['AMD']
ALL = [samples.load(p) for p in samples.turn_samples()]
S = ALL[:4]
FRESH = globals().get('FRESH', 10)
for s in S: rt.predict(s)
def run():
  d.synchronize(); d.collect_prof(); Compiled.profile_events.clear()
  for s in (ALL[4:7] if FRESH else S[1:]):
    rt.predict(s)
  d.synchronize(); d.collect_prof()
run()
t, cnt = collections.defaultdict(float), collections.Counter()
for ev in Compiled.profile_events:
  if isinstance(ev, ProfileGraphEvent):
    for e in ev.ents:
      name = e.name if isinstance(e.name, str) else getattr(e.name, 'display_name', str(e.name))
      t[name] += float(ev.sigs[e.en_id]-ev.sigs[e.st_id]); cnt[name] += 1
n = len(S)-1
print(f'GPU busy per predict: {sum(t.values())/n/1000:.1f} ms')
for k, us in sorted(t.items(), key=lambda x: -x[1])[:30]:
  print(f'{us/n/1000:7.2f} ms {cnt[k]/n:6.0f} {us/cnt[k]:8.1f} us  {k[:80]}')
