# 48 rebuilt Karel drive inputs (10 fresh images each), two rounds; saves round-2 outputs and prints stage medians.
import json, samples
from pathlib import Path
LABEL = globals().get('LABEL', 'run')
rt.stage_sync = True
S = [(p, samples.load(p)) for p in samples.karel_samples()]
res, stages, total = [], [], []
for rnd in range(2):
  for p, s in S:
    t = time.perf_counter(); out = rt.predict(s); rt.synchronize(); ms = (time.perf_counter()-t)*1e3
    if rnd:
      total.append(ms); stages.append(out['timings_ms'])
      res.append(dict(sample=str(p), ms=ms, xyz=out['xyz'], rot=out['rot'], tokens=out['generated_token_ids'], reasoning=out['reasoning'], timings=out['timings_ms']))
Path(f'/home/john/alpamayo2-edge/pc/parity/run-{LABEL}.json').write_text(json.dumps(res))
print('total median', round(float(np.median(total)), 1), 'mean', round(float(np.mean(total)), 1))
for k in stages[0]:
  print(f'{k:18s} {np.median([s[k] for s in stages]):8.1f}')
