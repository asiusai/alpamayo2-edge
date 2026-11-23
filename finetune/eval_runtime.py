"""Car-path evaluation: the selected tinygrad runtime (FP8, own greedy reasoning, 1 Euler step) on one GPU.

Usage (pc/env.sh): eval_runtime.py GPU OUT.npz SAMPLE_DIR [--fp8-cache DIR] [--steps 1-3] [--nav TEXT] [--limit N]
A fine-tuned FP8 cache (artifacts.json `fp8_finetunes`) runs with the base BF16 checkpoint, as in the car.
Writes OUT.npz ('<stem>' -> (64,3) xyz) and OUT.json (reasoning text per stem, mean timings). The loader thread starts
before the GPU opens; no process is created after that. A separate graph cache per weight set.
"""
import argparse
import json
import queue
import threading
import time
from pathlib import Path

import numpy as np

import alpamayo_pc

KEYS = ('image_frames', 'camera_indices', 'absolute_timestamps', 'ego_history_xyz', 'ego_history_rot')
ap = argparse.ArgumentParser()
ap.add_argument('gpu', type=int); ap.add_argument('out', type=Path); ap.add_argument('samples', type=Path)
ap.add_argument('--fp8-cache', type=Path); ap.add_argument('--limit', type=int); ap.add_argument('--steps', type=int, default=1); ap.add_argument('--nav', default=None, help='replace every sample\'s navigation text')
a = ap.parse_args()
files = sorted(f for f in a.samples.glob('*.npz') if not f.name.endswith(('.frozen.npz', '.tmp.npz')))[:a.limit]
loaded = queue.Queue(maxsize=4)


def loader():
  for f in files:
    with np.load(f) as z:
      s = {k: z[k] for k in KEYS}
      s['nav_text'] = [a.nav or str(z['nav_text'])]
    loaded.put((f.stem, s))
  loaded.put(None)


threading.Thread(target=loader, daemon=True).start()
overrides = {'diffusion_steps': a.steps}
if a.fp8_cache:
  overrides.update(fp8_cache=a.fp8_cache, cache_dir=alpamayo_pc.PC/'cache'/f'{a.fp8_cache.name}-gpu{a.gpu}')
rt = alpamayo_pc.load(a.gpu, **overrides)
preds, text, times = {}, {}, []
t0 = time.time()
for n, (stem, s) in enumerate(iter(loaded.get, None)):
  t = time.perf_counter()
  p = rt.predict(s)
  times.append((time.perf_counter()-t)*1e3)
  preds[stem] = np.asarray(p['xyz'], dtype=np.float32)[0]
  text[stem] = p['reasoning']
  if n % 50 == 0:
    print(f'{n+1}/{len(files)} {(time.time()-t0)/(n+1):.2f} s/sample {p["reasoning"][:60]!r}', flush=True)
np.savez(a.out, **preds)
a.out.with_suffix('.json').write_text(json.dumps({'reasoning': text, 'median_ms': float(np.median(times[10:] or times))}))
print('done', len(preds), flush=True)
