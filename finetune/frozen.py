"""Add the frozen car recipe's reasoning tokens and trajectory to each training sample (PC tinygrad runtime, one GPU).

Usage (pc/env.sh environment): frozen.py GPU [--nav TEXT] SAMPLE_DIR [SAMPLE_DIR ...]
With ALPAMAYO_SHARD=i/n this process takes every n-th remaining sample starting at i (one shard per GPU).
Writes <sample>.frozen.npz (reasoning_token_ids, baseline_xyz, baseline_rot); samples with one are skipped. No subprocess is started after the GPU opens.
--nav TEXT replaces every sample's navigation text and writes the side files under the mirror tree
FROZEN_ROOTS[TEXT]/<sample dir relative to the data root>/ (the sample and its meta stay unchanged).
"""
import json
import os
import queue
import sys
import threading
import time
from pathlib import Path

import numpy as np

import alpamayo_pc

DATA_ROOT = Path('/mnt/nvme-500/alpamayo-train')
FROZEN_ROOTS = {'Continue straight.': DATA_ROOT/'frozen-cs'}
KEYS = ('image_frames', 'camera_indices', 'absolute_timestamps', 'ego_history_xyz', 'ego_history_rot')


def side_path(f, nav):
  if nav is None:
    return f.with_suffix('.frozen.npz')
  return FROZEN_ROOTS[nav]/f.parent.resolve().relative_to(DATA_ROOT)/(f.stem + '.frozen.npz')


def main(gpu, dirs, nav=None):
  files = sorted(f for d in dirs for f in Path(d).glob('*.npz') if not f.name.endswith(('.frozen.npz', '.tmp.npz')))
  todo = []
  for f in files:
    if side_path(f, nav).exists() or not f.with_suffix('.json').exists():
      continue  # done, or still being written (the builder writes the .json after the .npz)
    if nav is not None:
      todo.append(f); continue
    with np.load(f) as z:
      if 'reasoning_token_ids' not in z.files:
        todo.append(f)
  shard, count = (int(v) for v in os.environ.get('ALPAMAYO_SHARD', '0/1').split('/'))
  todo = todo[shard::count]
  print(f'{len(todo)} of {len(files)} samples need the frozen pass in shard {shard}/{count}', flush=True)
  # The loader thread starts before the GPU opens (no thread or process is created after that) and decompresses ahead.
  loaded = queue.Queue(maxsize=4)
  def loader():
    for f in todo:
      with np.load(f) as z:
        loaded.put((f, {k: z[k] for k in z.files}))
    loaded.put(None)
  threading.Thread(target=loader, daemon=True).start()
  rt = alpamayo_pc.load(gpu)
  t0 = time.time()
  for n, (f, data) in enumerate(iter(loaded.get, None)):
    sample = {k: data[k] for k in KEYS}
    sample['nav_text'] = [nav or str(data['nav_text'])]
    p = rt.predict(sample)
    # A small side file: rewriting the 10 MB compressed sample cost more than the model call.
    side = side_path(f, nav); tmp = side.with_name(f.stem + '.frozen.tmp.npz')
    side.parent.mkdir(parents=True, exist_ok=True)
    np.savez(tmp, reasoning_token_ids=np.asarray(p['generated_token_ids'], dtype=np.int32),
             baseline_xyz=np.asarray(p['xyz'], dtype=np.float32)[0], baseline_rot=np.asarray(p['rot'], dtype=np.float32)[0],
             reasoning=np.array(p['reasoning']), nav_text=np.array(sample['nav_text'][0]))
    tmp.replace(side)
    if nav is None:
      meta = f.with_suffix('.json')
      m = json.loads(meta.read_text())
      m['reasoning'] = p['reasoning']
      meta.write_text(json.dumps(m))
    if n % 20 == 0:
      print(f'{n+1}/{len(todo)} {(time.time()-t0)/(n+1):.2f} s/sample {p["reasoning"][:60]!r}', flush=True)
  print('done', len(todo), flush=True)


if __name__ == '__main__':
  argv = sys.argv[2:]
  nav = None
  if argv[:1] == ['--nav']:
    nav, argv = argv[1], argv[2:]
    assert nav in FROZEN_ROOTS, f'no mirror root for {nav!r}'
  main(int(sys.argv[1]), argv, nav)
