"""Initial-noise test: the car's fixed noise vs zero noise vs random draws, at 1/2/10 Euler steps (frozen reasoning tokens).

Usage (alpamayo-train env): noise_test.py GPU SAMPLE_DIR OUT_PREFIX [--lora DIR] [--limit N] [--kinds clean]
Writes OUT_PREFIX-<noise>-s<steps>.npz with noise in {car, zero, rand0, rand1}.
"""
import argparse
import os
import sys
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument('gpu', type=int); ap.add_argument('samples', type=Path); ap.add_argument('out', type=str)
ap.add_argument('--lora', type=Path); ap.add_argument('--limit', type=int); ap.add_argument('--kinds', default='clean')
a = ap.parse_args()
TRAIN = str(Path(__file__).resolve().parents[1]/'train'); sys.path.insert(0, TRAIN)
os.chdir(TRAIN)
from alpamayo_train.common import start  # noqa: E402
gpu_mod = start(a.gpu, 8)
import numpy as np  # noqa: E402
import torch  # noqa: E402
from alpamayo_train.data import Preparer, discover  # noqa: E402
from alpamayo_train.model import ORIGINAL_CHECKPOINT, ExpertTrainer  # noqa: E402

refs, _ = discover([a.samples], kinds=set(a.kinds.split(',')) if a.kinds else None, limit=a.limit)
pre = Preparer(ORIGINAL_CHECKPOINT)
tr = ExpertTrainer(ORIGINAL_CHECKPOINT, gpu_mod.open_device(a.gpu))
if a.lora:
  tr.load_lora(a.lora); tr.merge_lora_inplace()
g = torch.Generator().manual_seed(7)
noises = {'car': None, 'zero': torch.zeros(1, 64, 2), 'rand0': torch.randn(1, 64, 2, generator=g), 'rand1': torch.randn(1, 64, 2, generator=g)}
steps = (1, 2, 10)
preds = {(n, s): {} for n in noises for s in steps}
for i, ref in enumerate(refs):
  try:
    item = pre(ref)
  except ValueError:
    continue
  prefix = tr.prefix(item)
  for n, z in noises.items():
    for s in steps:
      action = tr.sample_car(prefix, steps=s, noise=z)
      preds[(n, s)][ref.stem] = pre.decode(tr.stager.d2h(action), item)[0][0].numpy().astype(np.float32)
  del prefix
  if i % 50 == 0:
    print(i+1, '/', len(refs), flush=True)
for (n, s), p in preds.items():
  np.savez(f'{a.out}-{n}-s{s}.npz', **p)
print('done', flush=True)
