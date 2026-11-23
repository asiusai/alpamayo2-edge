"""Fresh random initial noise for every plan (NVIDIA's default sampling) vs zero noise, on car-format samples at 10 steps.

Usage (alpamayo-train env): noise_draws.py GPU SAMPLE_DIR OUT_PREFIX [--lora DIR] [--limit N] [--kinds clean] [--draws 4]
Frozen reasoning tokens, the car's sampler. Writes OUT_PREFIX-zero.npz and OUT_PREFIX-d<k>.npz; each draw uses its own
noise vector per sample (seeded by the sample name and k).
"""
import argparse
import os
import sys
import zlib
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument('gpu', type=int); ap.add_argument('samples', type=Path); ap.add_argument('out', type=str)
ap.add_argument('--lora', type=Path); ap.add_argument('--limit', type=int); ap.add_argument('--kinds', default='clean')
ap.add_argument('--draws', type=int, default=4); ap.add_argument('--steps', type=int, default=10)
ap.add_argument('--temperature', type=float, default=1.)
ap.add_argument('--stems', type=Path, help='text file of sample stems to use'); ap.add_argument('--shard', default='0/1')
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
if a.stems:
  keep = set(a.stems.read_text().split()); refs = [r for r in refs if r.stem in keep]
si, sn = (int(v) for v in a.shard.split('/')); refs = refs[si::sn]
pre = Preparer(ORIGINAL_CHECKPOINT)
tr = ExpertTrainer(ORIGINAL_CHECKPOINT, gpu_mod.open_device(a.gpu))
if a.lora:
  tr.load_lora(a.lora); tr.merge_lora_inplace()
names = ['zero']+[f'd{k}' for k in range(a.draws)]
preds = {n: {} for n in names}
for i, ref in enumerate(refs):
  try:
    item = pre(ref)
  except ValueError:
    continue
  prefix = tr.prefix(item)
  for n in names:
    z = torch.zeros(1, 64, 2) if n == 'zero' else a.temperature*torch.randn(1, 64, 2, generator=torch.Generator().manual_seed(zlib.crc32(f'{ref.stem}/{n}'.encode())))
    action = tr.sample_car(prefix, steps=a.steps, noise=z)
    preds[n][ref.stem] = pre.decode(tr.stager.d2h(action), item)[0][0].numpy().astype(np.float32)
  del prefix
  if i % 50 == 0:
    print(i+1, '/', len(refs), flush=True)
for n, p in preds.items():
  np.savez(f'{a.out}-{n}.npz', **p)
print('done', flush=True)
