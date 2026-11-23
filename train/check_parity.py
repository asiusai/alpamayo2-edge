"""Check (a): the BF16 PyTorch expert sampled the car's way reproduces the frozen model's trajectories.

For each sample: frozen prefix over prompt + baseline reasoning tokens, 1 Euler step from the car noise, CPU decode.
Compares with baseline_xyz (the frozen car recipe's own output). Also checks this harness's expert forward
against NVIDIA's HF path and that a fresh (zero-B) LoRA gives bit-identical outputs.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from alpamayo_train.common import DATA_ROOT, ade, log, start

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('samples', type=Path, nargs='+')
parser.add_argument('--gpu', type=int, default=1)
parser.add_argument('--checkpoint', type=Path, default=None)
parser.add_argument('--steps', type=int, default=1)
parser.add_argument('--reference-checks', type=int, default=4, help='samples compared with the upstream HF expert path')
parser.add_argument('--out', type=Path, default=Path(DATA_ROOT) / 'runs' / 'parity.json')
parser.add_argument('--limit', type=int, default=None)
args = parser.parse_args()

gpu = start(args.gpu)
import torch  # noqa: E402

from alpamayo_train.data import Preparer, Prefetcher, discover  # noqa: E402
from alpamayo_train.model import ORIGINAL_CHECKPOINT, ExpertTrainer  # noqa: E402

checkpoint = args.checkpoint or ORIGINAL_CHECKPOINT
refs, skipped = discover(args.samples, limit=args.limit)
log(f'{len(refs)} ready samples, skipped {skipped}')
preparer = Preparer(checkpoint)
prefetch = Prefetcher(preparer, refs)  # thread only
device = gpu.open_device(args.gpu)
t = time.perf_counter()
trainer = ExpertTrainer(checkpoint, device, rank=16)
log(f'model loaded in {time.perf_counter() - t:.1f} s, {torch.cuda.memory_allocated() / 2**30:.2f} GiB, kfd_evicted_ms {gpu.evicted_ms()}')

rows = []
for i, item in enumerate(prefetch):
  if 'error' in item:
    log('SKIP', item['error'])
    continue
  torch.cuda.synchronize()
  t0 = time.perf_counter()
  prefix = trainer.prefix(item)
  torch.cuda.synchronize()
  t1 = time.perf_counter()
  trainer.set_lora_enabled(False)
  action = trainer.sample_car(prefix, steps=args.steps)
  torch.cuda.synchronize()
  t2 = time.perf_counter()
  trainer.set_lora_enabled(True)
  action_lora = trainer.sample_car(prefix, steps=args.steps)
  row = {'stem': item['stem'], 'prefix_ms': (t1 - t0) * 1e3, 'expert_ms': (t2 - t1) * 1e3,
         'tokens': prefix['length'], 'kfd_evicted_ms': sum(gpu.evicted_ms().values()), 'lora_zero_identical': bool(torch.equal(action, action_lora))}
  if i < args.reference_checks:
    trainer.set_lora_enabled(False)
    tt = torch.zeros(1, device=device)
    mine = trainer.velocity(trainer.car_noise, tt, prefix)
    ref = trainer.reference_velocity(trainer.car_noise, tt, item)
    row['velocity_vs_upstream_max_abs'] = float((mine - ref).abs().max())
    row['velocity_max_abs'] = float(ref.abs().max())
    trainer.set_lora_enabled(True)
  xyz, rot = preparer.decode(trainer.stager.d2h(action), item)
  xyz = xyz[0].numpy()
  base = item['baseline_xyz']
  row['ade'], row['fde'] = ade(xyz, base)
  yaw = np.arctan2(rot[0, :, 1, 0].numpy(), rot[0, :, 0, 0].numpy())
  row['pred_y_3s'], row['base_y_3s'] = float(xyz[29, 1]), float(base[29, 1])
  row['yaw'] = yaw.tolist()
  row['xyz'] = xyz.tolist()
  rows.append(row)
  log(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items() if k not in ('xyz', 'yaw')}))
  del prefix

a = np.array([r['ade'] for r in rows])
f = np.array([r['fde'] for r in rows])
summary = {'n': len(rows), 'steps': args.steps, 'ade_median': float(np.median(a)), 'ade_p90': float(np.percentile(a, 90)),
           'ade_max': float(a.max()), 'fde_median': float(np.median(f)), 'fde_p90': float(np.percentile(f, 90)),
           'lora_zero_identical_all': all(r['lora_zero_identical'] for r in rows),
           'velocity_vs_upstream_max_abs': max((r.get('velocity_vs_upstream_max_abs', 0) for r in rows), default=None),
           'prefix_ms_median': float(np.median([r['prefix_ms'] for r in rows])),
           'expert_ms_median': float(np.median([r['expert_ms'] for r in rows])),
           'max_memory_gib': torch.cuda.max_memory_allocated() / 2**30, 'kfd_evicted_ms': gpu.evicted_ms()}
log('SUMMARY', json.dumps(summary))
args.out.parent.mkdir(parents=True, exist_ok=True)
args.out.write_text(json.dumps({'summary': summary, 'rows': rows}) + '\n')
