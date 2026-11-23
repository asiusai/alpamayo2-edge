"""Sample trajectories the car's way (frozen reasoning tokens, Euler steps from the car noise: zeros) for evaluation.

Writes PREDICTIONS.npz mapping each sample stem to a (64,3) float32 xyz in the query frame, plus PREDICTIONS.json
with per-sample distances to target and baseline. Without --lora this is the frozen BF16 model.
"""
import argparse
import json
import os
import sys
from pathlib import Path

from alpamayo_train.launch import child_env, run_children

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('samples', type=Path, nargs='+')
parser.add_argument('--out', type=Path, required=True, help='PREDICTIONS.npz')
parser.add_argument('--lora', type=Path, default=None, help='adapter directory (adapter.safetensors + adapter.json)')
parser.add_argument('--checkpoint', type=Path, default=None, help='BF16 checkpoint; e.g. a merged export')
parser.add_argument('--gpus', default='1')
parser.add_argument('--steps', type=int, default=10, help='Euler steps (the car uses 10)')
parser.add_argument('--fp8-weights', action='store_true',
                    help='round the (merged) expert linears to per-tensor FP8 E4M3 like prepare-fp8.py')
parser.add_argument('--generate', action='store_true',
                    help='the model writes its own greedy reasoning (needed after VLM training) instead of the frozen reasoning ids')
parser.add_argument('--x0', default=None,
                    help='ACCEL,CURVATURE: constant initial state in normalized action units instead of zeros '
                         '(0,-0.0103 starts from zero physical curvature; zeros start from the dataset mean, 2.7e-4 1/m left)')
parser.add_argument('--stems', type=Path, default=None, help='text file of sample stems to use')
parser.add_argument('--kinds', default=None)
parser.add_argument('--limit', type=int, default=None)
parser.add_argument('--threads', type=int, default=8)
args = parser.parse_args()
gpus = [int(g) for g in args.gpus.split(',')]
ctx = child_env()
if ctx is None:
  args.out.parent.mkdir(parents=True, exist_ok=True)
  code = run_children(gpus)
  if code:
    sys.exit(code)
  # Merge the per-rank parts (parent: CPU only, no torch).
  import numpy as np
  preds, rows = {}, []
  for rank in range(len(gpus)):
    part = args.out.with_name(f'{args.out.stem}.part{rank}.npz')
    with np.load(part) as d:
      preds.update({k: d[k] for k in d.files})
    rows += json.loads(part.with_suffix('.json').read_text())
    part.unlink()
    part.with_suffix('.json').unlink()
  np.savez(args.out, **preds)
  summary = {'n': len(rows), 'lora': str(args.lora), 'checkpoint': str(args.checkpoint), 'steps': args.steps,
             'fp8_weights': args.fp8_weights, 'generate': args.generate}
  if args.generate:
    summary['reasoning_same_as_frozen'] = float(np.mean([r.get('reasoning_same_as_frozen', False) for r in rows]))
  for kind in sorted({r['kind'] for r in rows} | {'all'}):
    sel = [r for r in rows if kind == 'all' or r['kind'] == kind]
    for key in ('ade_target', 'fde_target', 'ade_baseline'):
      summary[f'{kind}_{key}'] = float(np.mean([r[key] for r in sel]))
  args.out.with_suffix('.json').write_text(json.dumps({'summary': summary, 'rows': rows}, indent=1) + '\n')
  print(json.dumps(summary), flush=True)
  sys.exit(0)

from alpamayo_train.common import start  # noqa: E402

gpu_mod = start(ctx['gpu'], args.threads)
import numpy as np  # noqa: E402
import torch  # noqa: E402

from alpamayo_train.data import Preparer, Prefetcher, discover  # noqa: E402
from alpamayo_train.model import ORIGINAL_CHECKPOINT, ExpertTrainer  # noqa: E402

checkpoint = args.checkpoint or ORIGINAL_CHECKPOINT
refs, skipped = discover(args.samples, limit=args.limit, kinds=set(args.kinds.split(',')) if args.kinds else None)
if args.stems:
  keep = set(args.stems.read_text().split()); refs = [r for r in refs if r.stem in keep]
stems = [r.stem for r in refs]
if len(set(stems)) != len(stems):
  sys.exit('Duplicate sample stems across the given directories; sample one split per PREDICTIONS.npz')
refs = refs[ctx['rank']::ctx['world']]
preparer = Preparer(checkpoint)
data = Prefetcher(preparer, refs, epochs=1, depth=4)
device = gpu_mod.open_device(ctx['gpu'])
trainer = ExpertTrainer(checkpoint, device)
if args.lora:
  trainer.load_lora(args.lora)
  trainer.merge_lora_inplace()
if args.fp8_weights:
  trainer.fp8_weight_roundtrip()
preds, rows = {}, []
for item in data:
  if 'error' in item:
    print('SKIP', item['error'], flush=True)
    continue
  extra = {}
  if args.generate:
    frozen_ids = item['reasoning_ids']
    try:
      item['reasoning_ids'] = trainer.generate_reasoning(item)
    except ValueError as e:
      print('SKIP', item['stem'], e, flush=True)
      continue
    same = item['reasoning_ids'].shape == frozen_ids.shape and bool((item['reasoning_ids'] == frozen_ids).all())
    extra = {'reasoning': preparer.tokenizer.decode(item['reasoning_ids'][:-1]), 'reasoning_same_as_frozen': same}
  prefix = trainer.prefix(item)
  noise = None
  if args.x0:
    noise = torch.zeros(1, 64, 2)
    noise[..., 0], noise[..., 1] = (float(v) for v in args.x0.split(','))
  action = trainer.sample_car(prefix, steps=args.steps, noise=noise)
  xyz = preparer.decode(trainer.stager.d2h(action), item)[0][0].numpy().astype(np.float32)
  preds[item['stem']] = xyz
  d_t = np.linalg.norm(xyz[:, :2] - item['target_xyz'][:, :2], axis=-1)
  d_b = np.linalg.norm(xyz[:, :2] - item['baseline_xyz'][:, :2], axis=-1)
  rows.append({'stem': item['stem'], 'kind': item['meta'].get('kind'), 'dy_m': item['meta'].get('dy_m'),
               'dpsi_rad': item['meta'].get('dpsi_rad'), 'ade_target': float(d_t.mean()), 'fde_target': float(d_t[-1]),
               'ade_baseline': float(d_b.mean()), 'roundtrip_max_m': item['roundtrip_max_m'], **extra})
  del prefix
  if len(rows) % 50 == 0:
    print(f'rank {ctx["rank"]}: {len(rows)}/{len(refs)}', flush=True)
part = args.out.with_name(f'{args.out.stem}.part{ctx["rank"]}.npz')
np.savez(part, **preds)
part.with_suffix('.json').write_text(json.dumps(rows) + '\n')
print(json.dumps({'rank': ctx['rank'], 'n': len(rows), 'skipped': skipped, 'kfd_evicted_ms': gpu_mod.evicted_ms()}), flush=True)
os._exit(0)
