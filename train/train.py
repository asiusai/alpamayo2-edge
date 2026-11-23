"""Train LoRA adapters on the Alpamayo 2 Edge action expert with the flow-matching loss.

Per sample: frozen VLM prefix (10 images, ego history, nav text, teacher-forced frozen reasoning ending with the
trajectory anchor), then n_noise flow-matching draws through the expert. The VLM, reasoning and action projections
stay frozen; only the LoRA adapters on the expert attention and MLP projections train.
One process per GPU (``--gpus 0,1``); gradients are averaged with a Gloo all-reduce.
"""
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

from alpamayo_train.common import DATA_ROOT
from alpamayo_train.launch import child_env, run_children

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('samples', type=Path, nargs='*', help='sample directories')
parser.add_argument('--manifest', type=Path, default=None,
                    help='text file with one sample .npz path per line; a repeated path is drawn that many times per epoch')
parser.add_argument('--out', type=Path, required=True, help=f'run directory (under {DATA_ROOT}/runs)')
parser.add_argument('--gpus', default='1', help='comma-separated GPU indices (0=03:00.0, 1=0c:00.0)')
parser.add_argument('--checkpoint', type=Path, default=None, help='original BF16 checkpoint')
parser.add_argument('--rank', type=int, default=16)
parser.add_argument('--alpha', type=float, default=None, help='LoRA alpha (default 2*rank)')
parser.add_argument('--dropout', type=float, default=0.0)
parser.add_argument('--targets', default='q,k,v,o,fc1,fc2', help='subset of q,k,v,o,fc1,fc2')
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--weight-decay', type=float, default=0.0)
parser.add_argument('--warmup', type=int, default=20)
parser.add_argument('--steps', type=int, default=None, help='optimizer steps (default: --epochs over the data)')
parser.add_argument('--epochs', type=float, default=1.0)
parser.add_argument('--samples-per-step', type=int, default=4, help='samples per GPU per optimizer step')
parser.add_argument('--n-noise', type=int, default=8, help='flow-matching (t, noise) draws per sample')
parser.add_argument('--car-step-weight', type=float, default=0.0,
                    help='extra MSE weight on the car Euler result from the car noise (zeros)')
parser.add_argument('--car-steps', type=int, default=1, help='Euler steps of the car result in the car loss (the car runs 10)')
parser.add_argument('--lat-weight', type=float, default=0.0,
                    help='weight of the cross-track error (m, smooth L1) of the car result with the target speed profile')
parser.add_argument('--car-grad-steps', type=int, default=None, help='backpropagate through only the last N car steps (default: all)')
parser.add_argument('--val-steps', type=int, default=10, help='Euler steps of the validation sampler (the car runs 10)')
parser.add_argument('--clip', type=float, default=1.0)
parser.add_argument('--kinds', default=None, help='comma-separated meta kinds to use (recovery,clean)')
parser.add_argument('--limit', type=int, default=None, help='use the first N ready samples (sorted by stem)')
parser.add_argument('--cache-prefix', type=int, default=0, help='keep up to N frozen prefixes on the GPU (small sets)')
parser.add_argument('--grad-checkpointing', action='store_true')
parser.add_argument('--val', type=Path, nargs='*', default=[], help='validation sample directories')
parser.add_argument('--val-limit', type=int, default=64)
parser.add_argument('--val-every', type=int, default=100)
parser.add_argument('--save-every', type=int, default=200)
parser.add_argument('--log-every', type=int, default=1)
parser.add_argument('--init-lora', type=Path, default=None, help='continue from an adapter directory')
parser.add_argument('--vlm-rank', type=int, default=0, help='LoRA rank on the VLM (0: VLM frozen, the default)')
parser.add_argument('--vlm-scope', default='language', choices=('language', 'both'), help="VLM adapters: decoder only, or 'both' (+ vision)")
parser.add_argument('--vlm-lr', type=float, default=None, help='peak LR of the VLM adapters (default: --lr)')
parser.add_argument('--ce-weight', type=float, default=0.2, help='weight of the reasoning-token cross-entropy (VLM training only)')
parser.add_argument('--teacher', type=Path, nargs='*', default=[],
                    help='directories with <stem>.frozen.npz from another view set; their reasoning ids replace the sample\'s (distillation)')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--threads', type=int, default=8)
args = parser.parse_args()
gpus = [int(g) for g in args.gpus.split(',')]
ctx = child_env()
if ctx is None:
  if not str(args.out.resolve()).startswith(DATA_ROOT) and not str(args.out.resolve()).startswith(str(Path.home())):
    sys.exit(f'--out must be under {DATA_ROOT} or {Path.home()}')
  args.out.mkdir(parents=True, exist_ok=True)
  sys.exit(run_children(gpus))

# ---- child: one GPU ----------------------------------------------------------------------------------------------
from alpamayo_train.common import start  # noqa: E402

gpu_mod = start(ctx['gpu'], args.threads)
import numpy as np  # noqa: E402
import torch  # noqa: E402

from alpamayo_train import launch  # noqa: E402
from alpamayo_train.data import Preparer, Prefetcher, discover  # noqa: E402
from alpamayo_train.model import LORA_TARGETS, ORIGINAL_CHECKPOINT, ExpertTrainer, source_digest  # noqa: E402

rank, world = ctx['rank'], ctx['world']
main = rank == 0
checkpoint = args.checkpoint or ORIGINAL_CHECKPOINT
short = {'q': 'self_attn.q_proj', 'k': 'self_attn.k_proj', 'v': 'self_attn.v_proj', 'o': 'self_attn.o_proj',
         'fc1': 'mlp.fc1', 'fc2': 'mlp.fc2'}
targets = tuple(short[t] for t in args.targets.split(','))
assert set(targets) <= set(LORA_TARGETS)
kinds = set(args.kinds.split(',')) if args.kinds else None

if args.manifest:
  lines = [Path(x) for x in args.manifest.read_text().split()]
  pool, skipped = discover(sorted({x.parent for x in lines}), kinds=kinds)
  by_path = {r.path.resolve(): r for r in pool}
  refs = [by_path[x.resolve()] for x in lines if x.resolve() in by_path]
  skipped['not_in_pool'] = len(lines) - len(refs)
else:
  if not args.samples:
    sys.exit('Give sample directories or --manifest')
  refs, skipped = discover(args.samples, limit=args.limit, kinds=kinds)
if len(refs) < world:
  sys.exit(f'Need at least {world} ready samples, found {len(refs)} (skipped {skipped})')
shard = refs[rank::world]
per_rank = min(len(refs[r::world]) for r in range(world))
steps = args.steps or max(1, int(args.epochs * per_rank / args.samples_per_step))
val_refs = []
if args.val:
  val_refs, _ = discover(args.val, kinds=None)
  if args.val_limit:
    step_v = max(1, len(val_refs) // args.val_limit)
    val_refs = val_refs[::step_v][:args.val_limit]
  val_refs = val_refs[rank::world]

torch.manual_seed(args.seed * 1000 + rank)
preparer = Preparer(checkpoint)
if args.teacher:
  teacher_files = {f.name[:-len('.frozen.npz')]: f for d in args.teacher for f in Path(d).glob('*.frozen.npz')}
  base_call = preparer.__call__

  class TeacherPreparer(type(preparer)):
    """Teacher-forced reasoning from another view set of the same moment (e.g. 120/30 for 180/60 inputs)."""
    def __call__(self, ref):
      out = base_call(ref)
      side = teacher_files.get(ref.stem)
      if side is None:
        raise ValueError('no teacher reasoning')
      with np.load(side, allow_pickle=False) as z:
        ids = z['reasoning_token_ids'].astype(np.int64)
      if int(ids[-1]) != self.anchor or int((ids == self.anchor).sum()) != 1:
        raise ValueError('teacher reasoning lacks the anchor')
      out['reasoning_ids'] = torch.from_numpy(ids)
      return out
  preparer.__class__ = TeacherPreparer
  refs_with_teacher = [r for r in shard if r.stem in teacher_files]
  if main:
    print(f'teacher reasoning for {len(refs_with_teacher)}/{len(shard)} samples of this rank', flush=True)
  shard = refs_with_teacher
train_data = Prefetcher(preparer, shard, epochs=None, shuffle=True, seed=args.seed * 1000 + rank, depth=2 * args.samples_per_step)
device = gpu_mod.open_device(ctx['gpu'])
launch.init_group(ctx)

t_load = time.perf_counter()
trainer = ExpertTrainer(checkpoint, device, grad_checkpointing=args.grad_checkpointing)
if args.init_lora:
  trainer.load_lora(args.init_lora)
else:
  trainer.add_lora(args.rank, args.alpha if args.alpha is not None else 2.0 * args.rank, args.dropout, targets)
train_vlm = args.vlm_rank > 0 or any(n.startswith('vlm.') for n in trainer.lora)
if args.vlm_rank > 0 and not any(n.startswith('vlm.') for n in trainer.lora):
  trainer.add_vlm_lora(args.vlm_rank, 2.0 * args.vlm_rank, args.vlm_scope, args.dropout)
params = trainer.trainable_parameters()
launch.all_reduce_mean([p.data for p in params], trainer.stager)  # identical initial adapters on every rank
expert_params, vlm_params = trainer.parameter_groups()
lr_scale = {0: 1.0, 1: (args.vlm_lr or args.lr) / args.lr}
optimizer = torch.optim.AdamW([{'params': expert_params}] + ([{'params': vlm_params}] if vlm_params else []),
                              lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.999))


def lr_at(step):
  if step < args.warmup:
    return args.lr * (step + 1) / args.warmup
  progress = (step - args.warmup) / max(1, steps - args.warmup)
  return args.lr * 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))


run_info = {'args': json.loads(json.dumps(vars(args), default=str)),
            'samples': len(refs), 'skipped': skipped, 'steps': steps, 'world': world, 'gpus': gpus,
            'trainable_params': sum(p.numel() for p in params), 'nvidia_source_sha256': source_digest(),
            'torch': torch.__version__, 'hip': torch.version.hip, 'checkpoint': str(checkpoint)}
if main:
  (args.out / 'run.json').write_text(json.dumps(run_info, indent=2) + '\n')
  print(json.dumps({k: v for k, v in run_info.items() if k != 'args'}), flush=True)
  print(f'model+LoRA ready in {time.perf_counter() - t_load:.1f} s, {torch.cuda.memory_allocated() / 2**30:.2f} GiB', flush=True)
log_file = open(args.out / 'log.jsonl', 'a') if main else None
prefix_cache: dict[str, dict] = {}


def get_prefix(item):
  if item['stem'] in prefix_cache:
    return prefix_cache[item['stem']]
  p = trainer.prefix(item)
  if len(prefix_cache) < args.cache_prefix:
    prefix_cache[item['stem']] = p
  return p


def validate(step):
  """Car-way sampling (--val-steps Euler steps, car noise) on the validation shard; distances to target and baseline."""
  rows = []
  if val_refs:
    for item in Prefetcher(preparer, val_refs, epochs=1, depth=4):
      if 'error' in item:
        continue
      prefix = trainer.prefix(item)
      action = trainer.sample_car(prefix, steps=args.val_steps)
      xyz = preparer.decode(trainer.stager.d2h(action), item)[0][0].numpy()
      d_t = np.linalg.norm(xyz[:, :2] - item['target_xyz'][:, :2], axis=-1)
      d_b = np.linalg.norm(xyz[:, :2] - item['baseline_xyz'][:, :2], axis=-1)
      rows.append({'stem': item['stem'], 'kind': item['meta'].get('kind'), 'ade_target': float(d_t.mean()),
                   'fde_target': float(d_t[-1]), 'ade_baseline': float(d_b.mean()),
                   'lat_err_2s': float(xyz[19, 1] - item['target_xyz'][19, 1]),
                   'lat_err_4s': float(xyz[39, 1] - item['target_xyz'][39, 1])})
      del prefix
  rows = [r for part in launch.gather_objects(rows) for r in part]
  if main and rows:
    summary = {'step': step, 'val_n': len(rows)}
    for kind in sorted({r['kind'] for r in rows} | {'all'}):
      sel = [r for r in rows if kind == 'all' or r['kind'] == kind]
      for key in ('ade_target', 'fde_target', 'ade_baseline'):
        summary[f'{kind}_{key}'] = float(np.mean([r[key] for r in sel]))
      for key in ('lat_err_2s', 'lat_err_4s'):
        summary[f'{kind}_abs_{key}'] = float(np.mean([abs(r[key]) for r in sel]))
    print('VAL', json.dumps(summary), flush=True)
    log_file.write(json.dumps({'val': summary}) + '\n')
    log_file.flush()
    (args.out / f'val-{step:06d}.json').write_text(json.dumps({'summary': summary, 'rows': rows}) + '\n')


data_iter = iter(train_data)
if args.val_every and val_refs is not None:
  validate(0)
torch.cuda.reset_peak_memory_stats()
t_run = time.perf_counter()
for step in range(steps):
  for g, group in enumerate(optimizer.param_groups):
    group['lr'] = lr_at(step) * lr_scale[g]
  optimizer.zero_grad(set_to_none=False)
  stats = {'loss_fm': [], 'loss_car': [], 'loss_ce': [], 'loss_lat': [], 'cross_3s_m': [], 'roundtrip_max_m': []}
  t_wait = t_prefix = t_expert = 0.0
  n = 0
  while n < args.samples_per_step:
    t0 = time.perf_counter()
    item = next(data_iter)
    t1 = time.perf_counter()
    t_wait += t1 - t0
    if 'error' in item:
      print('SKIP', item['error'], flush=True)
      continue
    if train_vlm:
      prefix, loss_ce = trainer.prefix_train(item, ce=args.ce_weight > 0)
    else:
      prefix, loss_ce = get_prefix(item), None
    torch.cuda.synchronize()
    t2 = time.perf_counter()
    geom = None
    if args.lat_weight > 0:
      sp = preparer.action_space
      geom = {'v0': float(item['t0_states']['v'].reshape(-1)[0]), 'theta0': float(item['t0_states']['theta'].reshape(-1)[0]), 'dt': float(sp.dt),
              **{k: getattr(sp, k).float().to(trainer.device) for k in ('accel_mean', 'accel_std', 'curvature_mean', 'curvature_std')}}
    loss, s = trainer.loss(prefix, item['target_action'], args.n_noise, args.car_step_weight, car_steps=args.car_steps,
                           car_grad_steps=args.car_grad_steps, lat_weight=args.lat_weight, geom=geom)
    if loss_ce is not None:
      loss = loss + args.ce_weight * loss_ce
      s['loss_ce'] = float(loss_ce.detach())
    (loss / args.samples_per_step).backward()
    torch.cuda.synchronize()
    t3 = time.perf_counter()
    t_prefix += t2 - t1
    t_expert += t3 - t2
    for k, v in s.items():
      stats[k].append(v)
    stats['roundtrip_max_m'].append(item['roundtrip_max_m'])
    n += 1
    del prefix
  grads = [p.grad for p in params]
  launch.all_reduce_mean(grads, trainer.stager)
  grad_norm = float(torch.nn.utils.clip_grad_norm_(params, args.clip))
  optimizer.step()
  if main and (step % args.log_every == 0 or step == steps - 1):
    record = {'step': step + 1, 'lr': lr_at(step), 'grad_norm': grad_norm,
              'loss_fm': float(np.mean(stats['loss_fm'])),
              **({'loss_car': float(np.mean(stats['loss_car']))} if stats['loss_car'] else {}),
              **({'loss_ce': float(np.mean(stats['loss_ce']))} if stats['loss_ce'] else {}),
              **({'loss_lat': float(np.mean(stats['loss_lat'])), 'cross_3s_m': float(np.mean(stats['cross_3s_m']))} if stats['loss_lat'] else {}),
              'roundtrip_max_m': float(np.max(stats['roundtrip_max_m'])),
              'ms_per_sample': 1e3 * (t_prefix + t_expert) / n, 'prefix_ms': 1e3 * t_prefix / n,
              'expert_fwd_bwd_ms': 1e3 * t_expert / n, 'data_wait_ms': 1e3 * t_wait / n,
              'peak_gib': torch.cuda.max_memory_allocated() / 2**30,
              'reserved_gib': torch.cuda.max_memory_reserved() / 2**30,
              'elapsed_s': time.perf_counter() - t_run, 'kfd_evicted_ms': sum(gpu_mod.evicted_ms().values())}
    print(json.dumps({k: round(v, 5) if isinstance(v, float) else v for k, v in record.items()}), flush=True)
    log_file.write(json.dumps(record) + '\n')
    log_file.flush()
  done = step + 1
  if main and (done % args.save_every == 0 or done == steps):
    trainer.save_lora(args.out / ('final' if done == steps else f'step-{done:06d}'),
                      {'step': done, 'checkpoint': str(checkpoint), 'run': str(args.out)})
  if args.val_every and (done % args.val_every == 0 or done == steps):
    validate(done)
launch.barrier()
train_data.close()
if main:
  print(json.dumps({'done': steps, 'elapsed_s': time.perf_counter() - t_run, 'kfd_evicted_ms': gpu_mod.evicted_ms()}), flush=True)
os._exit(0)  # skip interpreter teardown of the data thread and process group
