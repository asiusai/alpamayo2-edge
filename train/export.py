"""Merge a LoRA adapter into the expert weights and save a checkpoint in the original layout (CPU only).

Output: the original shard names, index, config, tokenizer and processor files. Only the 168 targeted
``expert.expert.layers.N.{self_attn.q/k/v/o_proj,mlp.fc1/fc2}.weight`` tensors change (BF16(W + B@A*alpha/r)).
``export.json`` records hashes and how much of each update survives BF16 and the car's per-tensor FP8 E4M3 rounding.
"""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

ORIGINAL = Path(__file__).resolve().parents[1] / 'pc/data/original-checkpoint'
FILES = ('config.json', 'generation_config.json', 'tokenizer.json', 'tokenizer_config.json', 'processor_config.json',
         'chat_template.jinja', 'LICENSE', 'README.md', '.gitattributes', 'model.safetensors.index.json')

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('lora', type=Path, help='adapter directory')
parser.add_argument('out', type=Path, help='new checkpoint directory (must not exist)')
parser.add_argument('--checkpoint', type=Path, default=ORIGINAL)
args = parser.parse_args()
torch.set_num_threads(8)


def sha256(path: Path) -> str:
  with path.open('rb') as f:
    return hashlib.file_digest(f, 'sha256').hexdigest()


def fp8(w: torch.Tensor) -> torch.Tensor:
  """prepare-fp8.py weight rounding: per-tensor scale max|W|/448, E4M3, dequantized to float32."""
  scale = (w.abs().max() / 448).clamp(min=1e-12)
  return (w / scale).clamp(-448, 448).to(torch.float8_e4m3fn).float() * scale


if args.out.exists():
  raise SystemExit(f'{args.out} exists')
config = json.loads((args.lora / 'adapter.json').read_text())
adapter = load_file(str(args.lora / 'adapter.safetensors'))
scaling = config['alpha'] / config['rank']
names = sorted({k.rsplit('.', 1)[0] for k in adapter})
updates = {f'{n}.weight': (adapter[f'{n}.lora_B'] @ adapter[f'{n}.lora_A']) * scaling for n in names}
index = json.loads((args.checkpoint / 'model.safetensors.index.json').read_text())['weight_map']
missing = [n for n in updates if n not in index]
if missing:
  raise SystemExit(f'Adapter targets missing from checkpoint: {missing[:3]}')

tmp = args.out.with_name(args.out.name + '.tmp')
if tmp.exists():
  shutil.rmtree(tmp)
tmp.mkdir(parents=True)
stats = []
for shard in sorted(set(index.values())):
  with safe_open(str(args.checkpoint / shard), framework='pt') as f:
    metadata = f.metadata()
    tensors = {k: f.get_tensor(k) for k in f.keys()}
  for name, delta in updates.items():
    if index[name] != shard:
      continue
    w = tensors[name]
    merged = (w.float() + delta).to(w.dtype)
    wf, mf = w.float(), merged.float()
    dn = float((delta * delta).sum())
    stats.append({'name': name, 'rel_update': (dn ** 0.5) / float(wf.norm()),
                  'bf16_retained': float(((mf - wf) * delta).sum()) / dn if dn else 0.0,
                  'fp8_retained': float(((fp8(mf) - fp8(wf)) * delta).sum()) / dn if dn else 0.0,
                  'fp8_scale_change': float(mf.abs().max() / wf.abs().max()) - 1.0})
    tensors[name] = merged.contiguous()
  save_file(tensors, str(tmp / shard), metadata=metadata)
  print('wrote', shard, flush=True)
  del tensors
for name in FILES:
  if (args.checkpoint / name).exists():
    shutil.copyfile(args.checkpoint / name, tmp / name)


def mean(key):
  return sum(s[key] for s in stats) / len(stats)


info = {'source_checkpoint': str(args.checkpoint),
        'source_shards': {s: sha256(args.checkpoint / s) for s in sorted(set(index.values()))},
        'adapter': str(args.lora), 'adapter_sha256': sha256(args.lora / 'adapter.safetensors'), 'adapter_config': config,
        'merged_tensors': len(stats),
        'shards': {s: sha256(tmp / s) for s in sorted(set(index.values()))},
        'config_sha256': sha256(tmp / 'config.json'), 'index_sha256': sha256(tmp / 'model.safetensors.index.json'),
        'mean_rel_update': mean('rel_update'), 'mean_bf16_retained': mean('bf16_retained'),
        'mean_fp8_retained': mean('fp8_retained'), 'min_fp8_retained': min(s['fp8_retained'] for s in stats),
        'tensors': stats}
(tmp / 'export.json').write_text(json.dumps(info, indent=1) + '\n')
tmp.rename(args.out)
print(json.dumps({k: v for k, v in info.items() if k != 'tensors'}, indent=1))
