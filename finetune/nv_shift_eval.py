"""Recovery from a lateral or heading error on NVIDIA's own rig, by camera set (NVIDIA's torch inference path, BF16).

Usage (alpamayo-train env, one GPU): nv_shift_eval.py GPU SHIFT_DIR OUT.npz [--shard I/N] [--sets 6,2,4,L] [--nav TEXT|recorded]
SHIFT_DIR holds nv_shift.py variants (<stem>.{c,yL,yR,hL,hR}.npz). Each runs once per camera set with greedy reasoning,
zero initial noise (temperature 0) and 10 Euler steps:
  6: all six cameras   2: front wide + tele (the car)   4: front + both cross cameras   L: front + left cross + left rear
Saves '<stem>.<variant>/<set>' xyz (64,3) in the virtual ego frame.
"""
import argparse
import os
import sys
import zlib
from pathlib import Path

SETS = {'6': (0, 1, 2, 3, 5, 6), '2': (1, 6), '4': (0, 1, 2, 6), 'L': (0, 1, 3, 6), 'R': (1, 2, 5, 6), 'W': (1,)}
ap = argparse.ArgumentParser()
ap.add_argument('gpu', type=int); ap.add_argument('samples', type=Path); ap.add_argument('out', type=Path)
ap.add_argument('--shard', default='0/1'); ap.add_argument('--sets', default='6,2,4,L')
ap.add_argument('--nav', default='Follow the road.'); ap.add_argument('--steps', type=int, default=10)
ap.add_argument('--temperature', type=float, default=0.); ap.add_argument('--limit', type=int)
ap.add_argument('--draws', type=int, default=1, help='with --temperature > 0: noise draws per sample, saved as <key>#k')
ap.add_argument('--only', help='variant suffix filter, for example c')
a = ap.parse_args()
os.environ['HIP_VISIBLE_DEVICES'] = str(a.gpu)
import numpy as np  # noqa: E402
import torch  # noqa: E402

from alpamayo2_edge.checkpoint import load_model  # noqa: E402
from alpamayo2_edge.chat_template import build_conversation  # noqa: E402
from alpamayo2_edge.helper import _tokenize_messages, get_processor, to_device  # noqa: E402
from alpamayo2_edge.sample import CAMERA_PIXEL_BOUNDS  # noqa: E402

TRAIN = str(Path(__file__).resolve().parents[1]/'train'); sys.path.insert(0, TRAIN)
from alpamayo_train.model import ORIGINAL_CHECKPOINT  # noqa: E402

model, _ = load_model(str(ORIGINAL_CHECKPOINT), device='cuda', dtype='bfloat16', attn_implementation='sdpa', local_files_only=True)
model.eval()
processor = get_processor(model.tokenizer, model.config)
i, n = (int(v) for v in a.shard.split('/'))
files = sorted(f for f in a.samples.glob('*.npz') if not a.only or f.stem.endswith('.'+a.only))[:a.limit][i::n]
preds = {}


def run(z, keep, seed=43):
  nav = str(z['nav']) if a.nav == 'recorded' else a.nav
  idx = np.flatnonzero(np.isin(z['camera_indices'], keep))
  data = {'image_frames': torch.from_numpy(z['image_frames'][idx]), 'camera_indices': torch.from_numpy(z['camera_indices'][idx]),
          'absolute_timestamps': torch.from_numpy(z['absolute_timestamps'][idx]),
          'ego_history_xyz': torch.from_numpy(z['ego_history_xyz']), 'ego_history_rot': torch.from_numpy(z['ego_history_rot']),
          'nav_text': [nav], 'camera_pixel_bounds': dict(CAMERA_PIXEL_BOUNDS), 'min_pixels': 163840, 'max_pixels': 196608}
  cfg = model.config
  messages = build_conversation(data, num_tokens_per_history_traj=cfg.tokens_per_history_traj, include_camera_ids=cfg.include_camera_ids,
                                system_prompt=getattr(cfg, 'system_prompt', None))
  tokenized = _tokenize_messages(messages, processor, include_closed_thinking=True)
  prepared = to_device({'tokenized_data': tokenized, 'ego_history_xyz': data['ego_history_xyz'].unsqueeze(0),
                        'ego_history_rot': data['ego_history_rot'].unsqueeze(0)}, device='cuda')
  torch.manual_seed(seed)
  gen = torch.Generator(device='cuda').manual_seed(seed)
  with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
    xyz, _, _, _ = model.sample_trajectories_from_data(
      prepared, num_traj_sets=1, num_traj_samples=1, top_p=1.0, top_k=1, temperature=1.0, max_generation_length=256,
      diffusion_kwargs={'inference_step': a.steps, 'generator': gen, 'temperature': a.temperature}, return_extra=True)
  return xyz.reshape(-1, 64, 3)[0].float().cpu().numpy()


for k, f in enumerate(files):
  with np.load(f) as z:
    z = {key: z[key] for key in z.files}
  for name in a.sets.split(','):
    try:
      if a.draws > 1:
        for d in range(a.draws):  # a different noise vector for every sample and draw
          preds[f'{f.stem}/{name}#{d}'] = run(z, SETS[name], seed=zlib.crc32(f'{f.stem}/{d}'.encode()))
      else:
        preds[f'{f.stem}/{name}'] = run(z, SETS[name])
    except Exception as e:  # noqa: BLE001
      print('fail', f.stem, name, type(e).__name__, str(e)[:160], flush=True)
  if k % 10 == 0:
    print(k+1, '/', len(files), flush=True)
    np.savez(a.out, **preds)
np.savez(a.out, **preds)
print('done', len(preds), flush=True)
