"""6-camera vs 2-front-camera predictions on exported PhysicalAI-AV samples (NVIDIA's own torch inference path, BF16).

Usage (alpamayo-train env, one GPU): physicalai_eval.py GPU SAMPLE_DIR OUT.npz [--limit N] [--steps 10]
Each sample runs twice with deterministic reasoning (top_k 1) and a fixed diffusion seed: all six cameras (22 images) and
front wide + tele only (10 images, the car's layout). Saves '<stem>/6' and '<stem>/2' xyz (64,3) and the reasoning text.
"""
import argparse
import json
import os
import sys
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument('gpu', type=int); ap.add_argument('samples', type=Path); ap.add_argument('out', type=Path)
ap.add_argument('--limit', type=int); ap.add_argument('--steps', type=int, default=10)
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
files = sorted(a.samples.glob('*.npz'))[:a.limit]
preds, texts = {}, {}


def run(z, keep):
  idx = np.flatnonzero(np.isin(z['camera_indices'], keep))
  data = {'image_frames': torch.from_numpy(z['image_frames'][idx]), 'camera_indices': torch.from_numpy(z['camera_indices'][idx]),
          'absolute_timestamps': torch.from_numpy(z['absolute_timestamps'][idx]),
          'ego_history_xyz': torch.from_numpy(z['ego_history_xyz']), 'ego_history_rot': torch.from_numpy(z['ego_history_rot']),
          'nav_text': ['Follow the road.'], 'camera_pixel_bounds': dict(CAMERA_PIXEL_BOUNDS), 'min_pixels': 163840,
          'max_pixels': 196608}
  # prepare_model_inputs() without its 22-frame check (the car feeds 10 frames): the loader's frames are already in canonical
  # camera-then-time order, so both variants go through the same prompt and tokenizer path.
  cfg = model.config
  messages = build_conversation(data, num_tokens_per_history_traj=cfg.tokens_per_history_traj, include_camera_ids=cfg.include_camera_ids,
                                system_prompt=getattr(cfg, 'system_prompt', None))
  tokenized = _tokenize_messages(messages, processor, include_closed_thinking=True)
  prepared = to_device({'tokenized_data': tokenized, 'ego_history_xyz': data['ego_history_xyz'].unsqueeze(0),
                        'ego_history_rot': data['ego_history_rot'].unsqueeze(0)}, device='cuda')
  torch.manual_seed(43)
  gen = torch.Generator(device='cuda').manual_seed(43)
  with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
    xyz, _, _, extra = model.sample_trajectories_from_data(prepared, num_traj_sets=1, num_traj_samples=1, top_p=1.0, top_k=1,
                                                           temperature=1.0, max_generation_length=256,
                                                           diffusion_kwargs={'inference_step': a.steps, 'generator': gen}, return_extra=True)
  text = extra.get('cot') if isinstance(extra, dict) else None
  return xyz.reshape(-1, 64, 3)[0].float().cpu().numpy(), str(text)[:200] if text is not None else ''


for n, f in enumerate(files):
  with np.load(f) as z:
    z = {k: z[k] for k in z.files}
  for name, keep in (('6', (0, 1, 2, 3, 5, 6)), ('2', (1, 6))):
    try:
      preds[f'{f.stem}/{name}'], texts[f'{f.stem}/{name}'] = run(z, keep)
    except Exception as e:  # noqa: BLE001
      print('fail', f.stem, name, type(e).__name__, str(e)[:120], flush=True)
  if n % 10 == 0:
    print(n+1, '/', len(files), flush=True)
np.savez(a.out, **preds)
a.out.with_suffix('.json').write_text(json.dumps(texts))
print('done', len(preds), flush=True)
