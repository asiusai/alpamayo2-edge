"""Stage B of the six-to-two camera distillation set: six-camera teacher plans for the two-camera student samples.

Usage (alpamayo-train env, one GPU): nv_distill_teacher.py GPU SPOOL_DIR OUT_DIR [--shard I/N]
Consumes nv_distill_decode.py spool files. The teacher is the stock model on NVIDIA's torch path (BF16) with all six
cameras, greedy reasoning, zero initial noise, 10 Euler steps and the car's navigation text. Writes a training sample in
the alpamayo_train format to OUT_DIR/{train,heldout}/<clip>-<t0_ms>-<c|r>.npz (+ .json): the two front cameras at the
car's prepared size, the same ego history, and the teacher's trajectory as the target. One clip in eight is heldout.
Stops when SPOOL_DIR/../decode.done exists and the spool is empty.
"""
import argparse
import json
import os
import sys
import time
import zlib
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument('gpu', type=int); ap.add_argument('spool', type=Path); ap.add_argument('out', type=Path)
ap.add_argument('--shard', default='0/1')
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

NAV = 'Continue straight.'
model, _ = load_model(str(ORIGINAL_CHECKPOINT), device='cuda', dtype='bfloat16', attn_implementation='sdpa', local_files_only=True)
model.eval()
processor = get_processor(model.tokenizer, model.config)
shard, shards = (int(v) for v in a.shard.split('/'))
for split in ('train', 'heldout'):
  (a.out/split).mkdir(parents=True, exist_ok=True)


def teacher(z):
  data = {'image_frames': torch.from_numpy(z['image_frames']), 'camera_indices': torch.from_numpy(z['camera_indices']),
          'absolute_timestamps': torch.from_numpy(z['absolute_timestamps']),
          'ego_history_xyz': torch.from_numpy(z['ego_history_xyz']), 'ego_history_rot': torch.from_numpy(z['ego_history_rot']),
          'nav_text': [NAV], 'camera_pixel_bounds': dict(CAMERA_PIXEL_BOUNDS), 'min_pixels': 163840, 'max_pixels': 196608}
  cfg = model.config
  messages = build_conversation(data, num_tokens_per_history_traj=cfg.tokens_per_history_traj, include_camera_ids=cfg.include_camera_ids,
                                system_prompt=getattr(cfg, 'system_prompt', None))
  tokenized = _tokenize_messages(messages, processor, include_closed_thinking=True)
  prepared = to_device({'tokenized_data': tokenized, 'ego_history_xyz': data['ego_history_xyz'].unsqueeze(0),
                        'ego_history_rot': data['ego_history_rot'].unsqueeze(0)}, device='cuda')
  torch.manual_seed(43)
  with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
    xyz, rot, _, extra = model.sample_trajectories_from_data(
      prepared, num_traj_sets=1, num_traj_samples=1, top_p=1.0, top_k=1, temperature=1.0, max_generation_length=256,
      diffusion_kwargs={'inference_step': 10, 'temperature': 0.}, return_extra=True)
  text = extra.get('cot') if isinstance(extra, dict) else ''
  while isinstance(text, (list, tuple, np.ndarray)) and len(text):
    text = text[0]
  return xyz.reshape(-1, 64, 3)[0].float().cpu().numpy(), rot.reshape(-1, 64, 3, 3)[0].float().cpu().numpy(), str(text)[:300]


done = 0
while True:
  files = [f for f in sorted(a.spool.glob('*.npz')) if zlib.crc32(f.name.encode()) % shards == shard]
  if not files:
    if (a.spool.parent/'decode.done').exists():
      break
    time.sleep(3)
    continue
  for f in files:
    try:
      with np.load(f) as z:
        z = {k: z[k] for k in z.files}
      xyz, rot, text = teacher(z)
    except Exception as e:  # noqa: BLE001
      print('fail', f.name, type(e).__name__, str(e)[:160], flush=True)
      f.unlink(missing_ok=True)
      continue
    stem, key = f.name[:-4].rsplit('.', 1)
    clip, t0_ms = stem.rsplit('-', 1)
    split = 'heldout' if zlib.crc32(clip.encode()) % 8 == 0 else 'train'
    dy, dpsi = (float(v) for v in z['shift'])
    name = f'{stem}-{key}'
    tmp = a.out/split/f'{name}.tmp'
    with open(tmp, 'wb') as fh:
      np.savez_compressed(fh, image_frames=z['student_frames'], camera_indices=np.array([1]*5+[6]*5, np.int64),
                          absolute_timestamps=z['student_timestamps'].astype(np.int64),
                          ego_history_xyz=z['ego_history_xyz'].astype(np.float32), ego_history_rot=z['ego_history_rot'].astype(np.float32),
                          nav_text=np.array(NAV), target_xyz=xyz.astype(np.float32), target_rot=rot.astype(np.float32),
                          gt_xyz=z['gt_xyz'].astype(np.float32), gt_rot=z['gt_rot'].astype(np.float32))
    (a.out/split/f'{name}.json').write_text(json.dumps({
      'route': clip, 'query_ns': int(t0_ms)*1_000_000, 'dy_m': dy, 'dpsi_rad': dpsi, 'kind': 'clean' if key == 'c' else 'recovery',
      'speed_m_s': float(z['speed']), 'nav_text': NAV, 'source': 'physicalai', 'chunk': int(z['chunk']), 'teacher_reasoning': text}))
    tmp.rename(a.out/split/f'{name}.npz')
    f.unlink()
    done += 1
    if done % 20 == 0:
      print(done, 'samples', flush=True)
print('teacher done', done, flush=True)
