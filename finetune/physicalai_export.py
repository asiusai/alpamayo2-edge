"""Export NVIDIA PhysicalAI-AV six-camera samples (chunk files under LOCAL_DIR) for the 6-vs-2 camera test.

Usage (NVIDIA venv, CPU): physicalai_export.py LOCAL_DIR OUT_DIR [CHUNK] [T0_S ...]
Per clip and t0: NVIDIA's own loader (load_physical_aiavdataset: 6 cameras, 22 images, 20 history poses) plus the ground
truth future from egomotion at t0+0.1..6.4 s in the same ego frame. Writes OUT_DIR/<clip>-<t0_ms>.npz.
"""
import contextlib
import sys
from pathlib import Path

import numpy as np

from alpamayo2_edge.load_physical_aiavdataset import load_physical_aiavdataset
import physical_ai_av


def main(local_dir, out, chunk=0, t0s=(4.0, 8.0, 12.0)):
  out.mkdir(parents=True, exist_ok=True)
  avdi = physical_ai_av.PhysicalAIAVDatasetInterface(local_dir=str(local_dir), confirm_download_threshold_gb=1.)
  # physical_ai_av 0.2.0 open_file() looks only in the HF cache; files downloaded with local_dir live under local_dir.
  original = avdi.open_file

  @contextlib.contextmanager
  def open_local(filename, mode='rb', maybe_stream=False):
    path = Path(local_dir)/filename
    if path.is_file():
      with open(path, mode) as f:
        yield f
    else:
      with original(filename, mode, maybe_stream=maybe_stream) as f:
        yield f
  avdi.open_file = open_local
  clips = [c for c, row in avdi.clip_index.iterrows() if int(row['chunk']) == chunk and bool(row['clip_is_valid'])]
  print(len(clips), 'clips in chunk', chunk, flush=True)
  ok = 0
  for clip in clips:
    for t0 in t0s:
      path = out/f'{clip}-{int(t0*1000)}.npz'
      if path.exists():
        continue
      try:
        t0_us = int(t0*1e6)
        data = load_physical_aiavdataset(clip, t0_us, nav_text='Follow the road.', avdi=avdi, maybe_stream=False)
        ego = avdi.get_clip_feature(clip, avdi.features.LABELS.EGOMOTION, maybe_stream=False)
        future = ego(t0_us+np.arange(1, 65, dtype=np.int64)*100_000)
        now = ego(np.array([t0_us], dtype=np.int64))
        inv = now.pose.rotation[0].inv()
        xyz = inv.apply(np.asarray(future.pose.translation)-np.asarray(now.pose.translation)[0])
      except Exception as e:  # noqa: BLE001 - clips without coverage at t0
        print('skip', clip, t0, type(e).__name__, str(e)[:80], flush=True)
        continue
      np.savez_compressed(path, image_frames=data['image_frames'].numpy(), camera_indices=data['camera_indices'].numpy(),
                          absolute_timestamps=data['absolute_timestamps'].numpy(), ego_history_xyz=data['ego_history_xyz'].numpy(),
                          ego_history_rot=data['ego_history_rot'].numpy(), gt_xyz=xyz.astype(np.float32))
      ok += 1
      if ok % 20 == 0:
        print(ok, 'samples', flush=True)
  print('done', ok, flush=True)


if __name__ == '__main__':
  a = sys.argv[1:]
  main(Path(a[0]), Path(a[1]), int(a[2]) if len(a) > 2 else 0, tuple(float(x) for x in a[3:]) or (4.0, 8.0, 12.0))
