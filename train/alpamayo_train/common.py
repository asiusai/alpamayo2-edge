"""Process setup shared by the scripts: GPU selection before torch, source checks before the GPU opens."""
from __future__ import annotations

import os
import sys

DATA_ROOT = os.environ.get('ALPAMAYO_TRAIN_DATA', '/mnt/nvme-500/alpamayo-train')  # samples, runs, predictions, exports


def start(gpu: int, threads: int = 8):
  """Select the GPU, verify sources (git subprocess), then open the GPU and forbid child processes."""
  from alpamayo_train import gpu as gpu_mod
  gpu_mod.select(gpu)
  if os.path.realpath(DATA_ROOT).startswith('/mnt/nvme-500/tmp') or DATA_ROOT.startswith('/mnt/nvme-256'):
    raise RuntimeError('Invalid data root')
  from alpamayo_train.model import verify_sources
  verify_sources()
  import torch
  torch.set_num_threads(threads)
  return gpu_mod


def ade(a, b):
  import numpy as np
  d = np.linalg.norm(np.asarray(a)[..., :2] - np.asarray(b)[..., :2], axis=-1)
  return float(d.mean()), float(d[..., -1].mean() if d.ndim == 1 else d[:, -1].mean())


def lateral_at(xyz, seconds):
  """Lateral (left) coordinate at t = seconds on the 0.1 s grid (index 10*s - 1)."""
  import numpy as np
  return float(np.asarray(xyz)[int(round(seconds * 10)) - 1, 1])


def log(*args):
  print(*args, flush=True, file=sys.stdout)
