"""One process per GPU. The parent never imports torch or opens a GPU; it only starts and waits for its children.

Children start before any GPU is open (in the parent or in themselves), which keeps the no-fork-with-GPU-open rule.
"""
from __future__ import annotations

import os
import signal
import socket
import subprocess
import sys


def child_env() -> dict | None:
  """Rank settings when this process is a launched child, else None."""
  if 'ALPAMAYO_RANK' not in os.environ:
    return None
  return {'rank': int(os.environ['ALPAMAYO_RANK']), 'world': int(os.environ['ALPAMAYO_WORLD']),
          'gpu': int(os.environ['ALPAMAYO_GPU'])}


def run_children(gpus: list[int]) -> int:
  if 'torch' in sys.modules:
    raise RuntimeError('The launcher parent must not import torch')
  with socket.socket() as s:
    s.bind(('127.0.0.1', 0))
    port = s.getsockname()[1]
  procs = []
  for rank, gpu in enumerate(gpus):
    env = dict(os.environ, ALPAMAYO_RANK=str(rank), ALPAMAYO_WORLD=str(len(gpus)), ALPAMAYO_GPU=str(gpu),
               MASTER_ADDR='127.0.0.1', MASTER_PORT=str(port))
    procs.append(subprocess.Popen([sys.executable, *sys.argv], env=env))

  def forward(signum, _frame):
    for p in procs:
      if p.poll() is None:
        p.send_signal(signum)
  signal.signal(signal.SIGINT, forward)
  signal.signal(signal.SIGTERM, forward)
  codes = [p.wait() for p in procs]
  return max(abs(c) for c in codes)


def init_group(ctx: dict):
  """Gloo process group: gradient all-reduce goes through host memory, no RCCL peer setup."""
  import torch.distributed as dist
  if ctx['world'] > 1:
    dist.init_process_group('gloo', rank=ctx['rank'], world_size=ctx['world'])


def all_reduce_mean(tensors, stager):
  """Average GPU tensors across ranks: one flat D2H and H2D through the pinned stager, Gloo on the host."""
  import torch
  import torch.distributed as dist
  if not dist.is_initialized() or dist.get_world_size() == 1:
    return
  device = tensors[0].device
  flat = stager.d2h(torch.cat([t.detach().reshape(-1).float() for t in tensors]))
  dist.all_reduce(flat)
  flat /= dist.get_world_size()
  flat = stager.h2d(flat, device)
  offset = 0
  for t in tensors:
    n = t.numel()
    t.copy_(flat[offset:offset + n].view_as(t))
    offset += n


def gather_objects(obj):
  import torch.distributed as dist
  if not dist.is_initialized() or dist.get_world_size() == 1:
    return [obj]
  out = [None] * dist.get_world_size()
  dist.all_gather_object(out, obj)
  return out


def barrier():
  import torch.distributed as dist
  if dist.is_initialized():
    dist.barrier()
