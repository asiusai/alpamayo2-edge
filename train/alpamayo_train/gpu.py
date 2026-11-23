"""GPU selection and the no-child-process rule.

A child process forked from a process that holds a KFD device inherits the render-node file. When the child closes it
(exec, exit, close_fds), amdgpu_flush waits on the VM and KFD evicts and restores every queue of the parent through
the gfx11 MES firmware. Under load this reset the GPUs on this PC (see alpamayo2-edge/PC-RUNTIME.md, GPU faults).
So: select the GPU through ROCR_VISIBLE_DEVICES before torch initializes HIP, and refuse every fork, spawn and
subprocess after the GPU opens. Data loading uses threads only.
"""
import os
import sys
import traceback

# ROCr agent index -> PCI address on this PC (KFD topology location_id 0x300 and 0xc00).
GPU_PCI = {0: '0000:03:00.0', 1: '0000:0c:00.0'}
_opened = False


def select(gpu: int) -> None:
  """Expose exactly one discrete GPU to this process. Call before importing torch."""
  if 'torch' in sys.modules and getattr(sys.modules['torch'], 'cuda', None) is not None:
    import torch
    if torch.cuda.is_initialized():
      raise RuntimeError('Select the GPU before HIP initializes')
  if gpu not in GPU_PCI:
    raise ValueError(f'Unknown GPU {gpu}; use one of {sorted(GPU_PCI)}')
  os.environ['ROCR_VISIBLE_DEVICES'] = str(gpu)
  os.environ.pop('HIP_VISIBLE_DEVICES', None)
  os.environ.pop('CUDA_VISIBLE_DEVICES', None)


def _refuse(*args, **kwargs):
  raise RuntimeError('Child process requested after the GPU opened (KFD eviction / GPU reset risk):\n'
                     + ''.join(traceback.format_stack(limit=10)))


def open_device(gpu: int):
  """Open the selected GPU as cuda:0, check its PCI address, then forbid child processes."""
  global _opened
  import torch
  device = torch.device('cuda', 0)
  props = torch.cuda.get_device_properties(device)
  bus = int(GPU_PCI[gpu].split(':')[1], 16)
  if torch.cuda.device_count() != 1 or props.pci_bus_id != bus or props.gcnArchName.split(':')[0] != 'gfx1100':
    raise RuntimeError(f'GPU {gpu} mismatch: count={torch.cuda.device_count()} bus={props.pci_bus_id} arch={props.gcnArchName}')
  torch.cuda.set_device(device)
  torch.empty(1, device=device)
  _opened = True
  import _posixsubprocess
  os.fork = _refuse
  if hasattr(os, 'forkpty'):
    os.forkpty = _refuse
  os.posix_spawn = _refuse
  os.posix_spawnp = _refuse
  _posixsubprocess.fork_exec = _refuse
  return device


def evicted_ms() -> dict:
  """KFD queue eviction time of this process per GPU node; it must stay 0."""
  root = f'/sys/class/kfd/kfd/proc/{os.getpid()}'
  out = {}
  try:
    for name in os.listdir(root):
      if name.startswith('stats_'):
        with open(f'{root}/{name}/evicted_ms') as f:
          out[name[6:]] = int(f.read().strip() or 0)
  except OSError:
    pass
  return out
