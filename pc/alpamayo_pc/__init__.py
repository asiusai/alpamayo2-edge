"""Selected Alpamayo runtime on the PC's RX 7900 XTX GPUs. One process per GPU; see pc/README.md."""
from pathlib import Path
from .port import PC, install

ASSETS = PC/'assets'


def config(gpu, zero_noise=False, **overrides):
  """Config fields of the selected runtime with the PC asset paths.

  Defaults keep the pinned recipe (1 Euler step, the pinned car noise file), so results compare with the stored parity runs.
  The car uses zero initial noise since 2026-11-22 (`zero_noise=True`). A pinned fine-tune (artifacts.json `fp8_finetunes`)
  runs with `fp8_cache=<its FP8 cache>` and its own `cache_dir`. This snapshot accepts `diffusion_steps` 1 to 3.
  """
  values = dict(checkpoint=ASSETS/'checkpoint', fp8_cache=ASSETS/'fp8-cache', prefill_calibration=ASSETS/'fixed/prefill-scales.json',
                initial_noise=ASSETS/'fixed/initial_noise.npy', cache_dir=PC/'cache'/f'gpu{int(gpu)}', device_timeout_ms=120000)
  values.update(overrides)
  if values.get('compute_profile'):
    raise ValueError('compute_profile is a Chestnut SMU setting; the PC GPUs keep their own power settings')
  if zero_noise:
    import numpy as np
    path = Path(values['cache_dir'])/'initial_noise-zero.npy'
    if not path.exists():
      path.parent.mkdir(parents=True, exist_ok=True)
      np.save(path, np.zeros((1, 64, 2), dtype=np.float32))
    values['initial_noise'] = path
  return values


def load(gpu, warmup=False, zero_noise=False, **overrides):
  """Return a selected `Runtime` on GPU `gpu` (0 or 1) with the same `predict(sample)` contract as the car."""
  values = config(gpu, zero_noise, **overrides)
  runtime = install(gpu, values['cache_dir'])
  rt = runtime.Runtime(runtime.Config(**values))
  if warmup:
    rt.warmup()
  return rt
