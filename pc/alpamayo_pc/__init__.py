"""Selected Alpamayo runtime on the PC's RX 7900 XTX GPUs. One process per GPU; see pc/README.md."""
from pathlib import Path
from .port import PC, install

ASSETS = PC/'assets'


def config(gpu, **overrides):
  values = dict(checkpoint=ASSETS/'checkpoint', fp8_cache=ASSETS/'fp8-cache', prefill_calibration=ASSETS/'fixed/prefill-scales.json',
                initial_noise=ASSETS/'fixed/initial_noise.npy', cache_dir=PC/'cache'/f'gpu{int(gpu)}', device_timeout_ms=120000)
  values.update(overrides)
  return values


def load(gpu, warmup=False, **overrides):
  """Return a selected `Runtime` on GPU `gpu` (0 or 1) with the same `predict(sample)` contract as the car."""
  values = config(gpu, **overrides)
  runtime = install(gpu, values['cache_dir'])
  rt = runtime.Runtime(runtime.Config(**values))
  if warmup:
    rt.warmup()
  return rt
