"""Run saved model inputs (NPZ or six-NPY folders) through one runtime and save every prediction.

PC GPU:   $PY tools/run_samples.py --gpu 0 OUT.json SAMPLE...
Chestnut: python tools/run_samples.py --chestnut-cache DIR [--config live.json] OUT.json SAMPLE...
          Stock selected runtime, no PC adapter. DIR must hold the staged gfx1200 firmware. Asset paths come from the
          config's `inference` section (car) or default to pc/assets. Never run while another process owns Chestnut.
Inputs run in the given order with the fixed car noise, so decoder drafts match between runtimes.
--rounds N repeats the list and saves the last round (earlier rounds compile and capture graphs).
--zero-noise uses zero initial noise, as the car does since 2026-11-22.
"""
import argparse, json, sys, time
from pathlib import Path
PC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PC/'tools'))


def main():
  parser = argparse.ArgumentParser()
  target = parser.add_mutually_exclusive_group(required=True)
  target.add_argument('--gpu', type=int)
  target.add_argument('--chestnut-cache', type=Path)
  parser.add_argument('--config', type=Path, help='car config JSON with an inference section (Chestnut only)')
  parser.add_argument('--warmup', action='store_true')
  parser.add_argument('--diffusion-steps', type=int, default=1, help='Euler steps of the trajectory expert (PC GPU only)')
  parser.add_argument('--rounds', type=int, default=1)
  parser.add_argument('--zero-noise', action='store_true')
  parser.add_argument('--cache-dir', type=Path, help='PC GPU cache directory (default pc/cache/gpuN)')
  parser.add_argument('out', type=Path)
  parser.add_argument('samples', nargs='+', type=Path)
  args = parser.parse_args()
  if args.gpu is not None:
    import alpamayo_pc
    extra = {} if args.cache_dir is None else {'cache_dir': args.cache_dir}
    rt = alpamayo_pc.load(args.gpu, diffusion_steps=args.diffusion_steps, **extra)
  else:
    from openpilot.selfdrive.alpamayo.selected import Config, Runtime
    if args.config:
      inf = json.loads(args.config.read_text())['inference']
      paths = {k: inf[k] for k in ('checkpoint', 'fp8_cache', 'prefill_calibration', 'initial_noise')}
    else:
      assets = PC/'assets'
      paths = dict(checkpoint=assets/'checkpoint', fp8_cache=assets/'fp8-cache', prefill_calibration=assets/'fixed/prefill-scales.json',
                   initial_noise=assets/'fixed/initial_noise.npy')
    rt = Runtime(Config(**paths, cache_dir=args.chestnut_cache))
  import samples
  if args.warmup:
    rt.warmup()
  import numpy as np
  noise = np.zeros((1, 64, 2), dtype=np.float32) if args.zero_noise else None
  rows = []
  try:
    for rnd in range(args.rounds):
      rows = []
      for path in args.samples:
        s = samples.load(path)
        t = time.perf_counter(); p = rt.predict(s, noise); rt.synchronize(); ms = (time.perf_counter()-t)*1e3
        rows.append(dict(sample=path.name, ms=ms, xyz=p['xyz'], rot=p['rot'], tokens=p['generated_token_ids'], reasoning=p['reasoning'],
                         timings=p['timings_ms']))
        print(rnd, path.name, f'{ms:.1f} ms', p['generated_tokens'], repr(p['reasoning'][:60]), flush=True)
    ms = np.array([r['ms'] for r in rows])
    print(f'last round: n={len(ms)} median {np.median(ms):.1f} ms mean {ms.mean():.1f} p90 {np.percentile(ms, 90):.1f}', flush=True)
    for key in rows[0]['timings']:
      print(f'  {key:18s} {np.median([r["timings"][key] for r in rows]):8.1f}', flush=True)
  finally:
    args.out.write_text(json.dumps(dict(manifest=rt.manifest, rows=rows), default=str))
    rt.close()


if __name__ == '__main__':
  main()
