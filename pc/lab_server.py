"""Persistent Alpamayo lab on one PC GPU: load the selected runtime once, then execute snippets.

Same snippet contract as tools/alpamayo-car/bench/lab_server.py on the device: each LAB/in/*.py runs in a shared
namespace with `rt` (selected Runtime), `np`, `Tensor`, `dtypes`, `timeit(fn, n)`, `lab_import(name)` and `GPU`.
Output goes to LAB/out/<name>.txt, ending with `== done` or `== error`.
Usage: python lab_server.py --gpu 0 [--lab DIR] [--warmup] [--diffusion-steps N]
"""
import argparse, contextlib, importlib, io, sys, time, traceback
from pathlib import Path

PC = Path(__file__).resolve().parent


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--gpu', type=int, required=True)
  parser.add_argument('--lab', type=Path)
  parser.add_argument('--warmup', action='store_true', help='compile and capture every steady-state graph shape first')
  parser.add_argument('--diffusion-steps', type=int, default=1)
  args = parser.parse_args()
  lab = args.lab or PC/'lab'/f'gpu{args.gpu}'
  for sub in ('in', 'out', 'src'):
    (lab/sub).mkdir(parents=True, exist_ok=True)
  (lab/'ready').unlink(missing_ok=True)
  sys.path.insert(0, str(lab/'src'))
  sys.path.insert(0, str(PC/'tools'))
  started = time.monotonic()
  import alpamayo_pc
  rt = alpamayo_pc.load(args.gpu, warmup=args.warmup, diffusion_steps=args.diffusion_steps)
  import numpy as np
  from tinygrad import Tensor, dtypes

  def timeit(fn, n=20, warm=3):
    for _ in range(warm):
      fn()
    rt.model.synchronize()
    times = []
    for _ in range(n):
      t = time.perf_counter()
      fn()
      rt.model.synchronize()
      times.append((time.perf_counter()-t)*1e3)
    a = np.array(times)
    return f'median {np.median(a):.3f} ms  min {a.min():.3f}  p90 {np.percentile(a, 90):.3f}  (n={n})'

  def lab_import(name):
    module = sys.modules.get(name)
    return importlib.reload(module) if module is not None else importlib.import_module(name)

  ns = {'rt': rt, 'np': np, 'Tensor': Tensor, 'dtypes': dtypes, 'timeit': timeit, 'lab_import': lab_import, 'time': time, 'GPU': args.gpu}
  (lab/'ready').write_text(f'{time.monotonic()-started:.1f}\n')
  print(f'lab ready on GPU {args.gpu} after {time.monotonic()-started:.1f} s: {lab}', flush=True)
  try:
    while True:
      jobs = sorted((lab/'in').glob('*.py'))
      if not jobs:
        time.sleep(.02)
        continue
      job = jobs[0]
      code = job.read_text()
      job.unlink()
      buf = io.StringIO()
      status = 'done'
      t = time.monotonic()
      with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        try:
          exec(compile(code, job.name, 'exec'), ns)
        except BaseException:  # noqa: BLE001
          traceback.print_exc()
          status = 'error'
      out = lab/'out'/(job.stem+'.txt')
      tmp = out.with_suffix('.tmp')
      tmp.write_text(buf.getvalue()+f'== {status} {time.monotonic()-t:.1f} s\n')
      tmp.replace(out)
  finally:
    rt.close()


if __name__ == '__main__':
  main()
