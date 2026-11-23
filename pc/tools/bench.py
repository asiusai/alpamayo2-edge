"""Throughput benchmark on one PC GPU. Run one process per GPU (both at once for the dual-GPU total).

Modes: sweep   - every call has 10 new images (input sweeps, variant renders)
       rolling - every call adds one new image per camera; 8 history images stay cached (closed-loop simulation)
       cached  - all 10 images cached (repeated input; prefill, reasoning and diffusion only)
       live    - the car worker's pipeline on the rolling stream: between calls (off the clock) the early prefill parts
                 of the next call run (prefill_ahead); on the clock, prime() encodes the new image pair, then predict()
Usage: bench.py --gpu N --mode sweep --calls 48 [--out results.json]
"""
import argparse, json, os, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import samples


def slot_stream(path):
  """Consecutive 100 ms slots rendered by build_slots.py (one route, logged poses)."""
  with np.load(path) as f:
    z = {k: f[k] for k in f.files}  # NpzFile reads the whole array on every key access
  objs = [[np.ascontiguousarray(z['pixels'][i, c][None]) for c in (0, 1)] for i in range(len(z['pixels']))]
  for k in range(len(z['slots'])):
    idx = z['slot_images'][k]
    s = {'image_frames': np.ascontiguousarray(np.stack([objs[i][0] for i in idx]+[objs[i][1] for i in idx])),
         'camera_indices': np.array([1]*5+[6]*5, dtype=np.int64),
         'absolute_timestamps': np.concatenate([z['stamps'][idx, 0], z['stamps'][idx, 1]]),
         'ego_history_xyz': z['ego_history_xyz'][k][None], 'ego_history_rot': z['ego_history_rot'][k][None], 'nav_text': [str(z['nav_text'][k])]}
    s['frame_objects'] = [objs[i][0] for i in idx]
    s['anchors'] = [(objs[idx[-1]][0], {'view': 1, 'capture_ns': int(z['stamps'][idx[-1], 0])*1000}),
                    (objs[idx[-1]][1], {'view': 6, 'capture_ns': int(z['stamps'][idx[-1], 1])*1000})]
    yield s


def stream(mode, data):
  if mode in ('sweep', 'cached'):
    k = 0
    while True:
      yield data[0] if mode == 'cached' else data[k % len(data)]
      k += 1
  # Rolling: one camera-1 and one camera-6 image per call, five-image history per camera.
  # 48 samples x 5 images per camera, plus mirrored copies. Repeated source frames are removed. The stream must not repeat
  # an image inside one history window (duplicate features are invalid JIT inputs in the early-prefill stitcher).
  frames = {c: [s['image_frames'][i] for s in data for i in (range(5) if c == 1 else range(5, 10))] for c in (1, 6)}
  frames = {c: f+[np.ascontiguousarray(x[..., ::-1]) for x in f] for c, f in frames.items()}
  frames = {c: list({x.tobytes(): x for x in f}.values()) for c, f in frames.items()}  # samples repeat some source frames
  k = 5
  while True:
    s = dict(data[k % len(data)])
    idx = [(k-4+j) % min(len(f) for f in frames.values()) for j in range(5)]
    s['image_frames'] = np.ascontiguousarray(np.stack([frames[1][i] for i in idx]+[frames[6][i] for i in idx]))
    base = np.array([-2_000_000, -1_000_000, -500_000, -100_000, 0], dtype=np.int64)+k*100_000+10_000_000
    s['absolute_timestamps'] = np.concatenate([base, base])
    s['frame_objects'] = [frames[1][i] for i in idx]  # camera-1 history objects (early prefill parts read the first four)
    s['anchors'] = [(frames[1][idx[-1]], {'view': 1, 'capture_ns': int(base[-1])*1000}), (frames[6][idx[-1]], {'view': 6, 'capture_ns': int(base[-1])*1000})]
    yield s
    k += 1


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--gpu', type=int, required=True)
  parser.add_argument('--mode', choices=('sweep', 'rolling', 'cached', 'live'), default='sweep')
  parser.add_argument('--calls', type=int, default=48)
  parser.add_argument('--warm', type=int, default=24)
  parser.add_argument('--out', type=Path)
  parser.add_argument('--verbose', action='store_true')
  parser.add_argument('--slots', type=Path, help='build_slots.py output: realistic consecutive slots for rolling/live')
  parser.add_argument('--start-at', type=float, default=0, help='wall-clock time to start measuring (aligns two GPUs)')
  args = parser.parse_args()
  import alpamayo_pc
  rt = alpamayo_pc.load(args.gpu)
  rt.stage_sync = False
  from tinygrad import Device
  stats = Path(f'/sys/class/kfd/kfd/proc/{os.getpid()}/stats_{Device["AMD"].iface.gpu_id}/evicted_ms')
  def evicted():
    try: return int(stats.read_text())
    except OSError: return -1
  evicted_after_load = evicted()
  if args.slots:
    it = slot_stream(args.slots)
  else:
    data = [samples.load(p) for p in samples.karel_samples()]
    it = stream('rolling' if args.mode == 'live' else args.mode, data)
  def call(s):
    sample = {k: v for k, v in s.items() if k not in ('frame_objects', 'anchors')}
    if args.mode == 'live':
      rt.prime(s['anchors'], time.monotonic_ns()+600_000_000_000, wait=False)  # GPU encodes while the CPU prepares
    return rt.predict(sample)
  def ahead(s):
    if args.mode == 'live':
      rt.prefill_ahead(s['frame_objects'][:4])
      rt.synchronize()
  pending = next(it)
  for _ in range(args.warm):
    s, pending = pending, next(it)
    call(s); ahead(pending)
  rt.synchronize()
  while time.time() < args.start_at:
    time.sleep(.01)
  times, tokens, misses = [], [], []
  begin = time.perf_counter()
  ahead_ms, ahead_parts = [], []
  for _ in range(args.calls):
    s, pending = pending, next(it)
    t = time.perf_counter(); p = call(s); rt.synchronize(); times.append((time.perf_counter()-t)*1e3)
    ahead_parts.append(rt.metrics.prefix_ahead)
    t = time.perf_counter(); ahead(pending); ahead_ms.append((time.perf_counter()-t)*1e3)
    tokens.append(p['generated_tokens']); misses.append(rt.metrics.vision_misses)
    if args.verbose:
      print(len(times), f'{times[-1]:.1f} ms', rt.metrics.vision_misses, p['generated_tokens'], flush=True)
  wall = time.perf_counter()-begin
  result = dict(gpu=args.gpu, mode=args.mode, calls=args.calls, wall_s=wall, calls_per_s=args.calls/wall,
                ms_p50=float(np.median(times)), ms_p90=float(np.percentile(times, 90)), ms_mean=float(np.mean(times)),
                vision_misses_mean=float(np.mean(misses)), tokens_mean=float(np.mean(tokens)),
                prefix_ahead_mean=float(np.mean(ahead_parts)), ahead_ms_p50=float(np.median(ahead_ms)),
                kfd_evicted_ms_after_load=evicted_after_load, kfd_evicted_ms_end=evicted())
  print(json.dumps(result), flush=True)
  if args.out:
    args.out.write_text(json.dumps(result))
  rt.close()


if __name__ == '__main__':
  main()
