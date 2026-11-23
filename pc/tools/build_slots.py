"""Render consecutive 100 ms model inputs of one recorded route (open loop, logged poses) with the closed-loop sim code.

The output is a realistic live stream for benchmarks: one new image per camera per slot, the car's ring offsets
(-2, -1, -0.5, -0.1, 0 s), logged ego history and navigation. No GPU is used.
Usage: $PY tools/build_slots.py ROUTE_NAME OUT.npz [SECONDS]
"""
import sys, types
from pathlib import Path
import numpy as np
PC = Path(__file__).resolve().parents[1]
route_name, out = sys.argv[1], Path(sys.argv[2])
seconds = float(sys.argv[3]) if len(sys.argv) > 3 else 20.
GPU = 0
rt = types.SimpleNamespace(config=types.SimpleNamespace(cache_dir=PC/'cache'/'cpu'))
exec(open(PC/'tools/pc_sim.py').read())
run_device(CLOSEDLOOP+'/sim.py')
r = load(TRIMS+'/'+route_name)
t0 = (int(r.frames[6][0]['sof'])+3_000_000_000)//100_000_000*100_000_000+100_000_000
slots = range(t0, t0+int(seconds*1e9), 100_000_000)
images, stamps = {}, {}
need = sorted({s+int(o*1e9) for s in slots for o in (-2., -1., -.5, -.1, 0.)})
for ns in need:
  for cam in (1, 6):
    f = frame_at(r, cam, ns)
    images[(ns, cam)] = render(r, cam, f, calibration_at(r, ns), 0., 0.)
    stamps[(ns, cam)] = f['sof']//1000
keys = {ns: i for i, ns in enumerate(need)}
pix = np.stack([np.stack([images[(ns, cam)] for cam in (1, 6)]) for ns in need])  # (T, 2, 3, 432, 768)
ts = np.array([[stamps[(ns, cam)] for cam in (1, 6)] for ns in need], dtype=np.int64)
xyz, rot, nav, idx = [], [], [], []
for s in slots:
  poses = [(t, *logged_pose(r, t)) for t in range(s-2_100_000_000, s+1, 50_000_000)]
  a, b = history(poses, s); xyz.append(a[0]); rot.append(b[0])
  nav.append(navigation(tuple(n for n in r.nav if n.received <= s), s)[0])
  idx.append([keys[s+int(o*1e9)] for o in (-2., -1., -.5, -.1, 0.)])
np.savez(out, pixels=pix, stamps=ts, slot_images=np.array(idx), ego_history_xyz=np.array(xyz, dtype=np.float32),
         ego_history_rot=np.array(rot, dtype=np.float32), nav_text=np.array(nav), slots=np.array(list(slots), dtype=np.int64))
print(out, pix.shape, len(slots), 'slots')
