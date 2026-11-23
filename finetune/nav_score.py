"""Score nav_test.sh: does the plan follow a turn instruction, and how long is the generated reasoning?"""
import json, math, numpy as np
from pathlib import Path
H = Path('/mnt/nvme-500/alpamayo-train/nvdistill/samples/heldout'); O = Path('/mnt/nvme-500/alpamayo-train/predictions/nav')
def heading(p, i):
  d = p[i]-p[i-3]; return math.atan2(d[1], d[0])
def load(name):
  f = O/f'{name}.npz'
  if not f.exists(): return None, None
  z = np.load(f); rows = {}
  j = f.with_suffix('.json')
  if j.exists():
    try: rows = {r['stem']: r for r in json.loads(j.read_text()).get('rows', [])}
    except Exception: rows = {}
  return {k: z[k] for k in z.files}, rows
print(f'{"run":22s} {"n":>3s} {"turn ratio 3s":>13s} {"6.4s":>6s} {"end lateral vs human m":>23s} {"reasoning words":>16s} {"nominal":>8s}')
for model in ('stock', 'lat10'):
  for name in ('left-nav', 'left-cs', 'right-nav', 'right-cs', 'straight-far', 'straight-cs'):
    p, rows = load(f'{model}-{name}')
    if p is None: continue
    a = []
    for k, xyz in p.items():
      g = np.load(H/f'{k}.npz')['gt_xyz']
      a.append((heading(g, 29), heading(xyz, 29), heading(g, 63), heading(xyz, 63), xyz[63, 1]-g[63, 1]))
    a = np.array(a)
    r3 = np.sum(a[:, 1]*a[:, 0])/np.sum(a[:, 0]**2) if 'straight' not in name else float('nan')
    r6 = np.sum(a[:, 3]*a[:, 2])/np.sum(a[:, 2]**2) if 'straight' not in name else float('nan')
    words = [len(str(r.get('reasoning', '')).replace('<|traj_future_start|>', '').split()) for r in rows.values()]
    nominal = np.mean(['NOMINAL' in str(r.get('reasoning', '')) for r in rows.values()]) if rows else float('nan')
    print(f'{model+"-"+name:22s} {len(a):3d} {r3:13.2f} {r6:6.2f} {np.median(a[:, 4]):+10.2f} (mean {a[:, 4].mean():+.2f}) {np.median(words) if words else float("nan"):9.0f} max {max(words) if words else 0:3d} {nominal:8.2f}')
