"""Compare predictions of two runs (run_samples.py outputs, or Karel plan files via --plans) on the same inputs.

Usage: compare_runs.py A.json B.json        (both run_samples outputs)
       compare_runs.py A.json --plans        (B = the Karel plan JSONs named by each sample)
Prints ADE/FDE (m), maximum heading difference (deg) over 6.4 s, and token agreement.
"""
import json, sys
from pathlib import Path
import numpy as np

KAREL = Path('/home/john/alpamayo2-edge/pc/data/karel-outputs')  # copy of /mnt/nvme-500/tmp/alpamayo-karel-outputs-20261121


def yaw(rot):
  rot = np.asarray(rot)[0]
  return np.unwrap(np.arctan2(rot[:, 1, 0], rot[:, 0, 0]))


def plan_row(name):
  stem = name.removesuffix('.npz')
  session, query = stem.rsplit('-', 1)
  p = json.loads((KAREL/session/f'{query}.json').read_text())['prediction']
  return dict(xyz=p['xyz'], rot=p['rot'], tokens=p['generated_token_ids'], reasoning=p['reasoning'], ms=p['runtime_ms'])


def main(a, b):
  A = json.loads(Path(a).read_text())['rows']
  B = [plan_row(r['sample']) for r in A] if b == '--plans' else json.loads(Path(b).read_text())['rows']
  ade, fde, hd, tok = [], [], [], []
  for x, y in zip(A, B, strict=True):
    d = np.linalg.norm(np.asarray(x['xyz'])[0]-np.asarray(y['xyz'])[0], axis=-1)
    ade.append(d.mean()); fde.append(d[-1]); hd.append(np.degrees(np.abs(yaw(x['rot'])-yaw(y['rot'])).max())); tok.append(x['tokens'] == y['tokens'])
  ade, fde, hd = map(np.array, (ade, fde, hd))
  same = np.array(tok)
  print(f'n={len(A)} tokens equal {same.sum()}/{len(A)}')
  for label, m in (('all', np.ones_like(same)), ('same tokens', same)):
    if m.any():
      print(f'  {label:11s} ADE median {np.median(ade[m]):.3f} p90 {np.percentile(ade[m], 90):.3f} max {ade[m].max():.3f} | '
            f'FDE median {np.median(fde[m]):.3f} p90 {np.percentile(fde[m], 90):.3f} max {fde[m].max():.3f} | '
            f'heading median {np.median(hd[m]):.2f} p90 {np.percentile(hd[m], 90):.2f} max {hd[m].max():.2f} deg')
  print(f'  exact xyz {sum(np.array_equal(x["xyz"], y["xyz"]) for x, y in zip(A, B))}/{len(A)}; median ms {np.median([x["ms"] for x in A]):.1f} vs {np.median([y["ms"] for y in B]):.1f}')


if __name__ == '__main__':
  main(*sys.argv[1:3])
