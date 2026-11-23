"""CPU check of the targets: encode with the model's action space, decode, and report the round-trip error.

The flow-matching target is the encoded action, so the decoded round trip is the closest the expert can get.
Also reports action-space clamp hits (normalized |action| and curvature bounds).
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from alpamayo_train.data import discover, load_arrays

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('samples', type=Path, nargs='+')
parser.add_argument('--checkpoint', type=Path, default=Path(__file__).resolve().parents[1] / 'pc/data/original-checkpoint')
parser.add_argument('--limit', type=int, default=None)
args = parser.parse_args()
from alpamayo2_edge.action_space import UnicycleAccelCurvatureCourseActionSpace  # noqa: E402
cfg = dict(json.loads((args.checkpoint / 'config.json').read_text())['expert_config']['action_space_cfg'])
cfg.pop('_target_')
space = UnicycleAccelCurvatureCourseActionSpace(**cfg)
refs, skipped = discover(args.samples, limit=args.limit)
rows = []
for ref in refs:
  a = load_arrays(ref)
  hx, hr = torch.from_numpy(a['ego_history_xyz']), torch.from_numpy(a['ego_history_rot'])
  with torch.no_grad():
    t0 = space.estimate_t0_states(hx, hr)
    act = space.traj_to_action(hx, hr, torch.from_numpy(a['target_xyz'])[None], torch.from_numpy(a['target_rot'])[None], t0_states=t0)
    back, _ = space.action_to_traj(act, hx, hr, t0_states=t0)
  err = (back[0, :, :2] - torch.from_numpy(a['target_xyz'][:, :2])).norm(dim=-1).numpy()
  kappa = act[0, :, 1] * cfg['curvature_std'] + cfg['curvature_mean']
  rows.append({'stem': ref.stem, 'kind': ref.meta.get('kind'), 'dy_m': ref.meta.get('dy_m'), 'dpsi_rad': ref.meta.get('dpsi_rad'),
               'max_m': float(err.max()), 'mean_m': float(err.mean()), 'y_err_3s': float(back[0, 29, 1] - a['target_xyz'][29, 1]),
               'kappa_abs_max': float(kappa.abs().max()), 'theta0': float(t0['theta']), 'v0': float(t0['v'])})
for kind in sorted({r['kind'] for r in rows}):
  sel = [r for r in rows if r['kind'] == kind]
  mx = np.array([r['max_m'] for r in sel])
  print(json.dumps({'kind': kind, 'n': len(sel), 'roundtrip_max_m_median': float(np.median(mx)), 'p90': float(np.percentile(mx, 90)),
                    'max': float(mx.max()), 'mean_m': float(np.mean([r['mean_m'] for r in sel])),
                    'kappa_at_bound': sum(r['kappa_abs_max'] > 0.329 for r in sel)}))
worst = sorted(rows, key=lambda r: -r['max_m'])[:5]
for r in worst:
  print(json.dumps(r))
print('skipped', skipped)
