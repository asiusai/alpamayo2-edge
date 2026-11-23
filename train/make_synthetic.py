"""Build samples in the training format from the 48 Karel parity inputs and their Chestnut outputs.

Targets and baselines are the Chestnut predictions; reasoning_token_ids are the Chestnut tokens (they end with the anchor).
"""
import argparse
import json
from pathlib import Path

import numpy as np

PARITY = Path(__file__).resolve().parents[1] / 'pc/parity/karel'
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--out', type=Path, default=Path('/mnt/nvme-500/alpamayo-train/samples/synthetic-parity'))
args = parser.parse_args()
args.out.mkdir(parents=True, exist_ok=True)
rows = json.loads((PARITY / 'chestnut.json').read_text())['rows']
for row in rows:
  stem = row['sample'][:-4]
  with np.load(PARITY / 'samples' / row['sample'], allow_pickle=False) as d:
    arrays = {k: d[k] for k in d.files}
  xyz = np.asarray(row['xyz'], np.float32).reshape(64, 3)
  rot = np.asarray(row['rot'], np.float32).reshape(64, 3, 3)
  arrays['nav_text'] = np.array(str(arrays['nav_text'].reshape(-1)[0]))
  arrays.update(target_xyz=xyz, target_rot=rot, baseline_xyz=xyz, baseline_rot=rot,
                reasoning_token_ids=np.asarray(row['tokens'], np.int32))
  np.savez(args.out / f'{stem}.npz', **arrays)
  route, query = stem.rsplit('-', 1)
  (args.out / f'{stem}.json').write_text(json.dumps({'route': route, 'query_ns': int(query), 'dy_m': 0.0, 'dpsi_rad': 0.0,
                                                      'kind': 'clean', 'source': 'karel-parity-chestnut'}) + '\n')
print(len(rows), 'samples in', args.out)
