"""3 s heading error to the driven path on the 24 turn-accuracy samples, for PC runs and the stored BF16 CPU references.

Usage: turn_accuracy.py RUN.json [RUN.json ...]   (lab run_turn.py outputs: parity/turn-*.json)
"""
import json, sys
from pathlib import Path
import numpy as np


def yaw(rot):
  rot = np.asarray(rot).reshape(64, 3, 3)
  return np.unwrap(np.arctan2(rot[:, 1, 0], rot[:, 0, 0]))


def main(runs):
  first = json.loads(Path(runs[0]).read_text())
  samples = [Path(r['sample']) for r in first]
  truth = [json.loads((s/'truth.json').read_text())['truth'] for s in samples]
  def report(label, pred3, xyz=None, ref=None):
    err = np.degrees(np.abs(np.array(pred3)-np.array([t['3.0']['heading_rad'] for t in truth])))
    line = f'{label:28s} 3 s heading error mean {err.mean():5.2f} deg'
    if xyz is not None and ref is not None:
      d = np.linalg.norm(np.asarray(xyz)-np.asarray(ref), axis=-1)
      line += f' | vs BF16 W4-MLP reference: ADE median {np.median(d.mean(1)):.3f} m, FDE median {np.median(d[:, -1]):.3f} m'
    print(line)
  refs = {}
  for name in ('ref-1x5-6x5-steps1.json', 'ref-1x5-6x5-steps1-w4mlp.json', 'ref-1x5-6x5.json'):
    rows = [json.loads((s/name).read_text()) for s in samples]
    refs[name] = np.array([r['pred_xyz'] for r in rows]) if all('pred_xyz' in r for r in rows) else None
    report('BF16 CPU '+(name.removeprefix('ref-1x5-6x5').removesuffix('.json').lstrip('-') or 'steps10'), [r['pred_heading']['3.0'] for r in rows])
  for run in runs:
    rows = json.loads(Path(run).read_text())
    assert [Path(r['sample']) for r in rows] == samples
    report(Path(run).stem, [yaw(r['rot'])[29] for r in rows], np.array([np.asarray(r['xyz'])[0] for r in rows]), refs['ref-1x5-6x5-steps1-w4mlp.json'])


if __name__ == '__main__':
  main(sys.argv[1:])
