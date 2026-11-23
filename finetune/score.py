"""One-line score of a candidate: turn heldout metrics, straight lean, and recovery heldout metrics.

Usage: score.py NAME TURN_PREDICTIONS.npz [REC_PREDICTIONS.npz] [--turns DIR] [--rec DIR] [--json OUT]
Turn path error: median (as evaluate_turns.py), mean and 90th percentile over turning moments.
Straight lean: mean signed (predicted - human) lateral at 6.4 s on straight moments (|human heading change| < 0.03 rad);
right is positive (the model frame has left positive). Also the share of straight plans that end left of the human path.
"""
import argparse
import contextlib
import io
import json
from pathlib import Path

import numpy as np

import evaluate_recovery
import evaluate_turns

ap = argparse.ArgumentParser()
ap.add_argument('name'); ap.add_argument('turn_preds'); ap.add_argument('rec_preds', nargs='?')
ap.add_argument('--turns', default='/mnt/nvme-500/alpamayo-train/turns/samples-120/heldout')
ap.add_argument('--rec', default='/mnt/nvme-500/alpamayo-train/teacher/samples/heldout')
ap.add_argument('--json', type=Path)
a = ap.parse_args()
with contextlib.redirect_stdout(io.StringIO()):
  t = evaluate_turns.main(a.turns, a.turn_preds)
rows = evaluate_turns.load(a.turns, a.turn_preds)
lean, turn_err = [], []
for _, _, tgt, p in rows:
  h6, p6 = evaluate_turns.heading(tgt, 63), evaluate_turns.heading(p, 63)
  if h6 is not None and abs(h6) < .03:
    lean.append(-(p[63, 1] - tgt[63, 1]))
  if h6 is not None and p6 is not None and abs(h6) > .1 and evaluate_turns.heading(tgt, 29) is not None and evaluate_turns.heading(p, 29) is not None:
    turn_err.append(evaluate_turns.path_error(p, tgt))
lean = np.array(lean)
out = {'name': a.name, 'turn_ratio_3s': t['turn_ratio_3s'], 'turn_ratio_6s': t['turn_ratio_6s'], 'turn_path_m': t['turn_path_err_m'],
       'turn_path_mean_m': round(float(np.mean(turn_err)), 3), 'turn_path_p90_m': round(float(np.percentile(turn_err, 90)), 2),
       'straight_path_m': t['straight_path_err_m'], 'lat3_m': t['lat_err_3s_m'],
       'straight_lean_6s_m': round(float(lean.mean()), 3), 'straight_left_share': round(float((lean < 0).mean()), 2), 'n_straight': len(lean)}
if a.rec_preds:
  tr, pr = evaluate_recovery.load(a.rec, a.rec_preds)
  sel = np.abs(tr[:, 1]) >= evaluate_recovery.MIN_CORRECTION
  x, y = tr[sel, 1], pr[sel, 1]
  out.update(rec_ratio_3s=round(float(np.sum(x*y)/np.sum(x*x)), 3), rec_corr_3s=round(float(np.corrcoef(x, y)[0, 1]), 3),
             rec_median_m=round(float(np.median(np.abs(pr[:, 1]-tr[:, 1]))), 3), rec_n=len(tr))
print(json.dumps(out))
if a.json:
  a.json.write_text(json.dumps(out) + '\n')
