"""Decode the Asius six-camera replay (alpamayo2-edge/data/demo2) into model-input files for nv_shift_eval.py.

Usage (system python3 with PyAV): demo_decode.py DEMO_DIR OUT_DIR
One file per moment that reference.json scored: <time_us>.c.npz with image_frames (22,1,3,432,768) in camera then time
order, camera_indices, absolute_timestamps, ego history, gt_xyz (the recorded path), nav (recorded text) and speed.
"""
import json
import sys
from pathlib import Path

import av
import numpy as np

demo, out = Path(sys.argv[1]), Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
route = json.loads((demo/'route.json').read_text())
wanted = {p['time_us'] for p in json.loads((demo/'reference.json').read_text())['predictions']}
moments = [p for p in route['predictions'] if p['time_us'] in wanted]
frames = {}
for cam in route['cameras']:
  ts = cam['timestamps_us']
  position = {t: i for i, t in enumerate(ts)}
  need = {position[f['capture_time_us']] for p in moments for f in p['frames'] if f['camera_index'] == cam['index']}
  with av.open(str(demo/cam['file'])) as c:
    n = -1
    for n, frame in enumerate(c.decode(video=0)):
      if n in need:
        frames[(cam['index'], ts[n])] = frame.to_ndarray(format='rgb24').transpose(2, 0, 1)
  print(cam['index'], cam['file'], 'frames', n+1, 'listed', len(ts), 'kept', len(need), 'invalid kept',
        sum(not cam['valid_frames'][i] for i in need), flush=True)
for p in moments:
  order = sorted(p['frames'], key=lambda f: (f['camera_index'], f['capture_time_us']))
  np.savez(out/f'{p["time_us"]}.c.npz', image_frames=np.stack([frames[(f['camera_index'], f['capture_time_us'])] for f in order])[:, None],
           camera_indices=np.array([f['camera_index'] for f in order], np.int64),
           absolute_timestamps=np.array([f['capture_time_us'] for f in order], np.int64),
           ego_history_xyz=np.array(p['ego_history_xyz'], np.float32), ego_history_rot=np.array(p['ego_history_rot'], np.float32),
           gt_xyz=np.array(p['reference_xyz'], np.float32), nav=p['navigation'], speed=np.float32(p['speed_m_s']),
           shift=np.zeros(2, np.float32))
print('wrote', len(moments))
