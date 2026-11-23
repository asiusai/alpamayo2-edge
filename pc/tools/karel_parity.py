"""Stage 2 of Karel plan parity (lab snippet): rebuild each plan's inputs and compare PC predictions with Chestnut's.

Images: recorded footage frames (HEVC-decoded, so not byte-identical to the live NV12), sampled with the plan's own
per-frame calibration and the car geometry. Ego history, navigation text, timestamps and camera order: exact copies
from the plan JSON, checked against the plan's input hashes. Set PARITY_DIR and LABEL before exec.
"""
import hashlib, io, json
from pathlib import Path
from openpilot.selfdrive.alpamayo.selected.camera import calibrated_image

PARITY_DIR = Path(globals().get('PARITY_DIR', '/home/john/alpamayo2-edge/pc/parity/karel'))
LABEL = globals().get('LABEL', 'pc')
GEOMETRY = json.loads(Path('/home/john/alpamayo2-edge/pc/assets/geometry.json').read_text())
CACHE = rt.config.cache_dir


def npy_sha(array):
  header = io.BytesIO()
  np.lib.format.write_array_header_1_0(header, np.lib.format.header_data_from_array_1_0(array))
  digest = hashlib.sha256(header.getbuffer()); digest.update(np.ascontiguousarray(array).reshape(-1).view(np.uint8))
  return digest.hexdigest()


def build(plan):
  i = plan['input']
  images = []
  for f in i['frames']:
    nv12 = np.load(PARITY_DIR/'frames'/f"{i['route']}-{f['view']}-{f['frame_id']}.npy")
    images.append(calibrated_image(nv12, f['view'], GEOMETRY, f['image_calibration'], cache_dir=CACHE)[None])
  sample = {'image_frames': np.ascontiguousarray(np.stack(images)),
            'camera_indices': np.array([f['view'] for f in i['frames']], dtype=np.int64),
            'absolute_timestamps': np.array([f['capture_ns']//1000 for f in i['frames']], dtype=np.int64),
            'ego_history_xyz': np.asarray(i['ego_history_xyz'], dtype=np.float32),
            'ego_history_rot': np.asarray(i['ego_history_rot'], dtype=np.float32),
            'nav_text': [i['navigation_text']]}
  checks = {k: npy_sha(np.array(v[0]) if k == 'nav_text' else v) == i['input_sha256'][k] for k, v in sample.items()}
  return sample, checks


def yaw(rot):
  rot = np.asarray(rot)[0]
  return np.unwrap(np.arctan2(rot[:, 1, 0], rot[:, 0, 0]))


def compare(a, b):
  pa, pb = np.asarray(a['xyz'])[0], np.asarray(b['xyz'])[0]
  d = np.linalg.norm(pa-pb, axis=-1)
  return dict(ade_m=float(d.mean()), fde_m=float(d[-1]), max_m=float(d.max()), d3s_m=float(d[29]),
              heading_max_deg=float(np.degrees(np.abs(yaw(a['rot'])-yaw(b['rot'])).max())),
              heading3s_deg=float(np.degrees(abs(yaw(a['rot'])[29]-yaw(b['rot'])[29]))),
              tokens_equal=a['generated_token_ids'] == b['generated_token_ids'], text_equal=a['reasoning'] == b['reasoning'])


plans = [Path(p) for p in json.loads((PARITY_DIR/'plans.json').read_text())['plans']]
rows = []
rt.stage_sync = True
for path in plans:
  plan = json.loads(path.read_text())
  sample, checks = build(plan)
  t = time.perf_counter(); pred = rt.predict(sample); rt.synchronize(); ms = (time.perf_counter()-t)*1e3
  row = dict(plan=str(path), route=plan['input']['route'], hashes=checks, ms=ms, chestnut_ms=plan['prediction']['runtime_ms'],
             speed=plan['input']['vehicle_history'].get('speed_m_s'), chestnut_text=plan['prediction']['reasoning'], pc_text=pred['reasoning'],
             pc=dict(xyz=pred['xyz'], rot=pred['rot'], tokens=pred['generated_token_ids']), **compare(pred, plan['prediction']))
  rows.append(row)
  print(f"{path.parent.name[-6:]}/{path.stem} v={row['speed'] or 0:5.1f} ade={row['ade_m']:.3f} fde={row['fde_m']:.3f} hdg={row['heading_max_deg']:.2f} "
        f"tok={row['tokens_equal']} img_hash={checks['image_frames']} other_hashes={all(v for k, v in checks.items() if k != 'image_frames')}", flush=True)
(PARITY_DIR/f'results-{LABEL}.json').write_text(json.dumps(rows))
ade = np.array([r['ade_m'] for r in rows]); fde = np.array([r['fde_m'] for r in rows]); hd = np.array([r['heading_max_deg'] for r in rows])
print(f"n={len(rows)} ADE median {np.median(ade):.3f} p90 {np.percentile(ade, 90):.3f} max {ade.max():.3f}; FDE median {np.median(fde):.3f} "
      f"p90 {np.percentile(fde, 90):.3f} max {fde.max():.3f}; heading max median {np.median(hd):.2f} p90 {np.percentile(hd, 90):.2f} deg; "
      f"tokens equal {sum(r['tokens_equal'] for r in rows)}/{len(rows)}; non-image hashes ok {sum(all(v for k, v in r['hashes'].items() if k != 'image_frames') for r in rows)}")
