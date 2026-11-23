"""Stage 1 of Karel plan parity: copy the recorded NV12 frames that selected Karel plans used.

Run with the system python3 (PyAV, pycapnp, zstandard). Reads plan JSONs and Karel footage; writes one NV12 .npy per
(route, view, frame_id) into OUT/frames and the plan list into OUT/plans.json. Footage is read-only.
Usage: extract_karel_frames.py OUT PER_ROUTE SESSION_DIR [SESSION_DIR ...]
"""
import json, sys
from pathlib import Path
import av, capnp, numpy as np, zstandard

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'finetune'))
from paths import FOOTAGE  # car footage root: ALPAMAYO_FOOTAGE or finetune/paths.local.json
CEREAL = Path(__file__).resolve().parents[1]/'openpilot/cereal'
VIEWS = {6: ('narrowRoadEncodeIdx', 'fcamera.mp4'), 1: ('wideRoadEncodeIdx', 'ecamera.mp4')}


def pick(session, count):
  plans = sorted(p for p in session.glob('*.json') if p.stem.isdigit())
  rows = []
  for p in plans:
    d = json.loads(p.read_text())
    if 'prediction' not in d or len(d['input']['frames']) != 10:
      continue
    rows.append((p, d))
  # Spread evenly over the drive; prefer moving samples and keep every distinct reasoning prefix seen first.
  moving = [r for r in rows if r[1]['input']['vehicle_history'].get('speed_m_s', 0) > 3] or rows
  step = max(1, len(moving)//count)
  return [r[0] for r in moving[::step][:count]]


def main(out, per_route, sessions):
  log = capnp.load(str(CEREAL/'log.capnp'), imports=[str(CEREAL), str(CEREAL.parent/'cereal_imports')])
  (out/'frames').mkdir(parents=True, exist_ok=True)
  plans = [p for s in sessions for p in pick(Path(s), per_route)]
  need = {}  # route -> view -> set(frame_id)
  for p in plans:
    d = json.loads(p.read_text())
    for f in d['input']['frames']:
      need.setdefault(d['input']['route'], {}).setdefault(f['view'], set()).add(f['frame_id'])
  for route, views in need.items():
    segments = sorted(FOOTAGE.glob(route+'--*'), key=lambda s: int(s.name.rsplit('--', 1)[1]))
    for seg in segments:
      raw = zstandard.ZstdDecompressor().stream_reader((seg/'rlog.zst').open('rb')).read()
      index = {v: {} for v in VIEWS}
      events = log.Event.read_multiple_bytes(raw)
      while True:
        try:
          m = next(events)
        except StopIteration:
          break
        except Exception:  # noqa: BLE001 - the last segment of a drive can end mid-message
          break
        kind = m.which()
        for v, (name, _) in VIEWS.items():
          if kind == name:
            e = getattr(m, kind)
            index[v][int(e.segmentId)] = int(e.frameId)
      for v, (_, video) in VIEWS.items():
        wanted = {pos: fid for pos, fid in index[v].items() if fid in views.get(v, ()) and not (out/'frames'/f'{route}-{v}-{fid}.npy').exists()}
        if not wanted:
          continue
        last = max(wanted)
        with av.open(str(seg/video)) as container:
          for pos, frame in enumerate(container.decode(video=0)):
            if pos in wanted:
              assert (frame.width, frame.height) == (1344, 760)
              np.save(out/'frames'/f'{route}-{v}-{wanted[pos]}.npy', frame.reformat(format='nv12').to_ndarray().reshape(1140, 1344))
            if pos >= last:
              break
      print(seg.name, 'done', flush=True)
  missing = [(r, v, f) for r, vs in need.items() for v, fs in vs.items() for f in fs if not (out/'frames'/f'{r}-{v}-{f}.npy').exists()]
  (out/'plans.json').write_text(json.dumps({'plans': [str(p) for p in plans], 'missing_frames': missing}, indent=1))
  print('plans', len(plans), 'missing frames', len(missing))


if __name__ == '__main__':
  main(Path(sys.argv[1]), int(sys.argv[2]), sys.argv[3:])
