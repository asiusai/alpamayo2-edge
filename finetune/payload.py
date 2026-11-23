"""Sparse sweep payload from one recorded segment: all logged states, but only the camera frames that the selected
moments sample (t-2, t-1, t-0.5, t-0.1, t). Moments are human-steered (lateral control off from t-2.1 to t+3 s), at
>= 4 m/s, on a 1 s grid: every turning moment (heading change > 0.05 rad over 3 s) and every third straight one.
Usage: sparse_payload.py OUT SEGMENT_DIR [MAX_MOMENTS]"""
import json, math, subprocess, sys
from pathlib import Path
import av, numpy as np, zstandard
from openpilot.cereal import log
from openpilot.selfdrive.alpamayo.capture import state_value
RECORDINGS_UUID = 'aac28ecf-db48-4b4f-8a99-bde6a3726ab9'
FRAME_BYTES = 1344*1140
VIEWS = {6: ('narrowRoadEncodeIdx', 'narrowRoadCameraState', 'fcamera.mp4'), 1: ('wideRoadEncodeIdx', 'wideRoadCameraState', 'ecamera.mp4')}
STATES = ('carState', 'carParams', 'vehicleParameters', 'extrinsicsCalibration', 'deviceMotion', 'navInstruction')
OFFSETS = (-2., -1., -.5, -.1, 0.)
STRAIGHT_EVERY = 1  # training keeps every straight moment
FUTURE_NS = 6_500_000_000  # samples need 6.4 s of logged future
TEACHER_TURN = .35  # rad: teacher (comma plan) samples skip moments whose driven path or plan turns more by 6.4 s

def main(out, seg, max_moments=20, teacher=False):
  if subprocess.check_output(['findmnt', '-no', 'UUID', '-T', str(seg)], text=True).strip() != RECORDINGS_UUID:
    raise RuntimeError(f'{seg} is not on the recordings drive')
  raw = zstandard.ZstdDecompressor().stream_reader((seg/'rlog.zst').open('rb')).read()
  events, index, camera_rx, lat, yaw, speed = [], {v: {} for v in VIEWS}, {v: {} for v in VIEWS}, [], [], []
  plans = {}
  for m in log.Event.read_multiple_bytes(raw):
    kind = m.which()
    for v, (idx_name, state_name, _) in VIEWS.items():
      if kind == idx_name:
        e = getattr(m, kind); index[v][int(e.segmentId)] = (int(e.frameId), int(e.timestampSof), int(e.timestampEof))
      elif kind == state_name:
        camera_rx[v][int(getattr(m, kind).frameId)] = int(m.logMonoTime)
    if kind == 'carControl': lat.append((m.logMonoTime, m.carControl.latActive))
    if teacher and kind == 'modelV2' and len(m.modelV2.position.t) == 33:
      p = m.modelV2.position  # comma's plan for this road frame; y is right positive there, left positive here
      plans[int(m.modelV2.frameId)] = (list(p.t), list(p.x), [-y for y in p.y])
    if kind == 'deviceMotion': yaw.append((m.logMonoTime, -m.deviceMotion.orientationNED.z))
    if kind == 'carState': speed.append((m.logMonoTime, m.carState.vEgo))
    if kind in STATES:
      msg = getattr(m, kind); value, valid = state_value(kind, msg)
      stamp = int(msg.timestamp) if kind == 'deviceMotion' and msg.timestamp > 0 else int(m.logMonoTime)
      if kind == 'navInstruction':
        valid = valid and bool(value['maneuverPrimaryText']); value['routeRevision'] = 'replay' if valid else None
      events.append(dict(k=kind, stamp=stamp, received=int(m.logMonoTime), valid=bool(m.valid and valid), value=value))
  yt, yv = np.array([y[0] for y in yaw]), np.unwrap(np.array([y[1] for y in yaw]))
  st, sv = np.array([s[0] for s in speed]), np.array([s[1] for s in speed])
  lt, lv = np.array([l[0] for l in lat]), np.array([l[1] for l in lat], float)
  sofs = {v: sorted(s for (_, s, _) in index[v].values()) for v in VIEWS}
  first = max(min(sofs[v]) for v in VIEWS)
  grid = np.arange(math.ceil((first+2.6e9)/1e9)*1e9, min(yt[-1], max(sofs[6]), st[-1])-FUTURE_NS, 1e9).astype(np.int64)
  moments, nstraight = [], 0
  for t in grid:
    if np.interp(t, st, sv) < 4: continue
    if teacher:
      # Recovery targets from comma's plan: any driver (including Alpamayo drifting), but no intersection turns.
      if abs(np.interp(t+6.4e9, yt, yv)-np.interp(t, yt, yv)) > TEACHER_TURN: continue
    else:
      m = (lt >= t-2.1e9) & (lt <= t+6.5e9)  # human steering over the history and the whole target horizon
      if m.any() and lv[m].max() > 0: continue
    turn = abs(np.interp(t+3e9, yt, yv)-np.interp(t, yt, yv)) > .05
    if not turn:
      nstraight += 1
      if nstraight % STRAIGHT_EVERY: continue
    moments.append(int(t))
  moments = moments[:max_moments]
  # Frames as the car selects them: the query is the newest narrow exposure at or before the moment; each history frame is
  # the newest exposure at or before min(query+offset+FRAME_SLACK, query) (frontend.Store.select).
  need = {v: set() for v in VIEWS}
  by = {v: sorted((sof, pos) for pos, (_, sof, _) in index[v].items()) for v in VIEWS}
  sofs = {v: np.array([b[0] for b in by[v]]) for v in VIEWS}
  queries, targets = [], {}
  frame_of = {sof: fid for (fid, sof, _) in index[6].values()}
  for t in moments:
    qi = int(np.searchsorted(sofs[6], t, side='right'))-1
    if qi < 0: continue
    q = int(sofs[6][qi])
    if teacher:
      plan = plans.get(frame_of.get(q))
      if plan is None: continue
      T, X, Y = (np.array(v) for v in plan)
      i3 = int(np.searchsorted(T, 6.4))
      if abs(math.atan2(Y[i3]-Y[i3-1], X[i3]-X[i3-1])) > TEACHER_TURN: continue
      targets[str(q)] = plan
    queries.append(q)
    for v in VIEWS:
      for o in OFFSETS:
        i = int(np.searchsorted(sofs[v], min(q+int(o*1e9)+10_000_000, q), side='right'))-1
        if i >= 0: need[v].add(by[v][i][1])
  moments = queries
  out.mkdir(parents=True, exist_ok=False)
  counts = {v: 0 for v in VIEWS}
  for v, (_, _, video) in VIEWS.items():
    with (out/f'drive-{v}.nv12').open('wb') as bank, av.open(str(seg/video)) as container:
      for position, frame in enumerate(container.decode(video=0)):
        if position not in need[v] or position not in index[v]: continue
        frame_id, sof, eof = index[v][position]
        bank.write(frame.reformat(format='nv12').to_ndarray().reshape(-1).tobytes())
        events.append(dict(k='frame', v=v, id=frame_id, index=position, sof=sof, eof=eof,
                           received=camera_rx[v].get(frame_id, eof+1_500_000), offset=counts[v]*FRAME_BYTES))
        counts[v] += 1
  events.sort(key=lambda e: e['received'])
  (out/'drive-events.json').write_text(json.dumps(events, allow_nan=False))
  (out/'fixture.json').write_text(json.dumps(dict(source=[str(seg)], moments=moments, **({'teacher': targets} if teacher else {}),
      media={'drive': {str(v): {'file': f'drive-{v}.nv12', 'frames': counts[v]} for v in VIEWS}}), indent=1)+'\n')
  print(seg.name, 'moments', len(moments), 'frames', counts, flush=True)

if __name__ == '__main__':
  main(Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3]) if len(sys.argv) > 3 else 20, len(sys.argv) > 4 and sys.argv[4] == 'teacher')
