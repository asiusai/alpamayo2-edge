# Timeline of the end of a drive: Alpamayo plans, engagement, alerts, Chestnut and power, per 5 s.
# Usage (opendbc-coop venv, PYTHONPATH=op-coop): end_scan.py SEGMENT_DIR [...]
import sys, types, collections
from pathlib import Path
import zstandard
sys.modules.setdefault('setproctitle', types.SimpleNamespace(getproctitle=lambda: '', setproctitle=lambda _: None))
from openpilot.cereal import log
seen = collections.Counter()
for seg in sys.argv[1:]:
  raw = zstandard.ZstdDecompressor().stream_reader((Path(seg)/'rlog.zst').open('rb')).read()
  it = log.Event.read_multiple_bytes(raw)
  t0 = None; bucket = None; b = collections.defaultdict(list)
  def flush():
    if bucket is None: return
    g = lambda k, f=lambda v: v[-1]: (f(b[k]) if b[k] else '-')
    print(f'{Path(seg).name[-2:]:>2s} +{bucket*5:3d}s v {g("v")!s:>5} en {g("en")} lat {g("lat")} plans {len(b["plan"])}/{sum(b["plan"])} valid '
          f'chest {g("chest")} pv {g("pv")} alerts {sorted(set(b["alert"]))[:3]} {sorted(set(b["ev"]))[:6]}')
  while True:
    try: m = next(it)
    except Exception: break
    w = m.which(); seen[w] += 1
    if t0 is None: t0 = m.logMonoTime
    k = int((m.logMonoTime-t0)/5e9)
    if k != bucket:
      flush(); bucket = k; b = collections.defaultdict(list)
    if w == 'carState': b['v'].append(round(m.carState.vEgo, 1))
    elif w == 'selfdriveState':
      s = m.selfdriveState; b['en'].append(int(s.enabled))
      if s.alertText1: b['alert'].append((s.alertText1+' | '+s.alertText2)[:70])
    elif w == 'carControl': b['lat'].append(int(m.carControl.latActive))
    elif w == 'alpamayoPlan': b['plan'].append(bool(m.valid))
    elif w == 'chestnutState':
      c = m.chestnutState; b['chest'].append({k2: (round(getattr(c, k2), 1) if isinstance(getattr(c, k2), float) else getattr(c, k2)) for k2 in ('supplyVoltage', 'supplyFault', 'tempC') if hasattr(c, k2)})
    elif w == 'peripheralState': b['pv'].append(m.peripheralState.voltage)
    elif w == 'onroadEvents':
      for e in m.onroadEvents: b['ev'].append(str(e.name))
  flush()
print({k: v for k, v in seen.items() if 'chestnut' in k.lower() or 'alpamayo' in k.lower() or k in ('managerState', 'deviceState', 'errorLogMessage')})
