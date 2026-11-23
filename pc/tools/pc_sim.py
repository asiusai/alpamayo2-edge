# PC prelude for device lab scripts (closed-loop sim.py, controllers.py, inputsweep.py, batch*.py). Exec it in a PC lab
# snippet, then call run_device(path). Device paths map to PC copies; outputs go to pc/sim-out/gpu<GPU>/.
#   exec(open('/home/john/alpamayo2-edge/pc/tools/pc_sim.py').read()); run_device(CLOSEDLOOP+'/inputsweep.py')
import re as _re
from pathlib import Path as _Path
# Scripts: the live closed-loop folder when it still exists (others edit it), else the copy. Payloads: the home copy.
CLOSEDLOOP = next(p for p in ('/mnt/nvme-500/tmp/alpamayo-closedloop-20261121', '/home/john/alpamayo2-edge/pc/data/closedloop')
                  if _Path(p, 'sim.py').exists())
TRIMS = '/home/john/alpamayo2-edge/pc/data/closedloop/trims'
PC_SIM_OUT = _Path(globals().get('PC_SIM_OUT', f'/home/john/alpamayo2-edge/pc/sim-out/gpu{GPU}'))
PC_SIM_OUT.mkdir(parents=True, exist_ok=True)
_PC_PATHS = [
  ('/data/alpamayo-w4-live/live.json', '/home/john/asius/tools/alpamayo-car/bench/live.json'),
  ('/data/alpamayo-w4-live/cache', str(rt.config.cache_dir)),
  ('/data/alpamayo-bench/lab/src/', CLOSEDLOOP+'/'),
  ("'/data/alpamayo-bench/closedloop'", repr(TRIMS)),  # 34 s NV12 payloads built from Karel footage
  ('/data/alpamayo-bench/closedloop/', TRIMS+'/'),
  ('/data/alpamayo-bench/', str(PC_SIM_OUT)+'/'),
]
import sys as _sys, types as _types
if 'setproctitle' not in _sys.modules:
  _m = _types.ModuleType('setproctitle'); _m.getproctitle = lambda: 'sim'; _m.setproctitle = lambda *a: None; _sys.modules['setproctitle'] = _m


def _pc_text(text):
  for device, pc in _PC_PATHS:
    text = text.replace(device, pc)
  # PC payload folders have no .complete marker (it is written after the device sync).
  text = text.replace("(route/'.complete').exists()", 'route.is_dir()').replace("(p/'.complete').exists()", 'p.is_dir()')
  return _re.sub(r'exec\(open\(([^()]*)\)\.read\(\)\)', r'exec(_pc_text(open(\1).read()))', text)


def run_device(path):
  exec(compile(_pc_text(open(path).read()), path, 'exec'), globals())
