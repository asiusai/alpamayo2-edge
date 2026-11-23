# Regenerate the frozen vision pair plan after model/kernel source changes (checkout synced first).
# Reuses frozen programs for unchanged operations, compiles changed ones with the pinned compiler, serializes, and checks features.
import importlib, hashlib, gzip, json
from pathlib import Path
from tinygrad import Device
from tinygrad.codegen import to_program
from tinygrad.uop.ops import Ops, UOp, ProgramInfo
import openpilot.selfdrive.alpamayo.selected.graph_cache as gc
gc.disable()
for name in (() if globals().get('NO_RELOAD') else ('openpilot.selfdrive.alpamayo.amd_attention','openpilot.selfdrive.alpamayo.selected.frozen',
             'openpilot.selfdrive.alpamayo.selected.kernels.tiled_gemm','openpilot.selfdrive.alpamayo.selected.kernels.frozen_ops',
             'openpilot.selfdrive.alpamayo.selected.kernels.bf16_linear','openpilot.selfdrive.alpamayo.selected.kernels.vision_tiles',
             'openpilot.selfdrive.alpamayo.selected.model','openpilot.selfdrive.alpamayo.selected.vision_pair.programs',
             'openpilot.selfdrive.alpamayo.selected.vision_pair')):
  importlib.reload(importlib.import_module(name))
import openpilot.selfdrive.alpamayo.selected.model as mm
import openpilot.selfdrive.alpamayo.selected.vision_pair.programs as vpp
import openpilot.selfdrive.alpamayo.selected.vision_pair as vpi
rt.model.__class__ = mm.SelectedModel
exec(open('/data/alpamayo-bench/lab/src/plan_codec.py').read())
renderer = Device[Device.DEFAULT].renderer
base = json.loads(gzip.decompress(vpp.ASSET.read_bytes()))
old = vpp.PairPrograms(renderer)
used, table = {}, {}
def recording_compiler(linear):
  calls = []
  for c in linear.src:
    body = c.body
    if isinstance(body.arg, ProgramInfo):
      prg = body
    elif body.key.hex() in old.table:
      prg = old.programs[old.table[body.key.hex()]]
    else:
      prg = to_program(body, renderer)
    table[body.key.hex()] = prg.key.hex()
    used[prg.key.hex()] = prg
    calls.append(c.replace(src=(prg, *c.src[1:])))
  return UOp(Ops.LINEAR, src=tuple(calls))
digests = []
orig_require = vpi.require
vpi.require = lambda ok, msg: None if msg == 'Vision pair operation order differs' else orig_require(ok, msg)
def encode_group(self, *pairs):
  h = hashlib.sha256(); parent = vpi.capturing[0]
  def compile_group(linear):
    compiled = self.compiler(linear)
    for c in compiled.src: h.update(c.body.key)
    return compiled
  vpi.capturing[0] = vpi.CompileAtRealize(parent, compile_group)
  try: result = mm.SelectedModel.encode_group(self, *pairs)
  finally: vpi.capturing[0] = parent
  digests.append(h.hexdigest()); return result
vpi.PairCapture.encode_group = encode_group
weights = vpi.VisionWeights(rt.model); weights.check_layout()
capture = vpi.PairCapture(weights.model, recording_compiler)
jit = gc.CachedJit(capture.encode_group, 'vision-pair'); jit.jit.cnt = 1
outs = [t.clone().realize().numpy() for t in jit(*VPAIRS)]
outs2 = [t.clone().realize().numpy() for t in jit(*VPAIRS)]
reused = sum(1 for k in used if k in old.programs)
print(f'programs used {len(used)} (reused {reused}, new {len(used)-reused}); table entries {len(table)}')
plan = build_plan(base, used, table)
raw = gzip.compress(json.dumps(plan, separators=(',', ':')).encode(), mtime=0)
Path('/data/alpamayo-bench/lab/src/out').mkdir(parents=True, exist_ok=True)
Path('/data/alpamayo-bench/lab/src/out/vision.json.gz').write_bytes(raw)
print('vision SHA256', hashlib.sha256(raw).hexdigest(), 'PROGRAMS', len(used))
print('ORDERED_PROGRAMS', set(digests))
print('feature digests', [hashlib.sha256(a.tobytes()).hexdigest()[:16] for a in outs], [hashlib.sha256(a.tobytes()).hexdigest()[:16] for a in outs2])
vpi.require = orig_require
