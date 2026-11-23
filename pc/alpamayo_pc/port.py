"""RX 7900 XTX (gfx1100, RDNA3) adapter for the selected Alpamayo runtime.

The selected runtime targets gfx1200 over USB. This module keeps its code, weights, recipe, and frozen kernel sources.
It changes only what gfx1100 cannot run:
- Frozen HIP kernels recompile for gfx1100 with `gfx11_shim.h` (exact FP8 conversion, gfx12 WMMA through gfx11 WMMA).
- tinygrad FP8 tensor-core GEMMs convert each FP8 operand exactly to BF16 and use BF16 WMMA with FP32 accumulation.
- The frozen gfx1200 vision pair plan is off (it holds gfx1200 binaries). With FAST (default) the general path of the
  same recipe replaces the fused pair, projector and prefill MLP objects and the GEMMs read BF16 weight copies.
- ALPAMAYO_PC_TC_ATTENTION=1 replaces the online-softmax prefill attention object with unfused tensor-core attention.
- KFD replaces USB: ordinary readback, no firmware staging, no SMU access.
Call `install(gpu, cache_dir)` in a fresh process before anything imports tinygrad.
"""
import functools, hashlib, inspect, json, os, textwrap
from pathlib import Path

PC = Path(__file__).resolve().parents[1]
SHIM = Path(__file__).with_name('gfx11_shim.h')
ARCH = 'gfx1100'
_installed = None


def identity():
  """Digest of this adapter's sources; it keys the compiled objects and the persistent graph cache."""
  digest = hashlib.sha256()
  for path in (Path(__file__), SHIM, *sorted(KERNELS.glob('*.cpp'))):
    digest.update(path.read_bytes())
  digest.update(b'fast' if FAST else b'plain')
  digest.update(b'tc-attention' if TC_PREFILL_ATTENTION else b'frozen-attention')
  digest.update(b'flash' if FLASH_ATTENTION else b'no-flash')
  return digest.hexdigest()[:16]


def gfx11_source(kind, source):
  shim = SHIM.read_text()
  return shim+'\n#line 1\n'+source


# FAST (default): every FP8 tensor-core GEMM reads BF16 copies of its FP8 operands. FP8 values convert to BF16 exactly,
# so products, FP8 activation rounding and BF16 output rounding stay the same. Weights convert once at startup
# (about 4 GB more GPU memory). Gfx12-tuned fused vision-pair, projector and prefill-MLP objects give way to the
# general tinygrad path of the same recipe, which is 2-6x faster on gfx1100. Set ALPAMAYO_PC_FAST=0 for the plain port.
FAST = os.environ.get('ALPAMAYO_PC_FAST', '1') != '0'
# Images per vision encoder call. The car encodes pairs (sized for gfx1200); on gfx1100 larger groups make larger GEMMs.
# Rows of different images never mix, so each image's features stay the same. Default 2 keeps the car's grouping.
VISION_GROUP = int(os.environ.get('ALPAMAYO_PC_VISION_GROUP', '2'))
TC_PREFILL_ATTENTION = os.environ.get('ALPAMAYO_PC_TC_ATTENTION', '0') == '1'
# Native gfx1100 flash prefill attention (kernels/prefill_attention.cpp) instead of the shimmed gfx1200 object.
FLASH_ATTENTION = os.environ.get('ALPAMAYO_PC_FLASH', '1') != '0'
KERNELS = Path(__file__).with_name('kernels')
native_gemm = None  # set by install(): native gfx1100 BF16 GEMM on packed operands
_gemm_program = None  # its custom_kernel program builder
NATIVE_GEMM = os.environ.get('ALPAMAYO_PC_NATIVE_GEMM', '1') != '0'
GEMM_TILE = tuple(int(v) for v in os.environ.get('ALPAMAYO_PC_GEMM_TILE', '2,2').split(','))  # 16x16 tiles per wave (rows, cols)
# In captured graphs the native kernel beats tinygrad's TC schedule only for the attention projections (K=2048, N<=2048):
# 142 vs 167 us (q, o) and 97 vs 115 us (k, v) at 1344 rows; FC1/FC2 stay on tinygrad (714/653 us). Same bits either way.


def gemm_fxn(out, x, w, xs, ws, *extra, residual=False):
  """custom_kernel fxn: the native GEMM (bit-identical to amd_gemm._gemm on these shapes) or _gemm itself."""
  from openpilot.selfdrive.alpamayo.selected import amd_gemm
  rows, columns = out.shape
  native = (NATIVE_GEMM and _gemm_program is not None and x.dtype == w.dtype and x.dtype.name in ('bf16', 'bfloat16') and rows % 16 == 0
            and columns % (32*GEMM_TILE[1]) == 0 and columns <= 2048 and w.shape[1] == 2048 and (len(extra) == (1 if residual else 0)))
  if native:
    return _gemm_program(out, x, w, xs, ws, *extra)
  if residual:
    return amd_gemm._gemm(out, x, w, xs, ws, packed_input=True, residual=extra[0])
  return amd_gemm._gemm(out, x, w, xs, ws, *extra, packed_input=True)
BF16 = {}  # id(FP8 weight tensor) -> (that tensor, packed BF16 copy)


def bf16_weight(weight):
  entry = BF16.get(id(weight))
  return entry[1] if entry is not None and entry[0] is weight else None


def install_bf16_gemms(model, amd_gemm, tiled_gemm, vision_tiles):
  from functools import partial
  from tinygrad import Tensor, dtypes
  from openpilot.selfdrive.alpamayo.selected.kernels import frozen_ops

  def activation(x, packer):
    return packer(x).cast(dtypes.bfloat16) if x.dtype == dtypes.fp8e4m3 else packer(x)

  original_tiled = tiled_gemm.gemm
  def tiled(x, weight, scales, tile=(4, 4), row_scale=False, bias=None, column_scale=False, local=(1, 1), swap=False, unroll_k=1,
            group_m=1, residual=None):
    packed = bf16_weight(weight)
    if packed is None:
      return original_tiled(x, weight, scales, tile, row_scale, bias, column_scale, local, swap, unroll_k, group_m, residual)
    out = Tensor.empty(x.shape[0], weight.shape[0], device=x.device, dtype=dtypes.bfloat16)
    extra = (() if bias is None else (bias,))+(() if residual is None else (residual.reshape(out.shape),))
    return Tensor.custom_kernel(out, activation(x, amd_gemm.pack), packed, *scales, *extra,
                                fxn=partial(tiled_gemm.kernel, tile_m=tile[0], tile_n=tile[1], row_scale=row_scale,
                                            column_scale=column_scale, local_m=local[0], local_n=local[1], swap=swap, unroll_k=unroll_k,
                                            group_m=group_m, **({} if residual is None else {'residual': True})))[0]
  tiled_gemm.gemm = tiled

  original_vision = vision_tiles.gemm
  def vision(x, weight, scales, bias, tile, residual=None):
    packed = bf16_weight(weight)
    if packed is None:
      return original_vision(x, weight, scales, bias, tile, residual)
    images, rows, width = x.shape
    scales = (vision_tiles.row_scales(scales[0], images, rows), scales[1])
    return vision_tiles.packed(activation(x, vision_tiles.pack_images), packed, scales, bias, images, rows, tile, residual=residual)
  vision_tiles.gemm = vision

  original_gemm = amd_gemm.gemm
  def gemm(x, weight, scales, bias=None, dtype=dtypes.bfloat16, packed_input=True):
    packed = bf16_weight(weight)
    if packed is None or not packed_input:
      return original_gemm(x, weight, scales, bias, dtype, packed_input)
    out = Tensor.empty(x.shape[0], weight.shape[0], device=x.device, dtype=dtype)
    return Tensor.custom_kernel(out, activation(x, amd_gemm.pack), packed, *scales, *(() if bias is None else (bias,)),
                                fxn=partial(amd_gemm._gemm, packed_input=True))[0]
  amd_gemm.gemm = gemm

  def fused_mlp(x, weight, xs, ws, output_scale, rows):
    # FC1, then the frozen object's epilogue: BF16 ReLU, BF16 square, times 1/scale, clamp, FP8. Output stays packed.
    h = Tensor.custom_kernel(Tensor.empty(rows, weight.shape[0], device=x.device, dtype=dtypes.bfloat16), x, weight, xs, ws, fxn=gemm_fxn)[0]
    return amd_gemm.pack((h.relu().square().float()/output_scale).clamp(-448, 448).cast(dtypes.fp8e4m3))
  frozen_ops.fused_mlp = fused_mlp

  if TC_PREFILL_ATTENTION:
    # Causal prefill attention over KV cache rows 0..offset+rows with tensor-core score and value GEMMs (BF16 scores,
    # FP32 softmax, BF16 probabilities), as amd_attention does for full prompts. The frozen object is an online softmax.
    from openpilot.selfdrive.alpamayo import amd_attention
    source = textwrap.dedent(inspect.getsource(amd_attention._matmul))
    replaced = source.replace('def _matmul(out, a, b, *mask, shape, causal=False,', 'def _matmul_offset(out, a, b, *mask, shape, causal=False, offset=0,')
    replaced = replaced.replace('value = (n <= m).where(', 'value = (n <= m+offset).where(')
    assert replaced.count('offset') >= 2
    scope = dict(amd_attention.__dict__)
    exec(compile(replaced, f'{__file__}:_matmul_offset', 'exec'), scope)
    matmul_offset = scope['_matmul_offset']
    def prefill_attention(q, kv, offset=0):
      batch, heads, rows, width = q.shape
      kv_heads, keys = kv.shape[2], offset+rows
      k, v = kv[0, :, :, :keys], kv[1, :, :, :keys]
      scores = Tensor.empty(batch, heads, rows, keys, device=q.device, dtype=q.dtype)
      scores = Tensor.custom_kernel(scores, amd_attention._pack(q), amd_attention._pack(k),
        fxn=partial(matmul_offset, shape=(batch, heads, kv_heads, rows, keys, width), causal=True, offset=offset, scaled=True))[0]
      probs = Tensor.empty(batch, heads, rows, keys + (-keys % 16), device=q.device, dtype=q.dtype)
      probs = Tensor.custom_kernel(probs, scores, fxn=amd_attention._softmax)[0]
      out = Tensor.empty(q.shape, device=q.device, dtype=q.dtype)
      return Tensor.custom_kernel(out, probs, amd_attention._pack(v.transpose(-2, -1)),
        fxn=partial(amd_attention._matmul, shape=(batch, heads, kv_heads, rows, width, keys)))[0]
    frozen_ops.prefill_attention = prefill_attention

  replacements = {
    'packed_linear': [('packed.value,self.packed_weights[name]', 'packed.value.cast(dtypes.bfloat16),_bf16(self.packed_weights[name])'),
                      ('fxn=partial(amd_gemm._gemm,packed_input=True)', 'fxn=_gemm_fxn')],
    'text_layer': [('packed.value,self.packed_weights[o_proj]', 'packed.value.cast(dtypes.bfloat16),_bf16(self.packed_weights[o_proj])'),
                   ('residual,fxn=amd_gemm.prefill_residual)[0].reshape(x.shape)', 'residual,fxn=partial(_gemm_fxn,residual=True))[0].reshape(x.shape)'),
                   ('residual,fxn=amd_gemm.prefill_residual)[0]\n', 'residual,fxn=partial(_gemm_fxn,residual=True))[0]\n'),
                   ('frozen_ops.fused_mlp(packed.value,self.packed_weights[fc1]',
                    'frozen_ops.fused_mlp(packed.value.cast(dtypes.bfloat16),_bf16(self.packed_weights[fc1])'),
                   ('output,self.packed_weights[fc2]', 'output.cast(dtypes.bfloat16),_bf16(self.packed_weights[fc2])')],
    # The general vision path (tiled FP8 GEMMs, same arithmetic) replaces the fused pair layer and pair projector objects.
    'vision_layer': [("if x.shape == (2,720,1152) and self.images == 2", "if False and x.shape == (2,720,1152) and self.images == 2")],
    'encode_group': [('if count == 2:', 'if False:')] + ([('assert count in (1,2) and not self.images', 'assert count >= 1 and not self.images')]
                                                          if VISION_GROUP > 2 else []),
  }
  for name, pairs in replacements.items():
    source = textwrap.dedent(inspect.getsource(getattr(model.SelectedModel, name)))
    for old, new in pairs:
      assert source.count(old) == 1, (name, old)
      source = source.replace(old, new)
    scope = dict(model.__dict__, _bf16=lambda w: BF16[id(w)][1], _gemm_fxn=gemm_fxn)
    exec(compile(source, f'{__file__}:{name}', 'exec'), scope)
    setattr(model.SelectedModel, name, scope[name])

  original_init = model.SelectedModel.__init__
  def init(self, *args, **kwargs):
    original_init(self, *args, **kwargs)
    pairs = []  # (FP8 tensor, packed BF16 copy); unpacked weights and their packed twins share one copy
    for name, packed in self.packed_weights.items():
      copy = packed.cast(dtypes.bfloat16).contiguous()
      pairs += [(packed, copy), (self.weights[name+'.weight'], copy)]
    for name, value in self.projector_weights.items():
      pairs.append((value, amd_gemm.pack(value).cast(dtypes.bfloat16).contiguous()))
    for name, value in self.weights.items():
      if value.dtype == dtypes.fp8e4m3 and value.ndim == 2 and (name.startswith(('vlm.model.visual.encoder.layers.', 'vlm.model.visual.embeddings.'))
                                                                or name.removesuffix('.weight') in self.expert_weights):
        pairs.append((value, amd_gemm.pack(value).cast(dtypes.bfloat16).contiguous()))
    for i in range(0, len(pairs), 32):
      Tensor.realize(*(c for _, c in pairs[i:i+32]))
    for value, copy in pairs:
      BF16[id(value)] = (value, copy)
    # Named owners let the persistent graph cache rebind the BF16 copies after a restart.
    self.pc_bf16 = {}
    for value, copy in pairs:
      if id(copy) not in {id(c) for c in self.pc_bf16.values()}:
        self.pc_bf16[f'w{len(self.pc_bf16)}'] = copy
    self.pc_bf16_bytes = sum(c.nbytes() for c in self.pc_bf16.values())
  model.SelectedModel.__init__ = init


def install(gpu, cache_dir):
  global _installed
  if _installed is not None:
    if _installed != (gpu, Path(cache_dir)):
      raise RuntimeError('One GPU per process')
    from openpilot.selfdrive.alpamayo.selected import runtime
    return runtime
  import sys
  if 'tinygrad' in sys.modules:
    raise RuntimeError('Install the PC adapter before importing tinygrad')
  cache_dir = Path(cache_dir)
  cache_dir.mkdir(parents=True, exist_ok=True)
  from openpilot.selfdrive.alpamayo.selected import runtime
  values = json.loads((runtime.ASSETS/'environment.json').read_text())
  for key in ('AM_PERF_LEVEL', 'AM_POWER_LIMIT'):
    values.pop(key, None)
  values.update(DEV=f'KFD:{int(gpu)}+AMD:LLVM', PARALLEL='0', ALLOW_DEVICE_USAGE='1',
                HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false', PYTHONDONTWRITEBYTECODE='1',
                XDG_CACHE_HOME=str(cache_dir), CACHEDB=str(cache_dir/'tinygrad.db'))
  if os.environ.get('ALPAMAYO_PC_PROFILE') == '1':
    values['PROFILE'] = '1'  # graph kernel timestamps; separate graph cache entries
  os.environ.update(values)
  runtime.environment = lambda cache, power_limit_w=144: dict(values)  # the PC GPUs keep their own power settings
  runtime.require_firmware = lambda cache: None  # KFD uses the loaded kernel driver firmware.

  # No child processes after the device opens. A forked child inherits the render-node file that holds the KFD VM;
  # when the child closes it (exec, exit, close_fds), amdgpu_flush waits on the VM and KFD evicts and restores every
  # queue of this process through the gfx11 MES firmware. Compile-pool children (PARALLEL) did this for each graph
  # capture. Failed MES queue removals ("MES failed to response msg=3", "Failed to evict queue") then reset the GPU
  # with a shader fault at address 0x2000. So: compile in-process (PARALLEL=0), run the git source checks before the
  # device opens, and log any later fork with its stack.
  from openpilot.selfdrive.alpamayo import inference
  inference.verify_runtime()
  inference.verify_runtime = lambda: None  # verified above, before the device opened
  from openpilot.selfdrive.alpamayo.selected import camera
  camera.library(str(cache_dir))  # the native warp sampler compiles with a host compiler subprocess: do it now
  import traceback
  def forked():
    if _installed is not None:
      print('alpamayo_pc: fork with the GPU open evicts all KFD queues of this process:\n'+''.join(traceback.format_stack(limit=8)),
            file=__import__('sys').stderr, flush=True)
  os.register_at_fork(before=forked)

  # Pinned tinygrad frees a host (USERPTR) buffer by unmapping its CPU pages before it frees the KFD buffer object; the
  # munmap fires the userptr MMU notifier while the object is still registered, which also evicts the queues.
  # Free the KFD object first, then release the CPU pages.
  from tinygrad.runtime import ops_amd as _ops_amd
  def kfd_free(self, storage):
    self._unmap_handle(storage.meta.handle)
    _ops_amd.kfd.AMDKFD_IOC_FREE_MEMORY_OF_GPU(self.kfd, handle=storage.meta.handle)
    if storage.buf: _ops_amd.FileIOInterface.munmap(storage.buf, storage.meta.size)
  if os.environ.get('ALPAMAYO_PC_KFD_FREE_FIX', '1') != '0':
    _ops_amd.KFDIface.free = kfd_free

  from tinygrad import Device, dtypes
  from tinygrad.uop.ops import UOp
  from tinygrad.runtime.support.compiler_amd import compile_hip
  from openpilot.selfdrive.alpamayo.selected import backend, frozen, amd_gemm, model, vision, transport, decode, weights, graph_cache
  from openpilot.selfdrive.alpamayo.selected.kernels import tiled_gemm, vision_tiles

  d = Device['AMD']
  minor = d.iface.props['drm_render_minor']
  if d.arch != ARCH or type(d.iface).__name__ != 'KFDIface':
    raise RuntimeError(f'Expected KFD {ARCH}, found {type(d.iface).__name__} {d.arch}')

  def download(value):
    value = value.contiguous().realize()
    Device[value.device].synchronize()
    return value.numpy()
  transport.download = decode.download = download

  # The pinned scheduler leaves the extra gfx1100 tensor-core upcast range open in custom kernels. Close it at the END.
  from tinygrad.codegen.opt.postrange import Scheduler
  original_ast = Scheduler.get_optimized_ast
  def closed_ast(self, *args, **kwargs):
    ast = original_ast(self, *args, **kwargs)
    if ast.ranges:
      assert len(ast.src) == 1 and ast.src[0].op.name == 'END'
      ast = ast.replace(src=(ast.src[0].replace(src=ast.src[0].src+tuple(ast.ranges)),))
    return ast
  Scheduler.get_optimized_ast = closed_ast

  # Pinned generic float decomposition flushes FP8 subnormals on targets without FP8 hardware. Keep all E4M3FN values.
  from tinygrad.codegen.decomp import dtype as decomp
  original_f2f = decomp.f2f
  def exact_f2f(v, fr, to, sat=True):
    normal = original_f2f(v, fr, to, sat)
    if fr == dtypes.fp8e4m3:
      bits = v.cast(dtypes.uint32)
      small = (bits & 7).cast(dtypes.float32)*(1/512)
      small = (bits & 128).ne(0).where(-small, small).cast(to)
      return (bits & 0x78).eq(0).where(small, normal)
    if to == dtypes.fp8e4m3:
      value = v.bitcast(fr).cast(dtypes.float32)
      magnitude = value.abs()*512
      whole = magnitude.cast(dtypes.uint32)
      fraction = magnitude-whole.cast(dtypes.float32)
      rounded = whole+((fraction > .5) | ((fraction.eq(.5)) & ((whole & 1).ne(0)))).cast(dtypes.uint32)
      sign = (v.cast(dtypes.uint32) >> (fr.bitsize-8)) & 128
      return (value.abs() < 1/64).where((sign | rounded).cast(dtypes.uint8), normal)
    return normal
  decomp.f2f = exact_f2f

  class Backend:
    active = False
    def __init__(self, timeout_ms):
      from tinygrad.device import Compiled
      from tinygrad.runtime import ops_amd
      if Backend.active:
        raise RuntimeError('Use one selected runtime per dedicated process')
      self.device = d
      self.previous_compiler = d.renderer.compiler
      self.previous_packet = ops_amd.dispatch_packet
      self.previous_timeout = Compiled.wait_timeout_ms
      d.renderer.compiler = backend.WGPCompiler(d.arch)
      original = self.previous_packet
      ops_amd.dispatch_packet = lambda *a, **kw: [UOp.const(w, dtypes.uint32) if isinstance(w, int) else w for w in original(*a, **kw)]
      Compiled.wait_timeout_ms = timeout_ms
      Backend.active = True
    def close(self):
      from tinygrad.device import Compiled
      from tinygrad.runtime import ops_amd
      try:
        d.synchronize()
      finally:
        d.renderer.compiler = self.previous_compiler
        ops_amd.dispatch_packet = self.previous_packet
        Compiled.wait_timeout_ms = self.previous_timeout
        Backend.active = False
  backend.Backend = Backend

  objects = PC/'objects'
  objects.mkdir(exist_ok=True)
  def compiled(source):
    key = hashlib.sha256((ARCH+'\0'+source).encode()).hexdigest()
    path = objects/(key+'.hsaco')
    if not path.exists():
      binary = compile_hip(source, ARCH)
      temporary = path.with_suffix(f'.{os.getpid()}.tmp')
      temporary.write_bytes(binary)
      os.replace(temporary, path)
    return path.read_bytes()

  original_pair = frozen.object_pair
  @functools.lru_cache(None)
  def pair(kind, args):
    source, _ = original_pair(kind, args)  # checks the pinned gfx1200 source and object hashes first
    source = gfx11_source(kind, source)
    return source, compiled(source)
  frozen.object_pair = pair
  original_verify = frozen.verify_assets
  def verify():
    frozen.object_pair = original_pair
    try:
      original_verify()
    finally:
      frozen.object_pair = pair
  frozen.verify_assets = runtime.verify_assets = backend.verify_assets = verify

  original_hash = weights._hash_kernel
  @functools.lru_cache(maxsize=1)
  def hash_kernel():
    source, _ = original_hash()
    source = gfx11_source('weight_sha256', source)
    return source, compiled(source)
  weights._hash_kernel = hash_kernel
  # The GPU leaf-proof check stays; only its gfx1200 device guard changes.
  scope = dict(weights.__dict__)
  source = textwrap.dedent(inspect.getsource(weights._verify_storage)).replace("Device[device].arch != 'gfx1200'", f"Device[device].arch != '{ARCH}'")
  exec(compile(source, f'{__file__}:_verify_storage', 'exec'), scope)
  weights._verify_storage = scope['_verify_storage']

  # FP8 tensor-core GEMMs: same address layout and FP32 accumulation; each FP8 operand converts exactly to BF16.
  for module, name in ((amd_gemm, '_gemm'), (tiled_gemm, 'kernel'), (vision_tiles, 'kernel')):
    source = inspect.getsource(getattr(module, name))
    replaced = source.replace('(a*b).cast(dtypes.float32)', '(a.cast(dtypes.bfloat16)*b.cast(dtypes.bfloat16)).cast(dtypes.float32)')
    assert replaced != source
    exec(compile(replaced, f'{__file__}:{module.__name__}.{name}', 'exec'), module.__dict__)
  source = textwrap.dedent(inspect.getsource(model.SelectedModel.base_linear))
  replaced = source.replace('quantized.dot(weight.T,dtype=dtypes.float32)',
                            'quantized.cast(dtypes.bfloat16).dot(weight.cast(dtypes.bfloat16).T,dtype=dtypes.float32)')
  replaced = replaced.replace('super().linear(x,name)', 'EdgeModel.linear(self,x,name)')
  assert replaced != source
  scope = dict(model.__dict__)
  exec(compile(replaced, f'{__file__}:base_linear', 'exec'), scope)
  model.SelectedModel.base_linear = scope['base_linear']

  # The frozen pair plan holds gfx1200 binaries. Pair layers keep the frozen operations through the TinyJit encoder.
  vision.VisionCache.group_limit = VISION_GROUP
  original_vision_init = vision.VisionCache.__init__
  def vision_init(self, m):
    previous, m.amd = m.amd, False
    try:
      original_vision_init(self, m)
    finally:
      m.amd = previous
  vision.VisionCache.__init__ = vision_init

  if FAST:
    install_bf16_gemms(model, amd_gemm, tiled_gemm, vision_tiles)

  global native_gemm, _gemm_program
  gemm_template = (KERNELS/'gemm_bf16.cpp').read_text()
  def gemm_program(out, x, w, xs, ws, *residual):
    from tinygrad.uop.ops import KernelInfo, Ops
    (rows, columns), width = out.shape, w.shape[1]
    tm, tn = GEMM_TILE
    source = (f'#define TM {tm}\n#define TN {tn}\n'+gemm_template.replace('__M__', str(rows)).replace('__N__', str(columns)).replace('__K__', str(width))
              .replace('__RESIDUAL__', '1' if residual else '0'))
    dependency = x.flatten()[0].load().cast(dtypes.float32)+w.flatten()[0].load().cast(dtypes.float32)+xs.flatten()[0].load()+ws.flatten()[0].load()
    if residual:
      dependency = dependency+residual[0].flatten()[0].load().cast(dtypes.float32)
    sink = UOp.sink(out.flatten()[0].store(dependency.cast(out.dtype)), UOp.special((-(-rows//(32*tm)))*(columns//(32*tn)), 'gidx0'),
                    UOp.special(128, 'lidx0'), arg=KernelInfo(name='pc_gemm_bf16'))
    return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.LINEAR, src=tuple(sink.toposort())), UOp(Ops.SOURCE, arg=source),
                                 UOp(Ops.BINARY, arg=compiled(source))))
  def native_gemm(x, w, xs, ws, residual=None):
    """Packed BF16 x (rows, K) times packed BF16 w (N, K)^T with the _gemm epilogue; rows % 16 == 0, N % 128 == 0, K % 32 == 0."""
    from tinygrad import Tensor
    rows, columns = x.shape[0], w.shape[0]
    assert x.dtype == w.dtype == dtypes.bfloat16 and rows % 16 == 0 and columns % (32*GEMM_TILE[1]) == 0 and x.shape[1] % 32 == 0
    out = Tensor.empty(rows, columns, device=x.device, dtype=dtypes.bfloat16)
    return Tensor.custom_kernel(out, x, w, xs, ws, *(() if residual is None else (residual,)), fxn=gemm_program)[0]
  _gemm_program = gemm_program

  from openpilot.selfdrive.alpamayo.selected.kernels import frozen_ops
  frozen_ops.frozen_prefill_attention = frozen_ops.prefill_attention
  if FLASH_ATTENTION and not TC_PREFILL_ATTENTION:
    from tinygrad import Tensor
    from tinygrad.uop.ops import KernelInfo, Ops
    template = (KERNELS/'prefill_attention.cpp').read_text()
    def flash_program(out, q, kv, *, rows, offset):
      source = template.replace('__ROWS__', str(rows)).replace('__OFFSET__', str(offset))
      dependency = q.flatten()[0].load()+kv.flatten()[0].load()
      sink = UOp.sink(out.flatten()[0].store(dependency), UOp.special(16*((rows+63)//64), 'gidx0'), UOp.special(128, 'lidx0'),
                      arg=KernelInfo(name='pc_prefill_attention'))
      return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.LINEAR, src=tuple(sink.toposort())), UOp(Ops.SOURCE, arg=source),
                                   UOp(Ops.BINARY, arg=compiled(source))))
    def flash_attention(q, kv, offset=0):
      assert q.shape[:2] == (1, 16) and q.shape[3] == 128 and kv.shape == (2, 1, 8, 2304, 128) and q.dtype == kv.dtype == dtypes.bfloat16
      out = Tensor.empty(q.shape, device=q.device, dtype=q.dtype)
      return Tensor.custom_kernel(out, q.contiguous(), kv, fxn=functools.partial(flash_program, rows=q.shape[2], offset=offset))[0]
    frozen_ops.prefill_attention = flash_attention

  original_configure = graph_cache.configure
  def configure(directory, roots, key_parts):
    return original_configure(directory, roots, (*key_parts, ('pc-port', ARCH, identity())))
  graph_cache.configure = configure

  original_runtime_init = runtime.Runtime.__init__
  def runtime_init(self, config):
    original_runtime_init(self, config)
    self.manifest.update(arch=ARCH, gpu=int(gpu), render_minor=minor, pc_port=identity(),
                         pc_backend='KFD gfx1100; frozen kernels recompiled with gfx11_shim.h; FP8 GEMM operands as BF16 WMMA',
                         frozen_vision_pair_plan=False, pc_bf16_gemm_operands=FAST,
                         pc_bf16_weight_bytes=getattr(self.model, 'pc_bf16_bytes', 0))
  runtime.Runtime.__init__ = runtime_init
  _installed = (gpu, cache_dir)
  return runtime
