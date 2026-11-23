"""In-graph GPU timestamps for captured TinyJit graphs (lab only).

stamped(fn) returns (jit, read): the jit captures fn with one stamp launch after every kernel; each stamp reads the
kernel's first buffer (its output), so the queue flushes before it. read() returns [(kernel name, us)] for the last call.
"""
import collections, re, time
import numpy as np
from tinygrad import Tensor, TinyJit, dtypes
from tinygrad.uop.ops import Ops, UOp, KernelInfo
import tinygrad.engine.realize as realize

SRC = open('/data/alpamayo-bench/lab/src/stamp.cpp').read()
BIN = open('/data/alpamayo-bench/lab/src/stamp.hsaco','rb').read()
ANSI = re.compile(r'\x1b\[[0-9;]*m')
CAP = 32768

def _prog(ts, dep):
  sink = UOp.sink(ts.flatten()[0].store(dep.flatten()[0].load().cast(dtypes.uint64)), UOp.special(1,'gidx0'), UOp.special(32,'lidx0'),
                  arg=KernelInfo(name='lab_stamp'))
  return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.LINEAR, src=tuple(sink.toposort())), UOp(Ops.SOURCE, arg=SRC), UOp(Ops.BINARY, arg=BIN)))

def call_name(c):
  body = c.src[0]
  try:
    if body.op is Ops.PROGRAM: return ANSI.sub('', str(body.src[0].arg.name))
    if body.op is Ops.SINK: return ANSI.sub('', str(body.arg.name))
    return body.op.name.lower()
  except Exception as e:
    return f'?{body.op}'

def base_buffer(u):
  while u.op is not Ops.BUFFER:
    if not u.src: return None
    u = u.src[0]
  return u

class Stamped:
  def __init__(self, fn):
    self.ts = Tensor.zeros(CAP, dtype=dtypes.uint64).contiguous().realize()
    self.names, self.fn, self.nested = [], fn, None
    self.jit = TinyJit(fn)
    self.zero = TinyJit(lambda: self.ts.assign(Tensor.zeros(CAP, dtype=dtypes.uint64)).realize())
  def _add(self, linear):
    jit = self.jit
    out = []
    for c in linear.src:
      out.append(c)
      call = c.src[0] if c.op is Ops.AFTER else c
      if call.op is not Ops.CALL: continue
      buf = base_buffer(call.src[1]) if len(call.src) > 1 else None
      if buf is None: continue
      got = []
      class Collect:
        def add_linear(self, l): got.extend(l.src)
      saved = realize.capturing[0]
      realize.capturing[0] = Collect()  # the stamp's own linear; bypasses any compiler wrapper of the capture
      try:
        Tensor.custom_kernel(Tensor(self.ts.uop.base), Tensor(buf), fxn=_prog)[0].realize()
      finally:
        realize.capturing[0] = saved
      assert len(got) == 1, len(got)
      out.extend(got)
      self.names.append(call_name(call))
    jit._linears.append(linear.replace(src=tuple(out)))
  def __call__(self, *args):
    if self.jit.cnt == 1:
      self.names = []
      self.jit.add_linear = self._add
    return self.jit(*args)
  def compiled_names(self):
    out = []
    for call in self.jit.captured._linear.src:
      c = call.src[0] if call.op is Ops.AFTER else call
      aux = getattr(c.arg, 'aux', None)
      if aux is not None and hasattr(aux, 'kernels'):
        out.extend(ANSI.sub('', str(k[1])) for k in aux.kernels)
      else:
        out.append(call_name(c))
    return out
  def kernel_names(self):
    names = self.compiled_names()
    idx = [i for i, n in enumerate(names) if n == 'lab_stamp']
    # every stamp follows its kernel
    out = [names[i-1] for i in idx]
    assert len(out) == len(self.names), (len(out), len(self.names), len(names))
    return out
  def read(self, sync):
    sync()
    t = self.ts.numpy()
    n = int(t[0])
    st = t[1:1+n].astype(np.int64)
    return n, st
  def run(self, args, sync, n=5, warm=3, idle=0.0):
    """Per-kernel microseconds. n back-to-back runs after `warm` unmeasured ones (GPU clocks ramp up under load);
    returns names, the median over the last half of the runs, and all runs. With idle > 0 the host sleeps first and
    only one run is measured (cold start). The first kernel has no start stamp and is dropped."""
    for _ in range(max(warm, 2)): self(*args)
    names = self.kernel_names(); per = len(names)
    n = max(1, min(n, (CAP-1)//per))
    sync()
    if idle: time.sleep(idle); n = 1
    self.zero()
    for _ in range(n): self(*args)
    cnt, st = self.read(sync)
    assert cnt == n*per, (cnt, n, per)
    res = np.diff(st.reshape(n, per), axis=1)/100.0
    return names[1:], np.median(res[n//2:], axis=0), res

def table(names, us, top=30, per=1, label=''):
  agg, cnt = collections.defaultdict(float), collections.Counter()
  for n, u in zip(names, us): agg[n] += u; cnt[n] += 1
  total = sum(agg.values())
  print(f'== {label}: {len(names)} kernels, {total/1000:.3f} ms stamped total')
  print(f'{"ms":>8} {"share":>6} {"calls":>6} {"us/call":>8}  kernel')
  for k, v in sorted(agg.items(), key=lambda x: -x[1])[:top]:
    print(f'{v/1000/per:8.3f} {100*v/total:5.1f}% {cnt[k]/per:6.0f} {v/cnt[k]:8.1f}  {k[:70]}')
  return dict(agg), dict(cnt)
