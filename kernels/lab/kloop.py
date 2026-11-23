"""Short per-kernel test loop (lab only): record real inputs once, then A/B candidate objects in seconds.

  REC = kloop.record(kind_or_predicate, run)       # run() executes a stage uncaptured; keeps the matching kernel calls
  kloop.ab(REC, {'cand': (source_text, binary)}, kind=..., sync=rt.model.synchronize)

Each variant runs in its own captured graph (one launch per recorded call, a timestamp launch after each). Outputs are
compared with the current object bit for bit on the GPU. Times are in-graph GPU microseconds per call.
"""
import gc, time
import numpy as np
from tinygrad import Tensor, dtypes
from tinygrad.uop.ops import Ops
import openpilot.selfdrive.alpamayo.selected.frozen as frozen
import openpilot.selfdrive.alpamayo.selected.kernels.frozen_ops as fo
import stamplib as sl

BITS = {1: dtypes.uint8, 2: dtypes.uint16, 4: dtypes.uint32, 8: dtypes.uint64}

def _empty(t):
  return t.uop.base.op is Ops.ALLOC  # Tensor.empty: outputs and scratch

def record(match, run, limit=28):
  calls, last = [], [None]
  orig_ck, orig_fp = Tensor.custom_kernel, fo.frozen_program
  def spy_fp(sink, kind, args):
    last[0] = kind
    return orig_fp(sink, kind, args)
  def spy(self, *lst, fxn, grad_fxn=None):
    last[0] = None
    probe = orig_ck(self, *lst, fxn=fxn, grad_fxn=grad_fxn)
    hit = (last[0] == match) if isinstance(match, str) else match(fxn)
    if not hit or len(calls) >= limit: return probe
    bufs = (self,)+lst
    empties = tuple(i for i, t in enumerate(bufs) if _empty(t))
    ins = tuple(t if i in empties else t.contiguous().realize() for i, t in enumerate(bufs))
    calls.append((ins, fxn, empties))
    return orig_ck(*ins, fxn=fxn, grad_fxn=grad_fxn)
  Tensor.custom_kernel, fo.frozen_program = spy, spy_fp
  try: run()
  finally: Tensor.custom_kernel, fo.frozen_program = orig_ck, orig_fp
  return calls

class swap:
  """Replace the frozen object of `kind` (all args) with (source, binary) while building a graph."""
  def __init__(self, kind, pair): self.kind, self.pair = kind, pair
  def __enter__(self):
    self.orig = frozen.object_pair
    if self.pair is not None:
      frozen.object_pair = lambda kind, args: self.pair if kind == self.kind else self.orig(kind, args)
  def __exit__(self, *a): frozen.object_pair = self.orig

def relaunch(grid, local):
  """fxn_map for hip_call kernels: the same frozen kind with other launch dimensions."""
  def fxn_map(fxn):
    v = dict(zip(fxn.__code__.co_freevars, (c.cell_contents for c in fxn.__closure__)))
    return fo.hip_program(v['kind'], v['args'], v['name'], grid, local, v['outputs'])
  return fxn_map

def replay(calls, kind=None, pair=None, fxn_map=None):
  outs = []
  with swap(kind, pair):
    for bufs, fxn, empties in calls:
      args = [Tensor.empty(*b.shape, dtype=b.dtype, device=b.device) if i in empties else b for i, b in enumerate(bufs)]
      res = Tensor.custom_kernel(*args, fxn=fxn if fxn_map is None else fxn_map(fxn))
      row = [res[i] for i in empties]
      Tensor.realize(*row)
      outs.append(row)
  return outs

def mismatches(a, b):
  total = []
  for ra, rb in zip(a, b):
    for x, y in zip(ra, rb):
      u = BITS[x.dtype.itemsize]
      total.append((x.bitcast(u) != y.bitcast(u)).sum().reshape(1))
  return Tensor.cat(*total).numpy().reshape(len(a), -1)

def ab(calls, variants, kind=None, sync=None, rounds=6, idle=0.3):
  """variants: {name: (source, binary)} for a frozen kind, {name: {'pair': (source, binary), 'grid': g, 'local': l}} for
  other launch dimensions, or {name: fxn_map} for generated kernels. Prints a table."""
  gc.collect()  # the runtime disables automatic collection; drop graphs of earlier runs
  graphs, outs = {}, {}
  def build(name, v):
    def f():
      if isinstance(v, dict): o = replay(calls, kind, v['pair'], relaunch(v['grid'], v['local']))
      else: o = replay(calls, kind, v if isinstance(v, tuple) else None, v if callable(v) else None)
      outs[name] = o
      return o[-1][0]
    s = sl.Stamped(f)
    for _ in range(3): s()
    return s
  t0 = time.monotonic()
  graphs['current'] = build('current', None)
  for name, v in variants.items(): graphs[name] = build(name, v)
  sync()
  print(f'{len(calls)} recorded calls; graphs captured in {time.monotonic()-t0:.1f} s')
  res = {n: {'warm': [], 'cold': []} for n in graphs}
  for r in range(rounds):
    for n, s in (list(graphs.items()) if r % 2 == 0 else list(graphs.items())[::-1]):
      for _ in range(3): s()
      sync(); s.zero()
      for _ in range(3): s()
      cnt, st = s.read(sync); per = cnt//3
      res[n]['warm'].append(np.median(np.diff(st.reshape(3, per), axis=1)[1:]/100.0))
      sync(); time.sleep(idle); s.zero(); s()
      cnt, st = s.read(sync)
      res[n]['cold'].append(np.median(np.diff(st)/100.0))
  base = outs['current']
  print(f'{"variant":<22} {"warm us/call":>12} {"vs current":>10} {"after idle":>11} {"vs current":>10}  bit-exact')
  bw, bc = np.median(res['current']['warm']), np.median(res['current']['cold'])
  for n in graphs:
    w, c = np.median(res[n]['warm']), np.median(res[n]['cold'])
    if n == 'current': exact = 'reference'
    else:
      mm = mismatches(base, outs[n])
      exact = 'yes' if int(mm.sum()) == 0 else f'NO: {int(mm.sum())} differing words in {int((mm.sum(axis=1) > 0).sum())}/{len(calls)} calls, per output {mm.sum(axis=0).tolist()}'
    print(f'{n:<22} {w:12.1f} {100*(w/bw-1):+9.1f}% {c:11.1f} {100*(c/bc-1):+9.1f}%  {exact}')
  for s in graphs.values(): s.jit.reset(); s.zero.reset()
  graphs.clear(); outs.clear(); del base
  gc.collect()
  return res
