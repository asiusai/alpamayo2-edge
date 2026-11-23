# Stamped inventory of the reasoning fallbacks: 16- and 32-row draft verifier, and the six-token serial block.
sl = lab_import("stamplib")
import gc; gc.collect()
from functools import partial
from tinygrad.helpers import PROFILE
from tinygrad.uop.ops import UOp
from openpilot.selfdrive.alpamayo.selected.serial_coordinates import block_six
PROFILE.value = 0
m = rt.model
if 'gen' not in ARGS:
  dcls = type(rt.decoder); o_gen = dcls.generate
  def spy_gen(self, prefill, logits, dc, ds, prompt, placed, max_tokens=64):
    ARGS['gen'] = (logits, dc, ds, prompt, placed); return o_gen(self, prefill, logits, dc, ds, prompt, placed, max_tokens)
  dcls.generate = spy_gen
  try: run_sequence('prep2', n=2)
  finally: dcls.generate = o_gen
logits, dc, ds, prompt, placed = ARGS['gen']
print('decode args', logits.shape, dc.shape, prompt, placed, 'draft', rt.decoder.draft)
draft = list(placed) if placed else list(rt.decoder.draft)
for count in (16, 32):
  name = f'verify{count}'
  toks = rt.decoder.proposal(draft+[draft[-1]]*(count-len(draft)), logits.device)
  ST[name] = sl.Stamped(lambda t, c, s_, p, st: rt.decoder.verify(t, c, s_, p, st))
  SA[name] = (toks, dc, ds, UOp.variable('draft_position', 0, m.cache[0].shape[3]-count).bind(prompt), UOp.variable('draft_step', 0, dc.shape[2]-count).bind(0))
one = rt.decoder.select_logits(rt.prefill.last_logits, 0) if rt.prefill.last_logits is not None else logits
one = one.clone().realize()
ST['block6'] = sl.Stamped(partial(block_six, m))
SA['block6'] = (one, dc, ds, UOp.variable('block_position', 0, m.cache[0].shape[3]-6).bind(prompt), UOp.variable('block_step', 0, dc.shape[2]-6).bind(0))
for name in ('verify16', 'verify32', 'block6'):
  t0 = time.monotonic()
  s, args = ST[name], SA[name]
  names, us, raw = s.run(args, m.synchronize, n=6)
  _, cold, _ = s.run(args, m.synchronize, idle=0.3)
  INV[name] = (names, us, cold)
  print(f'[{name}] {time.monotonic()-t0:.0f} s; back-to-back run totals ms {np.round(raw.sum(axis=1)/1000, 2).tolist()}; after 0.3 s idle {cold.sum()/1000:.2f} ms')
  sl.table(names, us, top=22, per=1, label=name)
