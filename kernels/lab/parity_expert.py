# Full-prediction parity with the 'signed' expert_block (instance patch; trajectory graph recaptured without the graph cache).
import types
import openpilot.selfdrive.alpamayo.selected.graph_cache as gcache
en = lab_import('expert_new')
m = rt.model
_dir = gcache._DIRECTORY; gcache._DIRECTORY = None
m.trajectory_jits.clear()
m.expert_block = types.MethodType(en.make('signed'), m)
try:
  t0 = time.monotonic()
  run_sequence('expert_signed', n=3)
  print('parity', PARITY['expert_signed'], PARITY['expert_signed'] == ['30f6623d22a0d6e1', 'b1555edfe779218d', '3a805827a3549e04'], round(time.monotonic()-t0), 's')
  a, k = ARGS['traj']; actions, ac, ass, length, _, offset = a
  f = lambda: m.trajectory(actions, ac, ass, length, 10, offset).realize()
  print('10-step trajectory graph, candidate', timeit(f, 12))
finally:
  del m.expert_block; m.trajectory_jits.clear(); gcache._DIRECTORY = _dir
f = lambda: m.trajectory(actions, ac, ass, length, 10, offset).realize()
print('10-step trajectory graph, current  ', timeit(f, 12))
