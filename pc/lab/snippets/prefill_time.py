import samples
s = samples.load(samples.turn_samples()[3])
rt.stage_sync = True
for i in range(4): p = rt.predict(s)
ts = []
for i in range(8):
  rt.parts_done = [None, None]
  p = rt.predict(s); ts.append(p['timings_ms']['prefill'])
print('full prefill ms', np.median(ts), 'vision', p['timings_ms']['vision'])
