# Host CPU profile of rolling-mode predictions (unsynchronized stages).
import cProfile, pstats, io, sys
sys.path.insert(0, '/home/john/alpamayo2-edge/pc/tools')
import bench, samples
data = [samples.load(p) for p in samples.karel_samples()]
it = bench.stream(globals().get('MODE', 'rolling'), data)
rt.stage_sync = False
for _ in range(20): rt.predict(next(it))
rt.synchronize()
pr = cProfile.Profile()
t = time.perf_counter(); pr.enable()
for _ in range(10): rt.predict(next(it))
rt.synchronize(); pr.disable()
print('per call', (time.perf_counter()-t)*100, 'ms')
s = io.StringIO(); pstats.Stats(pr, stream=s).sort_stats('cumulative').print_stats(45); print(s.getvalue()[:9000])
s = io.StringIO(); pstats.Stats(pr, stream=s).sort_stats('tottime').print_stats(25); print(s.getvalue()[:6000])
