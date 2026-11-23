# Closed-loop smoke test on the PC: one recorded route, feedforward controller, 6 s.
exec(open('/home/john/alpamayo2-edge/pc/tools/pc_sim.py').read())
run_device(CLOSEDLOOP+'/sim.py'); run_device(CLOSEDLOOP+'/controllers.py')
route = TRIMS+'/00000091--c1af5805a9--0'
r = load(route)
t = time.time()
rec = run(route, Feedforward(1.), seconds=6.)
print('closed loop 6 s', round(time.time()-t, 1), 's wall', summarize(rec))
