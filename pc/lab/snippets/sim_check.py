# Closed-loop drift check on the PC: recorded controller (reference) and the deployed tracker (`Live`) driven by model plans.
# Use straight or gently curving routes (00000080, 0000005f); 00000091 has a 42-degree junction turn without navigation.
exec(open('/home/john/alpamayo2-edge/pc/tools/pc_sim.py').read())
run_device(CLOSEDLOOP+'/sim.py'); run_device(CLOSEDLOOP+'/controllers.py')
print('scripts from', CLOSEDLOOP)
for name in globals().get('ROUTES_CHECK', ['00000080--1c108fd104--3', '0000005f--1f057bc12e--3']):
  route = TRIMS+'/'+name
  r = load(route)
  for cname, c in (('recorded', Recorded(r)), ('live', Live(1.))):
    t = time.time(); rec = run(route, c, seconds=float(globals().get('SECONDS', 20.)))
    print(name, cname, round(time.time()-t, 1), 's', {k: round(v, 2) for k, v in summarize(rec).items()}, flush=True)
