NO_RELOAD = True
print('have pairs', 'pairs' in ARGS, [t.shape for t in ARGS['pairs']])
VPAIRS = ARGS['pairs']
exec(open('/data/alpamayo-bench/lab/src/refreeze_plan_kopt.py').read())
