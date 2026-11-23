names = {0: 'default', 1: 'fullscreen3d', 4: 'vr', 5: 'compute', 7: 'window3d', 8: 'directml'}
for bit in (1, 7, 8, 4, 5, 1):
  r = workload(bit); time.sleep(0.5)
  print(f'profile {names[bit]} ({r}): traj after idle', after_idle('traj', 0.3), after_idle('traj', 0.3), 'part0', after_idle('part0', 0.3, 3), 'vision', after_idle('vision', 0.3, 3))
