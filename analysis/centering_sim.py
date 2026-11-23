# Closed-loop straight-road simulation of the live PlanSelector with an Alpamayo plan model fitted to Karel's drives.
# Plan relative to the lane (right positive) at x m ahead = A(x)*car offset + H(x)*car heading error*x + B(x) + noise,
# fitted 2026-11-22 on engaged straights (routes 000000c8/ca/cb). Steering: 0.2 s delay + 0.15 s first-order lag.
import sys, types, math, importlib.util, os
import numpy as np
sys.modules.setdefault('setproctitle', types.SimpleNamespace(getproctitle=lambda: '', setproctitle=lambda _: None))
spec = importlib.util.spec_from_file_location('planner_under_test', os.environ['PLANNER_PATH'])
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)

XA = np.array([0., 10, 20, 30, 40, 60, 200])
A = np.array([1., 1.00, .95, .86, .76, .55, .2])
H = np.array([-.7, -.67, -.63, -.57, -.51, -.40, -.3])
K_BIAS = 1.8e-4  # left curvature of the plan, 1/m


class Line:
  def __init__(self, x, y): self.x, self.y = list(x), list(y)


class Model:
  def __init__(self, lines, probs): self.laneLines, self.laneLineProbs = lines, probs


def run(selector, v=15., seconds=120., seed=0, plan_noise=.15, disturbance=2e-4, prob=.9, latency=.5):
  rng = np.random.default_rng(seed)
  dt = .01
  X = Y = psi = 0.  # world: lane centre Y=0, left positive; psi left positive
  Y = .3
  k_act, k_act_target, queue = 0., 0., []
  plans = []  # (ready_t, msg)
  offs = []
  d_k = 0.
  t_plan = 0.
  for i in range(int(seconds/dt)):
    t = i*dt; ns = int(t*1e9)+1_000_000_000
    selector.motion(ns, v, psi)
    if t >= t_plan:
      t_plan += .1
      # plan in the car frame at query (left positive)
      xs = v*np.arange(1, 65)*.1
      off_r, head_r = -Y, psi  # car offset right+, lane heading relative to the car right+
      c = -off_r+0*xs  # lane centre right+ in the car frame (straight lane): c(x) = -off + psi*x
      c = c+head_r*xs
      plan_r = c+np.interp(xs, XA, A)*off_r+np.interp(xs, XA, H)*head_r*xs-.5*K_BIAS*xs*xs
      plan_r += rng.normal()*plan_noise*(xs/40)+rng.normal()*plan_noise*(xs/40)**2
      msg = types.SimpleNamespace(queryMonoTime=ns, readyMonoTime=ns+int(latency*1e9), x=list(xs), y=list(-plan_r),
                                  yaw=list(np.gradient(-plan_r, xs)))
      plans.append((t+latency, msg))
    while plans and plans[0][0] <= t:
      _, msg = plans.pop(0)
      selector.receive(msg, ns, msg.readyMonoTime)
    if i % 5 == 0:  # 20 Hz modelV2
      xs = np.linspace(0, 100, 33)
      lc = Y/np.cos(psi) + np.tan(psi)*xs  # lane centre right+ in the car frame (car left of centre: centre to the right)
      model = Model([Line(xs, lc-5.1), Line(xs, lc-1.65), Line(xs, lc+1.65), Line(xs, lc+5.1)], [.5, prob, prob, .5])
      selector.observe_lanes(model, ns)
      a = selector.candidate_action(ns, v, .35, .3)
      cmd = -a['desiredCurvature'] if a else k_act  # left positive
      queue.append((t+.2, cmd))
    while queue and queue[0][0] <= t:
      k_act_target = queue.pop(0)[1]
    k_act += dt/.15*(k_act_target-k_act)
    d_k += -d_k*dt/2.+rng.normal()*disturbance*math.sqrt(dt)  # slow curvature disturbance (crown, wind, alignment)
    psi += v*(k_act+d_k)*dt
    X += v*math.cos(psi)*dt; Y += v*math.sin(psi)*dt
    if t > 10: offs.append(Y)
  o = np.array(offs)
  return o.mean(), np.sqrt((o*o).mean()), np.abs(o).max(), np.mean(np.abs(o) > .5)


if __name__ == '__main__':
  configs = {'stock (live)': {}, 'bias only': dict(plan_curvature_bias=K_BIAS), 'centring .5': dict(lane_centering=.5),
             'centring .8': dict(lane_centering=.8), 'centring .8 + bias': dict(lane_centering=.8, plan_curvature_bias=K_BIAS),
             'centring 1 + bias': dict(lane_centering=1., plan_curvature_bias=K_BIAS)}
  for v in (12., 20., 30.):
    for prob in (.9, .55):
      print(f'-- v {v:.0f} m/s, lane prob {prob}')
      for name, cfg in configs.items():
        r = np.array([run(mod.PlanSelector(**cfg), v=v, seed=s, prob=prob) for s in range(4)])
        print(f'  {name:20s} mean {r[:,0].mean():+.2f} m (left+)  rms {r[:,1].mean():.2f}  max {r[:,2].max():.2f}  >0.5 m {r[:,3].mean()*100:.0f}%')
