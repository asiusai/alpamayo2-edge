# Closed-loop straight-road simulation of the live PlanSelector with plan-response models of different students.
# Plan relative to the lane at x m ahead = A(x)*car offset + H(x)*heading error*x - bias + noise (A = 1, H = -1: no correction).
# Question: does a student that corrects more firmly oscillate with the car's plan delay, and how much does delay matter?
# Usage: PLANNER_PATH=.../selfdrive/alpamayo/planner.py pingpong_sim.py
import sys, types, math, importlib.util, os
import numpy as np
sys.modules.setdefault('setproctitle', types.SimpleNamespace(getproctitle=lambda: '', setproctitle=lambda _: None))
spec = importlib.util.spec_from_file_location('planner_under_test', os.environ['PLANNER_PATH'])
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
XA = np.array([0., 10, 20, 40, 60, 200])
MODELS = {  # (A, H, curvature bias 1/m) from the shifted-view and drive-test scores
  'stock':  (np.array([1., .97, .89, .75, .66, .3]), -np.array([.85, .78, .70, .44, .33, .2]), 1.3e-4),
  'lat10':  (np.array([1., .97, .89, .76, .73, .3]), -np.array([.85, .78, .69, .38, .26, .2]), 0.6e-4),
  'lat2r':  (np.array([1., .95, .84, .59, .44, .2]), -np.array([.80, .72, .62, .26, .12, .1]), 0.),
}
class Line:
  def __init__(self, x, y): self.x, self.y = list(x), list(y)
class Model:
  def __init__(self, lines, probs): self.laneLines, self.laneLineProbs = lines, probs

def run(selector, A, H, bias, v=24., seconds=180., seed=0, plan_noise=.15, disturbance=2e-4, prob=.9, latency=.35, period=.25):
  rng = np.random.default_rng(seed); dt = .01
  X = psi = 0.; Y = .3
  k_act = k_act_target = d_k = 0.; queue, plans, offs = [], [], []
  t_plan = 0.
  for i in range(int(seconds/dt)):
    t = i*dt; ns = int(t*1e9)+1_000_000_000
    selector.motion(ns, v, psi)
    if t >= t_plan:
      t_plan += period
      xs = v*np.arange(1, 65)*.1
      off_r, head_r = -Y, psi
      c = -off_r+head_r*xs
      plan_r = c+np.interp(xs, XA, A)*off_r+np.interp(xs, XA, H)*head_r*xs-.5*bias*xs*xs
      plan_r += rng.normal()*plan_noise*(xs/40)+rng.normal()*plan_noise*(xs/40)**2
      msg = types.SimpleNamespace(queryMonoTime=ns, readyMonoTime=ns+int(latency*1e9), x=list(xs), y=list(-plan_r), yaw=list(np.gradient(-plan_r, xs)))
      plans.append((t+latency, msg))
    while plans and plans[0][0] <= t:
      _, msg = plans.pop(0); selector.receive(msg, ns, msg.readyMonoTime)
    if i % 5 == 0:
      xs = np.linspace(0, 100, 33); lc = Y/np.cos(psi)+np.tan(psi)*xs
      selector.observe_lanes(Model([Line(xs, lc-5.1), Line(xs, lc-1.65), Line(xs, lc+1.65), Line(xs, lc+5.1)], [.5, prob, prob, .5]), ns)
      a = selector.candidate_action(ns, v, .35, .3)
      queue.append((t+.2, -a['desiredCurvature'] if a else k_act))
    while queue and queue[0][0] <= t:
      k_act_target = queue.pop(0)[1]
    k_act += dt/.15*(k_act_target-k_act)
    d_k += -d_k*dt/2.+rng.normal()*disturbance*math.sqrt(dt)
    psi += v*(k_act+d_k)*dt; X += v*math.cos(psi)*dt; Y += v*math.sin(psi)*dt
    if t > 20: offs.append(Y)
  return np.array(offs)

def describe(o):
  y = o[::10]; y = y-np.convolve(y, np.ones(200)/200, 'same'); y = y[100:-100]
  f = np.fft.rfftfreq(len(y), .1); P = np.abs(np.fft.rfft(y*np.hanning(len(y))))**2; band = (f > 1/20) & (f < 1/1.5)
  per = 1/f[band][np.argmax(P[band])]; m = max(int(per/.1), 10)
  return o.mean(), y.std(), per, np.median([np.ptp(y[i:i+m]) for i in range(0, len(y)-m, m)])

if __name__ == '__main__':
  print('model  plan delay s  plan period s  noise | mean m (left+)  wander sd m  period s  peak-to-peak m')
  for name, (A, H, bias) in MODELS.items():
    for latency, period in ((.20, .20), (.35, .25), (.35, .45), (.60, .45)):
      for noise in (.05, .15):
        r = np.array([describe(run(mod.PlanSelector(), A, H, bias, latency=latency, period=period, plan_noise=noise, seed=s)) for s in range(3)])
        print(f'{name:6s} {latency:10.2f} {period:13.2f} {noise:6.2f} | {r[:,0].mean():+8.2f} {r[:,1].mean():12.3f} {np.median(r[:,2]):9.1f} {r[:,3].mean():10.2f}')
