print(rt.gpu_metrics())
from tinygrad import Device
dev = Device['AMD']
ad = dev.iface.dev_impl
smu = ad.smu
print(type(ad).__name__, ad.ip_ver if hasattr(ad,'ip_ver') else None)
sm = smu.smu_mod
print(sm.__name__)
clks = (sm.PPCLK_GFXCLK, sm.PPCLK_UCLK, sm.PPCLK_FCLK, sm.PPCLK_SOCCLK)
print('dpm levels', smu.read_clocks(clks))
import os
print({k: os.environ.get(k) for k in ('AM_POWER_LIMIT','AM_PERF_LEVEL')})
