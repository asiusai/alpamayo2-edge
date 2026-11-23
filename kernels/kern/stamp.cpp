// LAUNCH 1 32
// Lab timestamp: appends SHADER_REALTIME (100 MHz) to ts[1+ts[0]] and increments ts[0]. dep only orders the launch.
extern "C" __attribute__((device)) unsigned long __ockl_get_local_id(unsigned);
extern "C" __attribute__((global,amdgpu_flat_work_group_size(32,32)))
void lab_stamp(unsigned long *__restrict ts, const unsigned char *__restrict dep) {
  if (__ockl_get_local_id(0) == 0) { unsigned long i = ts[0]; ts[1+i] = __builtin_amdgcn_s_sendmsg_rtnl(0x83); ts[0] = i+1; }
}
