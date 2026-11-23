// LAUNCH 256 256
// Exhaustive check: the FC1 GELU with and without the exp2 range scaling, for every BF16 input. out[i] = xor of result bits.
#pragma clang fp contract(off)
#define DEV __attribute__((device,always_inline)) inline
extern "C" __attribute__((device)) unsigned long __ockl_get_local_id(unsigned);
extern "C" __attribute__((device)) unsigned long __ockl_get_group_id(unsigned);
DEV float frombf(unsigned short value) { return __builtin_bit_cast(float,((unsigned)value)<<16); }
DEV float gelu_old(float x) {
  float x3=(x*x)*x;
  float t=__builtin_fmaf(x3,__builtin_bit_cast(float,0x3d372713u),x);
  float y=t*__builtin_bit_cast(float,0xc0135761u);
  bool small=y<-126.0f;
  float e=__builtin_amdgcn_exp2f(small?y+64.0f:y);
  e=__builtin_amdgcn_ldexpf(e,small?-64:0);
  return x*__builtin_amdgcn_rcpf(1.0f+e);
}
DEV float gelu_new(float x) {
  float x3=(x*x)*x;
  float t=__builtin_fmaf(x3,__builtin_bit_cast(float,0x3d372713u),x);
  float y=t*__builtin_bit_cast(float,0xc0135761u);
  float e=__builtin_amdgcn_exp2f(y);
  return x*__builtin_amdgcn_rcpf(1.0f+e);
}
extern "C" __attribute__((global,amdgpu_flat_work_group_size(256,256)))
void gelu_check(unsigned *__restrict out, const unsigned *__restrict dep) {
  unsigned i=__ockl_get_group_id(0)*256+__ockl_get_local_id(0);
  float x=frombf((unsigned short)i);
  asm("" : "+v"(x));
  out[i]=__builtin_bit_cast(unsigned,gelu_old(x))^__builtin_bit_cast(unsigned,gelu_new(x));
}
