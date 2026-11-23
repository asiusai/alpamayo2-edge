// gfx1100 (RDNA3) replacements for the gfx12-only builtins used by the frozen selected kernels.
// FP8 E4M3FN conversion: exact decode; round-to-nearest-even encode with subnormals, saturating above 464, NaN kept.
// gfx12 WMMA: emulated with gfx11 WMMA. Operands move between lane halves with ds_bpermute. The first (A) operand
// rows are permuted so the gfx11 accumulator layout (row 2i+h) equals the gfx12 layout (row i+8h): C and D need no
// change. FP8 values convert exactly to FP16; products are exact in FP32. Only the hardware summation order can differ.
typedef _Float16 __pcs_h16 __attribute__((ext_vector_type(16)));
typedef short __pcs_s16 __attribute__((ext_vector_type(16)));
typedef short __pcs_s8 __attribute__((ext_vector_type(8)));
typedef int __pcs_i2 __attribute__((ext_vector_type(2)));
typedef int __pcs_i4 __attribute__((ext_vector_type(4)));
typedef float __pcs_f8 __attribute__((ext_vector_type(8)));
#define __PCS_DEV __attribute__((device,always_inline)) inline
extern "C" __attribute__((device)) unsigned long __ockl_get_local_id(unsigned);

__PCS_DEV float __pcs_fp8_to_f32(unsigned b) {
  b &= 0xffu;
  unsigned sign = (b & 0x80u) << 24, e = (b >> 3) & 15u, m = b & 7u;
  if ((b & 0x7fu) == 0x7fu) return __builtin_bit_cast(float, sign | 0x7fc00000u);
  if (e == 0u) { float v = (float)m * 0.001953125f; return __builtin_bit_cast(float, __builtin_bit_cast(unsigned, v) | sign); }
  return __builtin_bit_cast(float, sign | ((e + 120u) << 23) | (m << 20));
}
__PCS_DEV float __pcs_cvt_f32_fp8(int src, int sel) { return __pcs_fp8_to_f32(((unsigned)src) >> (8 * sel)); }
__PCS_DEV unsigned __pcs_f32_to_fp8(float v) {
  unsigned b = __builtin_bit_cast(unsigned, v), sign = (b >> 24) & 0x80u, a = b & 0x7fffffffu;
  if (a > 0x7f800000u) return sign | 0x7fu;
  float f = __builtin_bit_cast(float, a);
  if (f > 464.0f) return sign | 0x7eu;
  if (f < 0.015625f) return sign | (unsigned)__builtin_rintf(f * 512.0f);
  unsigned r = (a + 0x7ffffu + ((a >> 20) & 1u)) >> 20;
  return sign | (r - 960u);
}
__PCS_DEV int __pcs_cvt_pk_fp8_f32(float a, float b, int old, bool hi) {
  unsigned pk = __pcs_f32_to_fp8(a) | (__pcs_f32_to_fp8(b) << 8);
  return hi ? (int)(((unsigned)old & 0xffffu) | (pk << 16)) : (int)(((unsigned)old & 0xffff0000u) | pk);
}
__PCS_DEV unsigned __pcs_lane() { return __builtin_amdgcn_mbcnt_lo(~0u, 0u); }
__PCS_DEV int __pcs_bperm(unsigned src, int v) { return __builtin_amdgcn_ds_bpermute((int)(src * 4u), v); }
// Two FP8 bytes (bits 0-7, 8-15 of x) to two exact FP16 values: (b&0x7f)<<7 is the FP16 pattern scaled by 2^-8.
__PCS_DEV void __pcs_fp8x4_to_h(unsigned x, __pcs_h16 &out, int at) {
  #pragma unroll
  for (int i = 0; i < 4; i++) {
    unsigned byte = (x >> (8 * i)) & 0xffu;
    _Float16 h = __builtin_bit_cast(_Float16, (unsigned short)(((byte & 0x7fu) << 7)));
    h = h * (_Float16)256.0f;
    out[at + i] = (byte & 0x80u) ? -h : h;
  }
}
// Lane exchange with VALU permutes (no LDS): `own` is this half's value for lane rho' = sel(rho), `other` the other half's.
// A operand: lane rho needs row (rho>>1) + 8*(rho&1) of both halves; B operand: column rho of both halves.
__PCS_DEV void __pcs_rows(int v, int &lo, int &hi) {
  unsigned h = (__pcs_lane() >> 4) & 1u;
  int own = __builtin_amdgcn_permlane16(v, v, 0xB3A29180u, 0xF7E6D5C4u, false, false);
  int other = __builtin_amdgcn_permlanex16(own, own, 0x76543210u, 0xFEDCBA98u, false, false);
  lo = h ? other : own; hi = h ? own : other;
}
__PCS_DEV void __pcs_cols(int v, int &lo, int &hi) {
  unsigned h = (__pcs_lane() >> 4) & 1u;
  int other = __builtin_amdgcn_permlanex16(v, v, 0x76543210u, 0xFEDCBA98u, false, false);
  lo = h ? other : v; hi = h ? v : other;
}
__PCS_DEV __pcs_f8 __pcs_wmma_fp8(__pcs_i2 a, __pcs_i2 b, __pcs_f8 c) {
  __pcs_h16 ah, bh;
  int lo, hi;
  __pcs_rows(a[0], lo, hi); __pcs_fp8x4_to_h((unsigned)lo, ah, 0); __pcs_fp8x4_to_h((unsigned)hi, ah, 8);
  __pcs_rows(a[1], lo, hi); __pcs_fp8x4_to_h((unsigned)lo, ah, 4); __pcs_fp8x4_to_h((unsigned)hi, ah, 12);
  __pcs_cols(b[0], lo, hi); __pcs_fp8x4_to_h((unsigned)lo, bh, 0); __pcs_fp8x4_to_h((unsigned)hi, bh, 8);
  __pcs_cols(b[1], lo, hi); __pcs_fp8x4_to_h((unsigned)lo, bh, 4); __pcs_fp8x4_to_h((unsigned)hi, bh, 12);
  return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(ah, bh, c);
}
__PCS_DEV __pcs_f8 __pcs_wmma_bf16(__pcs_s8 a, __pcs_s8 b, __pcs_f8 c) {
  __pcs_i4 ai = __builtin_bit_cast(__pcs_i4, a), bi = __builtin_bit_cast(__pcs_i4, b);
  __pcs_i4 a0, a1, b0, b1;
  #pragma unroll
  for (int i = 0; i < 4; i++) {
    int lo, hi;
    __pcs_rows(ai[i], lo, hi); a0[i] = lo; a1[i] = hi;
    __pcs_cols(bi[i], lo, hi); b0[i] = lo; b1[i] = hi;
  }
  __pcs_s16 as, bs;
  __pcs_s8 x;
  x = __builtin_bit_cast(__pcs_s8, a0); for (int i = 0; i < 8; i++) as[i] = x[i];
  x = __builtin_bit_cast(__pcs_s8, a1); for (int i = 0; i < 8; i++) as[8 + i] = x[i];
  x = __builtin_bit_cast(__pcs_s8, b0); for (int i = 0; i < 8; i++) bs[i] = x[i];
  x = __builtin_bit_cast(__pcs_s8, b1); for (int i = 0; i < 8; i++) bs[8 + i] = x[i];
  return __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(as, bs, c);
}
// gfx12 GLOBAL_LOAD_TR_B128 (16-bit): lane j of half h receives column j of the 8x16 block whose row segments the 16 lanes
// of that half address. In the selected kernels a lane addresses row (l%8) of its half at column 8*((l/8)%2) with a row
// stride of W elements, so lane l reads its column directly: base - (l%8)*W + (l%8), then 8 rows of stride W.
__PCS_DEV __pcs_s8 __pcs_load_tr(const unsigned short *p, int lane, int stride) {
  int r = lane & 7;
  const unsigned short *q = p - r * stride + r;
  __pcs_s8 out;
  #pragma unroll
  for (int e = 0; e < 8; e++) out[e] = (short)q[e * stride];
  return out;
}
// Generic form: element e of lane l comes from the address of lane (l & ~7) + e, at element offset l % 8.
__PCS_DEV __pcs_s8 __pcs_load_tr_any(const unsigned short *p) {
  unsigned lane = __pcs_lane() & 31u;
  unsigned long a = (unsigned long)p;
  int lo = (int)(unsigned)a, hi = (int)(unsigned)(a >> 32);
  __pcs_s8 out;
  #pragma unroll
  for (unsigned e = 0; e < 8u; e++) {
    unsigned src = (lane & ~7u) + e;
    unsigned long q = ((unsigned long)(unsigned)__pcs_bperm(src, hi) << 32) | (unsigned)__pcs_bperm(src, lo);
    out[e] = (short)((const unsigned short *)q)[lane & 7u];
  }
  return out;
}
// LDS form: each lane stores its own 16-byte row segment in a per-wave LDS slot, then reads its column (8 u16 reads).
// Requires a one-dimensional workgroup of at most 512 threads.
__PCS_DEV __pcs_s8 __pcs_load_tr_lds(const unsigned short *p) {
  __attribute__((shared)) __pcs_s8 __pcs_tr_slots[16][32];
  unsigned lane = __pcs_lane() & 31u, wave = ((unsigned)__ockl_get_local_id(0) >> 5) & 15u;
  __pcs_s8 mine = *(const __pcs_s8 *)p;
  __builtin_amdgcn_wave_barrier();
  __pcs_tr_slots[wave][lane] = mine;
  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront");
  __builtin_amdgcn_wave_barrier();
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront");
  const unsigned short *base = (const unsigned short *)&__pcs_tr_slots[wave][lane & ~7u];
  __pcs_s8 out;
  #pragma unroll
  for (unsigned e = 0; e < 8u; e++) out[e] = (short)base[e * 8u + (lane & 7u)];
  __builtin_amdgcn_wave_barrier();
  return out;
}
#define __builtin_amdgcn_cvt_f32_fp8(src, sel) __pcs_cvt_f32_fp8((src), (sel))
#define __builtin_amdgcn_cvt_pk_fp8_f32(a, b, old, hi) __pcs_cvt_pk_fp8_f32((a), (b), (old), (hi))
#define __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a, b, c) __pcs_wmma_fp8((a), (b), (c))
#define __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a, b, c) __pcs_wmma_bf16((a), (b), (c))
#define __builtin_amdgcn_global_load_tr_b128_v8i16(p) __pcs_load_tr_lds((const unsigned short *)(p))
