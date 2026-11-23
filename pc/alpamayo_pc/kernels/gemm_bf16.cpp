// BF16 GEMM for gfx1100 on the selected packed layout: out[m,n] = bf16(sum_k x[m,k] w[n,k] * (xs*ws)) and, with
// RESIDUAL, bf16(residual + bf16(...)) as the frozen prefill objects round. x and w use amd_gemm.pack: each 16x16 tile
// is 256 contiguous elements at (row%16)*4 + (k%16/4)*64 + k%4. FP32 accumulation in WMMA; FP8-valued operands are exact.
// Workgroup: 4 waves in 2x2, each TM x TN 16x16 tiles (WG tile 32*TM x 32*TN); K stages of 32 through double-buffered LDS.
#define M __M__
#define N __N__
#define K __K__
#define RESIDUAL __RESIDUAL__
#ifndef TM
#define TM 2
#endif
#ifndef TN
#define TN 4
#endif
typedef unsigned short u16;
typedef short s16v __attribute__((ext_vector_type(16)));
typedef short s8v __attribute__((ext_vector_type(8)));
typedef float f8v __attribute__((ext_vector_type(8)));
typedef int i4v __attribute__((ext_vector_type(4)));
typedef int i2v __attribute__((ext_vector_type(2)));
extern "C" __attribute__((device)) unsigned long __ockl_get_local_id(unsigned);
extern "C" __attribute__((device)) unsigned long __ockl_get_group_id(unsigned);
#define DEV __attribute__((device)) __attribute__((always_inline)) inline
#define BARRIER() do { __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup"); __builtin_amdgcn_s_barrier(); __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup"); } while (0)
DEV unsigned short bf(float value) {  // exact pinned tinygrad BF16 cast (as the frozen objects)
  unsigned b = __builtin_bit_cast(unsigned, value);
  unsigned regular = b + ((b >> 16) & 1) + 0x7fffu;
  unsigned special = b | (((b & 0xffffu) != 0) << 16);
  unsigned mask = 0u - (((-b & 0x7f800000u) != 0));
  asm("" : "+v"(mask));
  return (unsigned short)(((regular & mask) | (special & ~mask)) >> 16);
}
// Fragment of one 16x16 tile in LDS: lane row r gets k = 0..15 as four 8-byte groups 64 elements apart.
DEV s16v frag(const u16 *tile, int r) {
  i2v g0 = *(const i2v *)(tile + 0 * 64 + r * 4), g1 = *(const i2v *)(tile + 1 * 64 + r * 4);
  i2v g2 = *(const i2v *)(tile + 2 * 64 + r * 4), g3 = *(const i2v *)(tile + 3 * 64 + r * 4);
  i4v lo = {g0[0], g0[1], g1[0], g1[1]}, hi = {g2[0], g2[1], g3[0], g3[1]};
  s16v out;
  s8v a = __builtin_bit_cast(s8v, lo), b = __builtin_bit_cast(s8v, hi);
  #pragma unroll
  for (int e = 0; e < 8; e++) { out[e] = a[e]; out[8 + e] = b[e]; }
  return out;
}
extern "C" __attribute__((global, amdgpu_flat_work_group_size(128, 128)))
void pc_gemm_bf16(u16 *__restrict out, const u16 *__restrict x, const u16 *__restrict w, const float *__restrict xs,
                  const float *__restrict ws
#if RESIDUAL
                  , const u16 *__restrict residual
#endif
                  ) {
  constexpr int KT = K / 16, AT = 2 * TM * 2, BT = 2 * TN * 2, MT = (M + 15) / 16;  // tiles per stage (row/col tiles x 2 k tiles)
  __attribute__((shared)) __attribute__((aligned(16))) u16 As[2][AT * 256];
  __attribute__((shared)) __attribute__((aligned(16))) u16 Bs[2][BT * 256];
  const int tid = __ockl_get_local_id(0), lane = tid & 31, wave = __builtin_amdgcn_readfirstlane(tid >> 5);
  const int block = __ockl_get_group_id(0), nblocks = N / (32 * TN);
  const int mb = block / nblocks, nb = block % nblocks;
  const int r = lane & 15, h = lane >> 4;
  const int wm = (wave >> 1) * TM, wn = (wave & 1) * TN;  // this wave's first row tile and col tile in the WG tile
  // Global tile bases (in elements) for stage 0: tile (row tile t, k tile kt) at (t*KT + kt)*256.
  const u16 *ag = x + (long)(mb * 2 * TM) * KT * 256, *bg = w + (long)(nb * 2 * TN) * KT * 256;
  f8v acc[TM][TN];
  #pragma unroll
  for (int i = 0; i < TM; i++)
    #pragma unroll
    for (int j = 0; j < TN; j++) acc[i][j] = (f8v){0};
  // Each stage moves AT + BT tiles of 512 B = (AT + BT) * 32 chunks of 16 B; Q per thread.
  constexpr int Q = (AT + BT) * 32 / 128;
  auto load = [&](int stage, i4v *regs) {
    #pragma unroll
    for (int q = 0; q < Q; q++) {
      int chunk = tid + q * 128, tile = chunk >> 5, part = chunk & 31;  // 32 x 16 B per tile
      if (tile < AT) {
        int rt = mb * 2 * TM + (tile >> 1);
        regs[q] = rt < MT ? *(const i4v *)(ag + ((long)(tile >> 1) * KT + stage * 2 + (tile & 1)) * 256 + part * 8) : (i4v){0};
      } else {
        regs[q] = *(const i4v *)(bg + ((long)((tile - AT) >> 1) * KT + stage * 2 + ((tile - AT) & 1)) * 256 + part * 8);
      }
    }
  };
  auto store = [&](int buf, const i4v *regs) {
    #pragma unroll
    for (int q = 0; q < Q; q++) {
      int chunk = tid + q * 128, tile = chunk >> 5, part = chunk & 31;
      u16 *dst = tile < AT ? &As[buf][tile * 256] : &Bs[buf][(tile - AT) * 256];
      *(i4v *)(dst + part * 8) = regs[q];
    }
  };
  i4v regs[Q];
  load(0, regs); store(0, regs);
  BARRIER();
  constexpr int STAGES = K / 32;
  for (int s = 0; s < STAGES; s++) {
    int buf = s & 1;
    if (s + 1 < STAGES) load(s + 1, regs);
    #pragma unroll
    for (int kt = 0; kt < 2; kt++) {
      s16v a[TM], b[TN];
      #pragma unroll
      for (int i = 0; i < TM; i++) a[i] = frag(&As[buf][((wm + i) * 2 + kt) * 256], r);
      #pragma unroll
      for (int j = 0; j < TN; j++) b[j] = frag(&Bs[buf][((wn + j) * 2 + kt) * 256], r);
      #pragma unroll
      for (int i = 0; i < TM; i++)
        #pragma unroll
        for (int j = 0; j < TN; j++) acc[i][j] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a[i], b[j], acc[i][j]);
    }
    if (s + 1 < STAGES) { store(buf ^ 1, regs); }
    BARRIER();
  }
  const float scale = xs[0] * ws[0];
  #pragma unroll
  for (int i = 0; i < TM; i++)
    #pragma unroll
    for (int j = 0; j < TN; j++)
      #pragma unroll
      for (int e = 0; e < 8; e++) {
        int row = (mb * 2 * TM + wm + i) * 16 + 2 * e + h, col = (nb * 2 * TN + wn + j) * 16 + r;
        if (row < M) {
          unsigned short v = bf(acc[i][j][e] * scale);
#if RESIDUAL
          v = bf(__builtin_bit_cast(float, ((unsigned)residual[(long)row * N + col]) << 16) + __builtin_bit_cast(float, ((unsigned)v) << 16));
#endif
          out[(long)row * N + col] = v;
        }
      }
}
