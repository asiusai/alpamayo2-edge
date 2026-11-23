// Causal prefill attention for gfx1100 (RDNA3, wave32 WMMA). Same arithmetic as the frozen gfx1200 object:
// FP32 scores x 1/sqrt(128), online softmax over 32-key blocks with exp2_exact, BF16 probabilities, FP32 PV
// accumulation, out = BF16(acc / denominator). Reads K and V rows below OFFSET+ROWS straight from the
// (2,1,8,CAP,128) KV cache. One workgroup (4 waves) holds 64 query rows of one head; K and V blocks go through LDS.
// Scores are computed transposed (S^T = K Q^T) so each lane holds one query: row softmax needs one lane exchange.
#define ROWS __ROWS__
#define OFFSET __OFFSET__
#define CAP 2304
typedef unsigned short u16;
typedef short s16v __attribute__((ext_vector_type(16)));
typedef short s8v __attribute__((ext_vector_type(8)));
typedef float f8v __attribute__((ext_vector_type(8)));
typedef int i4v __attribute__((ext_vector_type(4)));
extern "C" __attribute__((device)) unsigned long __ockl_get_local_id(unsigned);
extern "C" __attribute__((device)) unsigned long __ockl_get_group_id(unsigned);
#define DEV __attribute__((device)) __attribute__((always_inline)) inline
#define BARRIER() do { __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup"); __builtin_amdgcn_s_barrier(); __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup"); } while (0)
DEV short tobf(float x) { unsigned b = __builtin_bit_cast(unsigned, x); return (short)((b + 0x7fffu + ((b >> 16) & 1u)) >> 16); }
DEV float exp2_exact(float d) {
  const float C = 1.4426950408889634f;
  float t = d * C; bool s = t < -126.0f;
  return __builtin_amdgcn_exp2f(__builtin_fmaf(C, d, s ? 64.0f : 0.0f)) * (s ? 0x1p-64f : 1.0f);
}
DEV int xhalf(int v) { return __builtin_amdgcn_permlanex16(v, v, 0x76543210u, 0xFEDCBA98u, false, false); }
DEV float xhalff(float v) { return __builtin_bit_cast(float, xhalf(__builtin_bit_cast(int, v))); }

extern "C" __attribute__((global, amdgpu_flat_work_group_size(128, 128)))
void pc_prefill_attention(u16 *__restrict out, const u16 *__restrict q, const u16 *__restrict kv) {
#pragma clang fp contract(off)
  constexpr int H = 16, KH = 8, W = 128, BK = 32, KS = W + 8, VS = BK + 8, TILES = (ROWS + 63) / 64;
  __attribute__((shared)) __attribute__((aligned(16))) u16 Ks[BK * KS];
  __attribute__((shared)) __attribute__((aligned(16))) u16 Vt[W * VS];
  const int tid = __ockl_get_local_id(0), lane = tid & 31, wave = __builtin_amdgcn_readfirstlane(tid >> 5);
  const int block = __ockl_get_group_id(0);
  const int head = block % H, tile = (TILES - 1) - block / H, kh = head / (H / KH);
  const int h = lane >> 4, c = lane & 15;
  const int qrow = tile * 64 + wave * 16 + c;
  const bool qvalid = qrow < ROWS;
  const int qpos = OFFSET + qrow;
  const int last_row = (tile * 64 + 63 < ROWS ? tile * 64 + 63 : ROWS - 1);
  const int kend = OFFSET + last_row + 1;
  const int wave_last = OFFSET + (tile * 64 + wave * 16 + 15 < ROWS ? tile * 64 + wave * 16 + 15 : ROWS - 1);
  const u16 *qp = q + ((long)head * ROWS + (qvalid ? qrow : 0)) * W;
  s16v qf[8];
  #pragma unroll
  for (int d = 0; d < 8; d++) qf[d] = *(const s16v *)(qp + d * 16);
  const u16 *kb = kv + (long)kh * CAP * W, *vb = kv + (long)(KH + kh) * CAP * W;
  f8v acc[8];
  #pragma unroll
  for (int d = 0; d < 8; d++) acc[d] = (f8v){0};
  float maxima = -__builtin_inff(), denom = 0.0f;
  for (int start = 0; start < kend; start += BK) {
    BARRIER();
    #pragma unroll
    for (int r = 0; r < 4; r++) {
      int idx = tid + r * 128, key = idx >> 4, dim = (idx & 15) * 8;
      s8v kvec = *(const s8v *)(kb + (long)(start + key) * W + dim);
      s8v vvec = *(const s8v *)(vb + (long)(start + key) * W + dim);
      *(s8v *)(Ks + key * KS + dim) = kvec;
      int pos = (key >> 4) * 16 + 8 * (key & 1) + ((key & 15) >> 1);
      #pragma unroll
      for (int e = 0; e < 8; e++) Vt[(dim + e) * VS + pos] = vvec[e];
    }
    BARRIER();
    if (start > wave_last) continue;
    f8v s[2];
    #pragma unroll
    for (int t = 0; t < 2; t++) {
      s[t] = (f8v){0};
      #pragma unroll
      for (int d = 0; d < 8; d++) s[t] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(*(const s16v *)(Ks + (t * 16 + c) * KS + d * 16), qf[d], s[t]);
    }
    // Element i of lane (h, c): query c, key start + 16t + 2i + h.
    float local = -__builtin_inff();
    #pragma unroll
    for (int t = 0; t < 2; t++) {
      #pragma unroll
      for (int i = 0; i < 8; i++) {
        int key = start + t * 16 + 2 * i + h;
        float v = s[t][i] * 0.08838834764831845f;
        s[t][i] = key <= qpos ? v : -__builtin_inff();
        local = __builtin_fmaxf(local, s[t][i]);
      }
    }
    float next = __builtin_fmaxf(maxima, __builtin_fmaxf(local, xhalff(local)));
    float scale = next == -__builtin_inff() ? 1.0f : exp2_exact(maxima - next);
    float sum = 0.0f;
    #pragma unroll
    for (int t = 0; t < 2; t++) {
      #pragma unroll
      for (int i = 0; i < 8; i++) {
        s[t][i] = next == -__builtin_inff() ? 0.0f : exp2_exact(s[t][i] - next);
        sum += s[t][i];
      }
    }
    sum += xhalff(sum);
    denom = __builtin_fmaf(denom, scale, sum);
    maxima = next;
    if (scale != 1.0f) {
      #pragma unroll
      for (int d = 0; d < 8; d++) acc[d] *= scale;
    }
    #pragma unroll
    for (int t = 0; t < 2; t++) {
      i4v own, other;
      #pragma unroll
      for (int j = 0; j < 4; j++) {
        own[j] = (int)(unsigned short)tobf(s[t][2 * j]) | ((int)(unsigned short)tobf(s[t][2 * j + 1]) << 16);
        other[j] = xhalf(own[j]);
      }
      i4v lo = h ? other : own, hi = h ? own : other;
      s8v l8 = __builtin_bit_cast(s8v, lo), h8 = __builtin_bit_cast(s8v, hi);
      s16v pb;
      #pragma unroll
      for (int e = 0; e < 8; e++) { pb[e] = l8[e]; pb[8 + e] = h8[e]; }
      #pragma unroll
      for (int d = 0; d < 8; d++) acc[d] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(*(const s16v *)(Vt + (d * 16 + c) * VS + t * 16), pb, acc[d]);
    }
  }
  // Element i of acc[d] at lane (h, c): query c, dimension 16d + 2i + h.
  if (qvalid) {
    u16 *op = out + ((long)head * ROWS + qrow) * W;
    #pragma unroll
    for (int d = 0; d < 8; d++) {
      #pragma unroll
      for (int i = 0; i < 8; i++) op[d * 16 + 2 * i + h] = (u16)tobf(acc[d][i] / denom);
    }
  }
}
