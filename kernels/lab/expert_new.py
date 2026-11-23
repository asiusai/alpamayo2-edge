"""Candidate expert_block: the FC2 scale from the signed maximum of the FC1 output (ReLU^2 is monotonic above zero)."""
from tinygrad import Tensor, dtypes
from openpilot.selfdrive.alpamayo.selected.kernels import frozen_ops, tiled_gemm

SPLIT = (16, 256)  # staged maximum: 15360 x 16, then 60 x 256, then 60

def staged_max(x, split=SPLIT):
  # The maximum is order-free. Short per-thread loops; tinygrad's own split gives 1024 threads with 240-element loops.
  value = x.float().reshape(-1, split[0]).max(axis=1).contiguous()
  if split[1]: value = value.reshape(-1, split[1]).max(axis=1).contiguous()
  return value.max()

def make(mode):
  def expert_block(self, x, cos, sin, i, mask):
    assert self.trajectory_active and not self.verifying and x.shape == (1,64,512) and x.dtype == dtypes.bfloat16
    prefix = f'expert.expert.layers.{i}'
    def scale_of(maximum):
      return (maximum/448).clamp(min_=1e-12)
    def fp8(value, scale):
      return (value.float()/scale).clamp(-448,448).cast(dtypes.fp8e4m3)
    def projection(value, scale, name, tile, residual=None):
      return tiled_gemm.gemm(value, self.weights[name+'.weight'], (scale, self.weight_scales[name+'.weight']), tile=tile, residual=residual)
    h = self.norm(x, f'{prefix}.input_layernorm', rms=True).reshape(64,512)
    scale = scale_of(h.float().abs().max())
    qkv = tiled_gemm.gemm(fp8(h, scale), self.expert_qkv[i], (scale, self.expert_qkv_scales[i]), tile=(2,2), column_scale=True)
    q,k,v = (qkv[:,a:b].reshape(1,64,heads,128).transpose(1,2) for a,b,heads in ((0,2048,16),(2048,3072,8),(3072,4096,8)))
    q,k = self.rope(q,k,cos,sin)
    self.trace_counts['expert_attention'] += 1
    attention = frozen_ops.expert_attention(q,k,v,self.cache[i],mask)
    scale = scale_of(frozen_ops.maximum(attention))
    x = projection(fp8(attention.transpose(1,2).reshape(64,2048), scale), scale, f'{prefix}.self_attn.o_proj', (1,2), x.reshape(64,512))
    h = self.norm(x.reshape(1,64,512), f'{prefix}.post_attention_layernorm', rms=True).reshape(64,512)
    scale = scale_of(h.float().abs().max())
    raw = projection(fp8(h, scale), scale, f'{prefix}.mlp.fc1', (2,2))
    hidden = raw.relu().square()
    if mode == 'signed':      # max(relu(x)^2) = relu(max(x))^2 with the same BF16 rounding: both steps are monotonic
      peak = raw.float().max().cast(dtypes.bfloat16).relu().square().float().abs()
    elif mode == 'signed_staged':
      peak = staged_max(raw).cast(dtypes.bfloat16).relu().square().float().abs()
    elif mode == 'staged':
      peak = staged_max(hidden.float().abs())
    else:
      peak = hidden.float().abs().max()
    scale = scale_of(peak)
    return projection(fp8(hidden, scale), scale, f'{prefix}.mlp.fc2', (1,2), x).reshape(1,64,512).realize()
  return expert_block
