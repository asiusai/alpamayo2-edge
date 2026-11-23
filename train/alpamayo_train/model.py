"""Frozen Alpamayo 2 Edge prefix, trainable action expert with LoRA, flow-matching loss and car-style sampling."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import torch
import torch.nn as nn

REPO = Path(__file__).resolve().parents[2]  # the alpamayo2-edge checkout; pc/ holds the NVIDIA source and the checkpoint
NVIDIA_SRC = REPO / 'pc/vendor/nvidia/src/alpamayo2_edge'
NVIDIA_REVISION = '02f55a7aeabc1a2b9f7a80445c08e40bddc89f5e'
ORIGINAL_CHECKPOINT = REPO / 'pc/data/original-checkpoint'
# Expert linears that receive LoRA. Everything else (VLM, norms, action projections) stays frozen.
LORA_TARGETS = ('self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj', 'self_attn.o_proj', 'mlp.fc1', 'mlp.fc2')


def source_digest() -> str:
  """Digest of the NVIDIA Python sources used here (no git subprocess, safe with the GPU open)."""
  h = hashlib.sha256()
  for path in sorted(NVIDIA_SRC.rglob('*.py')):
    h.update(str(path.relative_to(NVIDIA_SRC)).encode())
    h.update(path.read_bytes())
  return h.hexdigest()


def verify_sources() -> None:
  """Check the NVIDIA checkout revision. Uses a git subprocess, so call it before the GPU opens."""
  import subprocess
  root = NVIDIA_SRC.parent.parent
  head = subprocess.run(['git', '-C', str(root), 'rev-parse', 'HEAD'], capture_output=True, text=True, check=True).stdout.strip()
  dirty = subprocess.run(['git', '-C', str(root), 'status', '--porcelain', '--untracked-files=no'],
                         capture_output=True, text=True, check=True).stdout.strip()
  if head != NVIDIA_REVISION or dirty:
    raise RuntimeError(f'NVIDIA source must be clean {NVIDIA_REVISION}, got {head} dirty={bool(dirty)}')


def load_car_noise() -> torch.Tensor:
  """The car's initial noise: zeros (/data/alpamayo-live-v2/fixed/initial_noise-zero.npy on the car, since 2026-11-22)."""
  return torch.zeros(1, 64, 2)


# VLM adapter targets: language decoder layers (28) and SigLIP2 vision encoder layers (27).
VLM_LANGUAGE_TARGETS = ('self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj', 'self_attn.o_proj', 'mlp.fc1', 'mlp.fc2')
VLM_VISION_TARGETS = ('self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj', 'self_attn.out_proj', 'mlp.fc1', 'mlp.fc2')


class LoRALinear(nn.Module):
  """y = base(x) + (alpha / r) * B(A(dropout(x))); A Kaiming-uniform, B zero, so training starts at the base model."""

  def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float = 0.0):
    super().__init__()
    self.base = base
    self.rank, self.alpha, self.scaling = rank, alpha, alpha / rank
    self.lora_A = nn.Parameter(torch.empty(rank, base.in_features, dtype=torch.float32, device=base.weight.device))
    self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank, dtype=torch.float32, device=base.weight.device))
    nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
    self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
    self.enabled = True

  @property
  def weight(self):
    return self.base.weight

  def forward(self, x):
    y = self.base(x)
    if not self.enabled:
      return y
    # Under autocast the two small matmuls run in BF16; the adapter parameters stay FP32.
    return y + (self.dropout(x) @ self.lora_A.t() @ self.lora_B.t()) * self.scaling

  def delta(self) -> torch.Tensor:
    """Merged weight update (out, in) in float32."""
    return (self.lora_B.float() @ self.lora_A.float()) * self.scaling


class PinnedStager:
  """Every host<->GPU copy goes through pinned host buffers that are allocated once and never freed.

  A pageable copy makes HIP lock (userptr-register) the pageable pages. When Python later frees or unmaps that
  memory (tensors, the safetensors mmap), the MMU notifier makes KFD evict and restore every queue of this process.
  A GPU access racing with that invalidation faults on the host page (seen on GPU 1 at 19:05, address 0x7b23ce45a000,
  followed by MES failures and a mode1 reset). hipHostMalloc memory is a kernel-owned GTT allocation, not userptr.
  """

  def __init__(self, nbytes: int = 1 << 28):
    self.buffers = [torch.empty(nbytes, dtype=torch.uint8, pin_memory=True)]

  def _view(self, nbytes: int) -> torch.Tensor:
    if nbytes > self.buffers[-1].numel():
      self.buffers.append(torch.empty(nbytes, dtype=torch.uint8, pin_memory=True))  # keep older buffers alive
    return self.buffers[-1][:nbytes]

  def h2d(self, t: torch.Tensor, device: torch.device, chunk: int = 1 << 28) -> torch.Tensor:
    t = t.detach().contiguous()
    out = torch.empty(t.shape, dtype=t.dtype, device=device)
    src, dst = t.reshape(-1).view(torch.uint8), out.reshape(-1).view(torch.uint8)
    for start in range(0, src.numel(), chunk):
      n = min(chunk, src.numel() - start)
      staging = self._view(n)
      staging.copy_(src[start:start + n])
      dst[start:start + n].copy_(staging, non_blocking=True)
      torch.cuda.current_stream(device).synchronize()
    return out

  def d2h(self, t: torch.Tensor) -> torch.Tensor:
    t = t.detach().contiguous()
    nbytes = t.numel() * t.element_size()
    staging = self._view(nbytes)
    staging.copy_(t.reshape(-1).view(torch.uint8), non_blocking=True)
    torch.cuda.current_stream(t.device).synchronize()
    return staging.clone().view(t.dtype).reshape(t.shape)

  def module_to(self, module: nn.Module, device: torch.device) -> nn.Module:
    moved: dict[int, torch.Tensor] = {}
    for m in module.modules():
      for name, p in list(m._parameters.items()):
        if p is not None:
          if id(p) not in moved:
            moved[id(p)] = nn.Parameter(self.h2d(p.data, device), requires_grad=False)
          m._parameters[name] = moved[id(p)]
      for name, b in list(m._buffers.items()):
        if b is not None:
          if id(b) not in moved:
            moved[id(b)] = self.h2d(b, device)
          m._buffers[name] = moved[id(b)]
    return module


STAGER: PinnedStager | None = None


def stager() -> PinnedStager:
  global STAGER
  if STAGER is None:
    STAGER = PinnedStager()
  return STAGER


class PrefixKV:
  """Read-only HF cache: each expert layer attends to the frozen VLM rows plus its own 64 action rows.

  update() never stores, so the expert can run many times (and recompute under checkpointing) on one prefix.
  """

  def __init__(self, layers: list[tuple[torch.Tensor, torch.Tensor]], batch: int):
    self.layers_kv, self.batch = layers, batch

  def update(self, key_states, value_states, layer_idx, *args, **kwargs):
    k, v = self.layers_kv[layer_idx]
    b = self.batch
    return (torch.cat([k.expand(b, -1, -1, -1), key_states], dim=2),
            torch.cat([v.expand(b, -1, -1, -1), value_states], dim=2))

  def get_seq_length(self, layer_idx: int = 0) -> int:
    return self.layers_kv[0][0].shape[2]


class ExpertTrainer:
  """Owns the frozen model on one GPU, the LoRA adapters, the prefix forward and the expert forward."""

  def __init__(self, checkpoint: Path, device: torch.device, *, rank: int = 0, alpha: float | None = None,
               dropout: float = 0.0, targets=LORA_TARGETS, grad_checkpointing: bool = False):
    from alpamayo2_edge.checkpoint import load_model
    self.device = device
    # Load on the CPU, then copy every tensor through pinned staging (no pageable or mmap-backed GPU copies).
    self.model, _ = load_model(str(checkpoint), device='cpu', dtype='bfloat16', attn_implementation='sdpa',
                               local_files_only=True)
    self.stager = stager()
    self.stager.module_to(self.model, device)
    self.model.requires_grad_(False)
    self.model.eval()
    self.expert = self.model.expert
    self.layers = self.expert.expert.layers
    self.grad_checkpointing = grad_checkpointing
    self.lora: dict[str, LoRALinear] = {}
    self.lora_config = None
    if rank > 0:
      self.add_lora(rank, alpha if alpha is not None else 2.0 * rank, dropout, targets)
    self.n_waypoints, self.action_dim = self.expert.action_space.get_action_space_dims()
    self.car_noise = self.stager.h2d(load_car_noise(), device)

  # ---- LoRA -------------------------------------------------------------------------------------------------
  def add_lora(self, rank: int, alpha: float, dropout: float, targets=LORA_TARGETS):
    self.lora_config = {**(self.lora_config or {}), 'rank': rank, 'alpha': alpha, 'dropout': dropout, 'targets': list(targets)}
    for i, layer in enumerate(self.layers):
      for target in targets:
        parent_name, child = target.rsplit('.', 1)
        parent = layer.get_submodule(parent_name)
        base = getattr(parent, child)
        if isinstance(base, LoRALinear):
          raise RuntimeError('LoRA already applied')
        wrapped = LoRALinear(base, rank, alpha, dropout)
        setattr(parent, child, wrapped)
        self.lora[f'expert.expert.layers.{i}.{target}'] = wrapped

  def add_vlm_lora(self, rank: int, alpha: float, scope: str = 'language', dropout: float = 0.0):
    """LoRA on the VLM: 'language' (decoder layers) or 'both' (plus the vision encoder). Reasoning then changes too."""
    if scope not in ('language', 'both'):
      raise ValueError(f'Unknown VLM LoRA scope {scope}')
    self.lora_config = {**(self.lora_config or {}), 'vlm': {'rank': rank, 'alpha': alpha, 'scope': scope}}
    groups = [('vlm.model.language_model.layers', VLM_LANGUAGE_TARGETS)]
    if scope == 'both':
      groups.append(('vlm.model.visual.encoder.layers', VLM_VISION_TARGETS))
    for root, targets in groups:
      for i, layer in enumerate(self.model.get_submodule(root)):
        for target in targets:
          parent_name, child = target.rsplit('.', 1)
          parent = layer.get_submodule(parent_name)
          base = getattr(parent, child)
          if isinstance(base, LoRALinear):
            raise RuntimeError('VLM LoRA already applied')
          wrapped = LoRALinear(base, rank, alpha, dropout)
          setattr(parent, child, wrapped)
          self.lora[f'{root}.{i}.{target}'] = wrapped

  def parameter_groups(self) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
    """(expert adapter parameters, VLM adapter parameters)."""
    expert, vlm = [], []
    for name, m in self.lora.items():
      (vlm if name.startswith('vlm.') else expert).extend([m.lora_A, m.lora_B])
    return expert, vlm

  def trainable_parameters(self) -> list[nn.Parameter]:
    params = []
    for m in self.lora.values():
      params += [m.lora_A, m.lora_B]
    return params

  def set_lora_enabled(self, enabled: bool):
    for m in self.lora.values():
      m.enabled = enabled

  def lora_state_dict(self) -> dict[str, torch.Tensor]:
    out = {}
    for name, m in self.lora.items():
      out[f'{name}.lora_A'] = self.stager.d2h(m.lora_A.float())
      out[f'{name}.lora_B'] = self.stager.d2h(m.lora_B.float())
    return out

  def save_lora(self, path: Path, extra: dict | None = None):
    from safetensors.torch import save_file
    path.mkdir(parents=True, exist_ok=True)
    tmp = path / 'adapter.safetensors.tmp'
    save_file(self.lora_state_dict(), str(tmp))
    tmp.replace(path / 'adapter.safetensors')
    (path / 'adapter.json').write_text(json.dumps({**self.lora_config, **(extra or {})}, indent=2) + '\n')

  @staticmethod
  def read_lora(path: Path) -> tuple[dict, dict[str, torch.Tensor]]:
    from safetensors.torch import load_file
    return json.loads((path / 'adapter.json').read_text()), load_file(str(path / 'adapter.safetensors'))

  def load_lora(self, path: Path):
    config, tensors = self.read_lora(path)
    if not self.lora:
      self.add_lora(config['rank'], config['alpha'], 0.0, tuple(config['targets']))
    if config.get('vlm') and not any(name.startswith('vlm.') for name in self.lora):
      self.add_vlm_lora(config['vlm']['rank'], config['vlm']['alpha'], config['vlm']['scope'])
    for name, m in self.lora.items():
      m.lora_A.data.copy_(self.stager.h2d(tensors[f'{name}.lora_A'], self.device))
      m.lora_B.data.copy_(self.stager.h2d(tensors[f'{name}.lora_B'], self.device))
    return config

  def prompt_tensors(self, prepared: dict):
    """Prompt + teacher-forced reasoning ids, token types, pixels and grid on the GPU (pinned staging)."""
    reasoning = prepared['reasoning_ids'][None]
    ids = torch.cat([prepared['input_ids'], reasoning], dim=1)
    mm = torch.cat([prepared['mm_token_type_ids'], torch.zeros_like(reasoning, dtype=prepared['mm_token_type_ids'].dtype)], dim=1)
    h2d = self.stager.h2d
    return (h2d(ids, self.device), h2d(mm, self.device), h2d(prepared['pixel_values'], self.device),
            h2d(prepared['image_grid_thw'], self.device))

  # ---- frozen prefix ----------------------------------------------------------------------------------------
  @torch.no_grad()
  def prefix(self, prepared: dict) -> dict:
    """Frozen VLM over prompt + teacher-forced reasoning (ending with the anchor). Returns per-layer K/V."""
    ids, mm, pixels, grid = self.prompt_tensors(prepared)
    vlm = self.model.vlm.model
    vlm.rope_deltas = None
    with torch.autocast('cuda', dtype=torch.bfloat16):
      out = vlm(input_ids=ids, attention_mask=torch.ones_like(ids), pixel_values=pixels, image_grid_thw=grid,
                mm_token_type_ids=mm, use_cache=True)
    cache = out.past_key_values
    kv = [(layer.keys, layer.values) for layer in cache.layers]
    rope_delta = int(vlm.rope_deltas.reshape(-1)[0].item())
    vlm.rope_deltas = None
    length = ids.shape[1]
    if kv[0][0].shape[2] != length:
      raise RuntimeError('Prefix cache length mismatch')
    return {'kv': kv, 'length': length, 'rope_delta': rope_delta}

  def prefix_train(self, prepared: dict, ce: bool = True) -> tuple[dict, torch.Tensor | None]:
    """Prefix with gradients into the VLM adapters, and the mean cross-entropy of the teacher-forced reasoning ids
    (including the closing trajectory anchor). Used only when VLM adapters train."""
    ids, mm, pixels, grid = self.prompt_tensors(prepared)
    reasoning = ids.shape[1] - prepared['input_ids'].shape[1]
    vlm = self.model.vlm.model
    vlm.rope_deltas = None
    with torch.autocast('cuda', dtype=torch.bfloat16):
      out = vlm(input_ids=ids, attention_mask=torch.ones_like(ids), pixel_values=pixels, image_grid_thw=grid,
                mm_token_type_ids=mm, use_cache=True)
      loss_ce = None
      if ce and reasoning > 0:
        hidden = out.last_hidden_state[:, -reasoning - 1:-1]
        logits = self.model.get_output_embeddings()(hidden).float()
        loss_ce = torch.nn.functional.cross_entropy(logits[0], ids[0, -reasoning:])
    kv = [(layer.keys, layer.values) for layer in out.past_key_values.layers]
    rope_delta = int(vlm.rope_deltas.reshape(-1)[0].item())
    vlm.rope_deltas = None
    return {'kv': kv, 'length': ids.shape[1], 'rope_delta': rope_delta}, loss_ce

  @torch.no_grad()
  def generate_reasoning(self, prepared: dict, max_new_tokens: int = 256) -> torch.Tensor:
    """Greedy reasoning from the (adapted) VLM, ending with the trajectory anchor, like the car's decoder (CPU int64)."""
    import copy
    from transformers import StoppingCriteriaList
    from transformers.generation.logits_process import LogitsProcessorList
    from alpamayo2_edge.models.alpamayo2_edge import MaskDiscreteTrajectoryLogitsProcessor, MaskTokenIdsLogitsProcessor
    from alpamayo2_edge.models.expert_utils import StopAfterEOS
    m = self.model
    h2d = self.stager.h2d
    ids = h2d(prepared['input_ids'], self.device)
    data = {'input_ids': ids, 'attention_mask': torch.ones_like(ids), 'pixel_values': h2d(prepared['pixel_values'], self.device),
            'image_grid_thw': h2d(prepared['image_grid_thw'], self.device),
            'mm_token_type_ids': h2d(prepared['mm_token_type_ids'], self.device)}
    anchor = m.config.traj_ids['future_start']
    gc = copy.deepcopy(m.vlm.generation_config)
    gc.do_sample, gc.num_beams, gc.top_p, gc.top_k, gc.temperature = False, 1, None, None, None
    gc.max_new_tokens, gc.output_logits, gc.output_scores = max_new_tokens, False, False
    gc.return_dict_in_generate, gc.use_cache, gc.pad_token_id = True, True, m.tokenizer.pad_token_id
    eos = gc.eos_token_id
    eos = [eos] if isinstance(eos, int) else list(eos or [])
    gc.eos_token_id = None
    processors = LogitsProcessorList([
        MaskDiscreteTrajectoryLogitsProcessor(m.config.traj_ids['history_id0'], m.config.history_vocab_size),
        MaskDiscreteTrajectoryLogitsProcessor(m.config.traj_ids['future_id0'], m.config.future_vocab_size)])
    blocked = [t for t in eos if t != anchor]
    if blocked:
      processors.append(MaskTokenIdsLogitsProcessor(blocked))
    m.vlm.model.rope_deltas = None
    with torch.autocast('cuda', dtype=torch.bfloat16):
      out = m._generate_with_shared_prefill(data, generation_config=gc, n_samples_total=1, logits_processor=processors,
                                            stopping_criteria=StoppingCriteriaList([StopAfterEOS(anchor)]))
    m.vlm.model.rope_deltas = None
    new = out.sequences[0, ids.shape[1]:].cpu()
    hits = (new == anchor).nonzero()
    if len(hits) == 0:
      raise ValueError('Generated reasoning has no trajectory anchor')
    return new[:int(hits[0]) + 1].to(torch.int64)

  # ---- expert -----------------------------------------------------------------------------------------------
  def velocity(self, x: torch.Tensor, t: torch.Tensor, prefix: dict) -> torch.Tensor:
    """Flow velocity for actions x (B,64,2) at times t (B,) given one sample's frozen prefix."""
    b = x.shape[0]
    expert = self.expert
    with torch.autocast('cuda', dtype=torch.bfloat16):
      h = expert.action_in_proj(x, t.reshape(b, 1, 1)).reshape(b, self.n_waypoints, -1)
      pos = torch.arange(self.n_waypoints, device=x.device) + prefix['length'] + prefix['rope_delta']
      pos = pos.view(1, 1, -1).expand(3, b, -1)
      pe = expert.expert.rotary_emb(h, pos)
      cache = PrefixKV(prefix['kv'], b)
      for layer in self.layers:
        if self.grad_checkpointing and torch.is_grad_enabled():
          h = torch.utils.checkpoint.checkpoint(self._layer, layer, h, pe, cache, use_reentrant=False)
        else:
          h = self._layer(layer, h, pe, cache)
      h = expert.expert.norm(h)
      v = expert.action_out_proj(h)
    return v.float().reshape(b, self.n_waypoints, self.action_dim)

  @staticmethod
  def _layer(layer, h, pe, cache):
    return layer(h, attention_mask=None, position_embeddings=pe, past_key_values=cache, is_causal=False)

  def loss(self, prefix: dict, target_action: torch.Tensor, n_noise: int, car_step_weight: float = 0.0,
           generator: torch.Generator | None = None, car_steps: int = 1, car_grad_steps: int | None = None,
           lat_weight: float = 0.0, geom: dict | None = None) -> tuple[torch.Tensor, dict]:
    """Flow-matching loss (NVIDIA FlowMatching.construct_training_data / compute_loss_from_pred) with n_noise draws.

    Optional car_step_weight adds the car's exact car_steps-step Euler result from the car noise (zeros), MSE to the
    target action. car_grad_steps < car_steps runs the first steps without gradient (truncated backpropagation).
    """
    car_grad_steps = car_steps if car_grad_steps is None else min(car_grad_steps, car_steps)
    diffusion = self.expert.diffusion
    target = self.stager.h2d(target_action.float(), self.device)
    x = target.expand(n_noise, -1, -1).contiguous()
    fm = diffusion.construct_training_data(x)  # t ~ 0.999*(1-Beta(1.5,1)), noise ~ N(0, I)
    xs, ts = [fm['noisy_x']], [fm['timesteps'].reshape(-1)]
    batched_car = car_step_weight > 0 and car_grad_steps == car_steps
    if batched_car:
      xs.append(self.car_noise)
      ts.append(torch.zeros(1, device=self.device))
    pred = self.velocity(torch.cat(xs), torch.cat(ts), prefix)
    loss_fm = diffusion.compute_loss_from_pred(fm, pred[:n_noise])
    stats = {'loss_fm': float(loss_fm.detach())}
    loss = loss_fm
    if car_step_weight > 0:
      # The car's Euler result from the car noise: with full gradient, step 1 shares the batched pass above.
      if batched_car:
        x1, first = self.car_noise + pred[n_noise:] / car_steps, 1
      else:
        x1, first = self.car_noise, 0
        with torch.no_grad():
          for i in range(car_steps - car_grad_steps):
            x1 = x1 + self.velocity(x1, torch.full((1,), i / car_steps, device=self.device), prefix) / car_steps
          first = car_steps - car_grad_steps
        x1 = x1.detach()
      for i in range(first, car_steps):
        x1 = x1 + self.velocity(x1, torch.full((1,), i / car_steps, device=self.device), prefix) / car_steps
      loss_car = torch.nn.functional.mse_loss(x1, target)
      loss = loss + car_step_weight * loss_car
      stats['loss_car'] = float(loss_car.detach())
      if lat_weight > 0:
        # Cross-track error of the car result's path (m), with the target's speed profile so only steering counts.
        # The action MSE barely sees a constant curvature error that moves the path 0.2 m at 40 m.
        txy, tth = self.path(target, geom)
        pxy, _ = self.path(torch.stack([target[..., 0], x1[..., 1]], -1), geom)
        d = pxy - txy
        cross = -torch.sin(tth) * d[..., 0] + torch.cos(tth) * d[..., 1]
        loss_lat = torch.nn.functional.smooth_l1_loss(cross, torch.zeros_like(cross), beta=0.3)
        loss = loss + lat_weight * loss_lat
        stats['loss_lat'] = float(loss_lat.detach())
        stats['cross_3s_m'] = float(cross[..., 29].detach().abs().mean())
    return loss, stats

  def path(self, action: torch.Tensor, geom: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """Differentiable unicycle integration of normalized actions (B,64,2), as NVIDIA's action_to_traj: xy (B,64,2) and
    course (B,64) in the query frame. geom: v0, theta0 (floats), dt and the action normalization tensors."""
    accel = action[..., 0] * geom['accel_std'] + geom['accel_mean']
    kappa = action[..., 1] * geom['curvature_std'] + geom['curvature_mean']
    dt = geom['dt']
    v0 = torch.full_like(accel[..., :1], geom['v0'])
    th0 = torch.full_like(accel[..., :1], geom['theta0'])
    v = torch.cat([v0, v0 + torch.cumsum(accel * dt, -1)], -1)
    theta = torch.cat([th0, th0 + torch.cumsum(kappa * v[..., :-1] * dt, -1) + torch.cumsum(kappa * accel * (0.5 * dt * dt), -1)], -1)
    x = torch.cumsum((v[..., :-1] * torch.cos(theta[..., :-1]) + v[..., 1:] * torch.cos(theta[..., 1:])) * (0.5 * dt), -1)
    y = torch.cumsum((v[..., :-1] * torch.sin(theta[..., :-1]) + v[..., 1:] * torch.sin(theta[..., 1:])) * (0.5 * dt), -1)
    return torch.stack([x, y], -1), theta[..., 1:]

  @torch.no_grad()
  def sample_car(self, prefix: dict, steps: int = 1, noise: torch.Tensor | None = None) -> torch.Tensor:
    """The car's sampler: Euler from t=0 with the car noise (zeros); 10 steps on the car. Returns actions (1,64,2) fp32."""
    x = (self.car_noise if noise is None else self.stager.h2d(noise, self.device)).float()
    for i in range(steps):
      t = torch.full((x.shape[0],), i / steps, device=self.device)
      x = x + self.velocity(x, t, prefix) / steps
    return x

  @torch.no_grad()
  def reference_velocity(self, x: torch.Tensor, t: torch.Tensor, prepared: dict) -> torch.Tensor:
    """Upstream HF path (DynamicCache + expert.expert forward with the 4-D mask), as in Alpamayo2Edge._sample_actions."""
    from alpamayo2_edge.models.expert_utils import build_expert_pos_ids_and_attn_mask
    dev = self.device
    ids, mm, pixels, grid = self.prompt_tensors(prepared)
    vlm = self.model.vlm.model
    vlm.rope_deltas = None
    with torch.autocast('cuda', dtype=torch.bfloat16):
      out = vlm(input_ids=ids, attention_mask=torch.ones_like(ids), pixel_values=pixels, image_grid_thw=grid,
                mm_token_type_ids=mm, use_cache=True)
      cache, length = out.past_key_values, ids.shape[1]
      offsets = torch.tensor([length], device=dev)
      position_ids, mask = build_expert_pos_ids_and_attn_mask(offset=offsets, rope_deltas=vlm.rope_deltas, kv_cache_seq_len=length,
                                                              n_diffusion_tokens=64, b_star=1, device=dev,
                                                              prefix_mask=torch.ones_like(ids))
      emb = self.expert.action_in_proj(x, t.reshape(1, 1, 1)).reshape(1, 64, -1)
      o = self.expert.expert(inputs_embeds=emb, position_ids=position_ids, past_key_values=cache, attention_mask=mask,
                             use_cache=True, is_causal=False)
      v = self.expert.action_out_proj(o.last_hidden_state).reshape(-1, 64, 2)
    vlm.rope_deltas = None
    return v.float()

  def fp8_weight_roundtrip(self):
    """Round the expert layer linears (merged with LoRA) to the car's per-tensor FP8 E4M3 weights, in place, for eval.

    This simulates prepare-fp8.py weight quantization only (not the car's FP8 activation rounding). Irreversible.
    """
    self.merge_lora_inplace()
    for layer in self.layers:
      for target in LORA_TARGETS:
        lin = layer.get_submodule(target)
        w = lin.weight.data.float()
        scale = (w.abs().max() / 448).clamp(min=1e-12)
        q = (w / scale).clamp(-448, 448).to(torch.float8_e4m3fn).float() * scale
        lin.weight.data.copy_(q.to(lin.weight.dtype))

  @torch.no_grad()
  def merge_lora_inplace(self):
    """Fold LoRA into the base BF16 weights and unwrap (merged weight rounded to BF16 like the export)."""
    for name, m in list(self.lora.items()):
      w = m.base.weight
      w.data.copy_((w.data.float() + m.delta()).to(w.dtype))
      parent_name, child = name.rsplit('.', 1)
      setattr(self.model.get_submodule(parent_name), child, m.base)
    self.lora.clear()
