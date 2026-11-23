"""Training samples: discovery, validation, CPU preparation and a thread prefetcher (no worker processes).

One sample is ``<stem>.npz`` plus ``<stem>.json`` (meta). The frozen-model outputs (``reasoning_token_ids``,
``baseline_xyz``, ``baseline_rot``) come from ``FROZEN_ROOT/<sample dir relative to DATA_ROOT>/<stem>.frozen.npz``:
the frozen car recipe run with the car's navigation text ``NAV_TEXT`` (``finetune/frozen.py --nav``).
A sample without them is not ready and is skipped. The prompt always uses ``NAV_TEXT``, as the car does.
"""
from __future__ import annotations

import json
import os
import queue
import random
import threading
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

INPUT_KEYS = ('image_frames', 'absolute_timestamps', 'ego_history_xyz', 'ego_history_rot', 'nav_text')
TARGET_KEYS = ('target_xyz', 'target_rot')
FROZEN_KEYS = ('reasoning_token_ids', 'baseline_xyz', 'baseline_rot')
META_KEYS = ('route', 'query_ns', 'dy_m', 'dpsi_rad', 'kind')
CAMERAS = [1] * 5 + [6] * 5
DATA_ROOT = Path(os.environ.get('ALPAMAYO_TRAIN_DATA', '/mnt/nvme-500/alpamayo-train'))
# ALPAMAYO_NAV overrides the text for navigation tests; reasoning must then be regenerated (sample.py --generate).
NAV_TEXT = os.environ.get('ALPAMAYO_NAV', 'Continue straight.')  # the car's default navigation text (NVIDIA's no-maneuver default)
FROZEN_ROOT = DATA_ROOT / 'frozen-cs'  # reasoning generated with NAV_TEXT


@dataclass(frozen=True)
class SampleRef:
  stem: str
  path: Path            # main .npz
  frozen: Path          # side file with the frozen outputs (reasoning from NAV_TEXT)
  meta: dict


def _npz_keys(path: Path) -> set[str]:
  with zipfile.ZipFile(path) as z:
    return {n[:-4] for n in z.namelist() if n.endswith('.npy')}


def discover(directories, limit: int | None = None, kinds: set[str] | None = None) -> tuple[list[SampleRef], dict]:
  """Return ready samples (sorted by stem) and counts of skipped files."""
  refs, skipped = [], {'not_ready': 0, 'no_meta': 0, 'bad': 0, 'kind': 0}
  for directory in directories:
    mirror = FROZEN_ROOT / Path(directory).resolve().relative_to(DATA_ROOT)
    for path in sorted(Path(directory).glob('*.npz')):
      if path.name.endswith('.frozen.npz'):
        continue
      stem = path.name[:-4]
      meta_path = path.with_suffix('.json')
      if not meta_path.exists():
        skipped['no_meta'] += 1
        continue
      try:
        meta = json.loads(meta_path.read_text())
        keys = _npz_keys(path)
      except (OSError, ValueError, zipfile.BadZipFile):
        skipped['bad'] += 1
        continue
      side = mirror / (stem + '.frozen.npz')
      if not side.exists():
        skipped['not_ready'] += 1
        continue
      frozen = side
      if not set(INPUT_KEYS + TARGET_KEYS) <= keys:
        skipped['bad'] += 1
        continue
      if kinds and meta.get('kind') not in kinds:
        skipped['kind'] += 1
        continue
      refs.append(SampleRef(stem, path, frozen, meta))
  refs.sort(key=lambda r: r.stem)
  if limit:
    refs = refs[:limit]
  return refs, skipped


def load_arrays(ref: SampleRef) -> dict:
  with np.load(ref.path, allow_pickle=False) as d:
    arrays = {k: d[k] for k in d.files}
  with np.load(ref.frozen, allow_pickle=False) as d:
    arrays.update({k: d[k] for k in FROZEN_KEYS})
    if str(d['nav_text']) != NAV_TEXT and 'ALPAMAYO_NAV' not in os.environ:
      raise ValueError(f'frozen reasoning used {str(d["nav_text"])!r}, not {NAV_TEXT!r}')
  arrays['nav_text'] = np.array(NAV_TEXT)
  # ALPAMAYO_TARGET=gt: clean samples that carry the recorded human path (PhysicalAI distillation set) train on it
  # instead of the six-camera teacher plan. Recovery samples keep the teacher plan.
  if os.environ.get('ALPAMAYO_TARGET') == 'gt' and 'gt_rot' in arrays and ref.meta.get('kind') == 'clean':
    arrays['target_xyz'], arrays['target_rot'] = arrays['gt_xyz'], arrays['gt_rot']
  return arrays


def _nav_text(value) -> str:
  value = np.asarray(value)
  if value.ndim == 0:
    return str(value.item())
  if value.shape == (1,):
    return str(value[0])
  raise ValueError(f'nav_text must be a string, got shape {value.shape}')


def validate(arrays: dict, anchor: int, traj_ranges: list[tuple[int, int]]) -> None:
  """Raise ValueError naming the first missing or malformed field."""
  def need(key, shape, dtype):
    if key not in arrays:
      raise ValueError(f'missing {key}')
    a = arrays[key]
    if tuple(a.shape) != shape or a.dtype != np.dtype(dtype):
      raise ValueError(f'{key}: expected {shape} {dtype}, got {tuple(a.shape)} {a.dtype}')
    if a.dtype.kind == 'f' and not np.isfinite(a).all():
      raise ValueError(f'{key}: nonfinite values')
  need('image_frames', (10, 1, 3, 432, 768), np.uint8)
  need('absolute_timestamps', (10,), np.int64)
  need('ego_history_xyz', (1, 20, 3), np.float32)
  need('ego_history_rot', (1, 20, 3, 3), np.float32)
  need('target_xyz', (64, 3), np.float32)
  need('target_rot', (64, 3, 3), np.float32)
  need('baseline_xyz', (64, 3), np.float32)
  need('baseline_rot', (64, 3, 3), np.float32)
  if 'camera_indices' in arrays and arrays['camera_indices'].tolist() != CAMERAS:
    raise ValueError(f'camera_indices must be {CAMERAS}')
  _nav_text(arrays['nav_text'])
  ids = arrays['reasoning_token_ids']
  if ids.ndim != 1 or ids.dtype.kind not in 'iu' or ids.size < 1:
    raise ValueError('reasoning_token_ids must be a nonempty 1-D integer array')
  if int(ids[-1]) != anchor or int((ids == anchor).sum()) != 1:
    raise ValueError(f'reasoning_token_ids must end with the trajectory anchor {anchor} exactly once')
  for lo, hi in traj_ranges:
    if ((ids >= lo) & (ids < hi)).any():
      raise ValueError('reasoning_token_ids contain discrete trajectory tokens')
  if np.abs(arrays['ego_history_xyz'][0, -1]).max() > 1e-4:
    raise ValueError('ego_history_xyz must end at the query origin')


class Preparer:
  """CPU input preparation identical to the car's InputAdapter.prepare (NVIDIA processor, same prompt)."""

  def __init__(self, checkpoint: Path):
    import torch
    from alpamayo2_edge.action_space import UnicycleAccelCurvatureCourseActionSpace
    from alpamayo2_edge.config import Alpamayo2EdgeConfig, build_alpamayo2_edge_tokenizer, normalize_checkpoint_config
    from alpamayo2_edge.helper import get_processor
    root = Path(checkpoint)
    self.config = Alpamayo2EdgeConfig(**normalize_checkpoint_config(json.loads((root / 'config.json').read_text())))
    self.config._name_or_path = str(root.resolve())
    self.tokenizer = build_alpamayo2_edge_tokenizer(self.config._name_or_path)
    self.processor = get_processor(self.tokenizer, self.config)
    action_config = dict(self.config.expert_config.action_space_cfg)
    action_config.pop('_target_')
    self.action_space = UnicycleAccelCurvatureCourseActionSpace(**action_config)
    ids = self.config.traj_ids
    self.anchor = ids['future_start']
    self.traj_ranges = [(ids['history_id0'], ids['history_id0'] + self.config.history_vocab_size),
                        (ids['future_id0'], ids['future_id0'] + self.config.future_vocab_size)]
    self.torch = torch

  def prompt(self, arrays: dict) -> dict:
    """Tokenized prompt (car layout) for the six model input fields."""
    torch = self.torch
    from alpamayo2_edge.chat_template import build_conversation
    from alpamayo2_edge.helper import _tokenize_messages
    from alpamayo2_edge.models.delta_tokenizer import DeltaTrajectoryTokenizer
    from alpamayo2_edge.sample import CAMERA_PIXEL_BOUNDS
    data = {'image_frames': torch.from_numpy(np.ascontiguousarray(arrays['image_frames'])),
            'camera_indices': torch.tensor(CAMERAS, dtype=torch.int64),
            'absolute_timestamps': torch.from_numpy(np.ascontiguousarray(arrays['absolute_timestamps'])),
            'ego_history_xyz': torch.from_numpy(np.ascontiguousarray(arrays['ego_history_xyz'])),
            'ego_history_rot': torch.from_numpy(np.ascontiguousarray(arrays['ego_history_rot'])),
            'nav_text': [_nav_text(arrays['nav_text'])], 'camera_pixel_bounds': CAMERA_PIXEL_BOUNDS}
    tokenized = _tokenize_messages(build_conversation(data), self.processor, include_closed_thinking=True)
    ids = tokenized['input_ids'].clone()
    xyz, rot = data['ego_history_xyz'], data['ego_history_rot']
    history = DeltaTrajectoryTokenizer(pad_origin_at_beginning=False).encode(None, None, xyz, rot)
    pad = ids == self.config.traj_ids['history_pad']
    if int(pad.sum()) != history.numel():
      raise ValueError('History placeholder count mismatch')
    ids[pad] = (history + self.config.traj_ids['history_id0']).flatten().to(ids.dtype)
    return {'input_ids': ids, 'mm_token_type_ids': tokenized['mm_token_type_ids'],
            'pixel_values': tokenized['pixel_values'], 'image_grid_thw': tokenized['image_grid_thw'],
            'history_xyz': xyz, 'history_rot': rot}

  def __call__(self, ref: SampleRef) -> dict:
    torch = self.torch
    arrays = load_arrays(ref)
    validate(arrays, self.anchor, self.traj_ranges)
    out = self.prompt(arrays)
    out['reasoning_ids'] = torch.from_numpy(arrays['reasoning_token_ids'].astype(np.int64))
    hx, hr = out['history_xyz'], out['history_rot']
    target_xyz = torch.from_numpy(arrays['target_xyz'])[None]
    target_rot = torch.from_numpy(arrays['target_rot'])[None]
    with torch.no_grad():
      t0 = self.action_space.estimate_t0_states(hx, hr)
      action = self.action_space.traj_to_action(hx, hr, target_xyz, target_rot, t0_states=t0)
      back, _ = self.action_space.action_to_traj(action, hx, hr, t0_states=t0)
    out.update(stem=ref.stem, meta=ref.meta, target_action=action.float(), t0_states=t0,
               target_xyz=arrays['target_xyz'], baseline_xyz=arrays['baseline_xyz'],
               roundtrip_max_m=float((back[0, :, :2] - target_xyz[0, :, :2]).norm(dim=-1).max()),
               action_abs_max=float(action.abs().max()))
    return out

  def decode(self, action, prepared: dict):
    """Normalized CPU actions (1,64,2) -> xyz (1,64,3), rot (1,64,3,3) on the CPU, as the car does."""
    if action.device.type != 'cpu':
      raise ValueError('decode() takes CPU actions; copy them with PinnedStager.d2h')
    with self.torch.no_grad():
      return self.action_space.action_to_traj(action.float(), prepared['history_xyz'], prepared['history_rot'],
                                              t0_states=prepared['t0_states'])


class Prefetcher:
  """Prepare samples on one background thread. Threads never fork, so this is safe with the GPU open."""

  def __init__(self, preparer: Preparer, refs: list[SampleRef], *, epochs: int | None = 1, shuffle: bool = False,
               seed: int = 0, depth: int = 4):
    self.preparer, self.refs, self.epochs, self.shuffle, self.seed = preparer, refs, epochs, shuffle, seed
    self.q: queue.Queue = queue.Queue(maxsize=depth)
    self.stop = threading.Event()
    self.thread = threading.Thread(target=self._run, daemon=True)
    self.thread.start()

  def _run(self):
    rng = random.Random(self.seed)
    epoch = 0
    try:
      while self.epochs is None or epoch < self.epochs:
        order = list(range(len(self.refs)))
        if self.shuffle:
          rng.shuffle(order)
        for i in order:
          if self.stop.is_set():
            return
          try:
            item = self.preparer(self.refs[i])
          except ValueError as e:
            item = {'error': f'{self.refs[i].stem}: {e}'}
          item['epoch'] = epoch
          self.q.put(item)
        epoch += 1
      self.q.put(None)
    except BaseException as e:  # surface thread failures in the consumer
      self.q.put({'fatal': repr(e)})

  def __iter__(self):
    while True:
      item = self.q.get()
      if item is None:
        return
      if 'fatal' in item:
        raise RuntimeError(f'Data thread failed: {item["fatal"]}')
      yield item

  def close(self):
    self.stop.set()
