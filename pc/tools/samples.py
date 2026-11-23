"""Load model-input samples saved as six NPY files (image_frames, camera_indices, absolute_timestamps,
ego_history_xyz, ego_history_rot, nav_text) or as one NPZ with the same keys."""
from pathlib import Path
import numpy as np

KEYS = ('image_frames', 'camera_indices', 'absolute_timestamps', 'ego_history_xyz', 'ego_history_rot')
TURN = Path('/mnt/nvme-500/tmp/alpamayo-turn-accuracy-20261120')  # deleted by the tmp cleaner on 2026-11-21
KAREL = Path(__file__).resolve().parents[1]/'parity/karel'


def load(path):
  path = Path(path)
  if path.suffix == '.npz':
    with np.load(path, allow_pickle=False) as z:
      sample = {k: z[k] for k in KEYS}
      nav = z['nav_text']
  else:
    sample = {k: np.load(path/f'{k}.npy', allow_pickle=False) for k in KEYS}
    nav = np.load(path/'nav_text.npy', allow_pickle=False)
  sample['nav_text'] = [str(nav.reshape(-1)[0]) if nav.size else '']
  return sample


def turn_samples():
  return sorted(p for group in ('samples-city-turns', 'samples-cruise', 'samples-fast-curves') for p in (TURN/group).glob('q*'))


def karel_samples():
  """48 rebuilt Karel drive inputs (HEVC footage images, exact plan histories), in plan order."""
  return [KAREL.parents[1]/line for line in (KAREL/'samples.txt').read_text().split()]
