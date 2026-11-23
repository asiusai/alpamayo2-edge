"""Recovery views: the car's exact camera projection, seen from a pose shifted dy metres left and turned dpsi radians left.

With dy = dpsi = 0 the output equals selected.camera.nv12_image bit for bit (checked by check_identity). The shift uses a
road-plane depth: rays below the horizon hit flat ground at the camera height, rays above it are treated as far away.
Objects above the road (cars, poles) are therefore only approximately moved; that is the known artifact of this method.
"""
import json
import math
from functools import lru_cache

import numpy as np
from scipy.spatial.transform import Rotation

from openpilot.selfdrive.alpamayo import geometry
from openpilot.selfdrive.alpamayo.selected import camera as cam

CAMERA_HEIGHT = 1.23  # m, Karel's calibrated camera height
FAR = 200.            # m, depth for rays at or above the horizon
HOOD_DEPTH = 4.0      # m: rays reaching the ground closer than this end on the car's own hood, which moves with the car


@lru_cache(maxsize=64)
def perturbed_maps(shape, camera, geometry_json, rpy, wide_rpy, dy, dpsi):
  src = json.loads(geometry_json)['cameras'][str(camera)]
  left, top, right, bottom = src['content_box']
  fx, fy = src['focal_pixels']
  cx, cy = src['principal_point']
  rays = np.array(geometry.lens_rays(camera, geometry_json))  # view frame of the calibrated camera: x right, y down, z forward
  if dy or dpsi:
    c, s = math.cos(dpsi), math.sin(dpsi)
    turn = np.array([[c, 0., -s], [0., 1., 0.], [s, 0., c]])  # the simulated heading, dpsi to the left, in the recorded frame
    depth = np.where(rays[..., 1] > CAMERA_HEIGHT/FAR, CAMERA_HEIGHT/np.maximum(rays[..., 1], 1e-9), FAR)
    points = (rays*depth[..., None]) @ turn.T + np.array([-dy, 0., 0.])  # the simulated camera sits dy metres to the left
    moved = points/np.linalg.norm(points, axis=-1, keepdims=True)
    rays = np.where((depth < HOOD_DEPTH)[..., None], rays, moved)
  # From here on: geometry.coordinates and camera.maps unchanged.
  rotation = Rotation.from_euler('xyz', rpy).as_matrix()
  if camera == 1:
    rotation = Rotation.from_euler('xyz', wide_rpy).as_matrix() @ rotation
  rays = rays @ (geometry.VIEW @ rotation @ geometry.VIEW.T).T
  radial = np.hypot(rays[..., 0], rays[..., 1])
  theta = np.arctan2(radial, rays[..., 2])
  radius = theta if src['projection'] == 'equidistant' else np.tan(theta)
  factor = np.divide(radius, radial, out=np.ones_like(radius), where=radial > 1e-12)
  sx, sy = cx+fx*rays[..., 0]*factor, cy+fy*rays[..., 1]*factor
  source_h, source_w = shape
  sy, sx = (sy-top+.5)*source_h/(bottom-top)-.5, (sx-left+.5)*source_w/(right-left)-.5
  h, w = shape
  sx, sy = np.clip(sx, 0, w-1), np.clip(sy, 0, h-1)
  xy = np.stack([sx, sy], axis=-1)
  offsets = np.floor(xy).astype(np.int32)
  return np.ascontiguousarray(offsets), np.ascontiguousarray(xy-offsets)


def render(nv12, camera, geometry_dict, calibration, dy=0., dpsi=0., *, cache_dir):
  """Planar CHW uint8 image, like the car's projection of this NV12 frame, from the shifted and turned pose."""
  h, w = nv12.shape[0]*2//3, nv12.shape[1]
  offsets, fractions = perturbed_maps((h, w), camera, json.dumps(geometry_dict, sort_keys=True), tuple(calibration['rpyCalib']),
                                      tuple(calibration['wideFromDeviceEuler']) if camera == 1 else (), float(dy), float(dpsi))
  rgb = np.ascontiguousarray(cam.rgb24(nv12))
  out = np.empty((3, *offsets.shape[:2]), dtype=np.uint8)
  cam.library(str(cache_dir)).warp_hwc_u8(rgb.ctypes.data, out.ctypes.data, offsets.ctypes.data, fractions.ctypes.data,
                                         offsets.size//2, w, h)
  return out


def check_identity(nv12, camera, geometry_dict, calibration, cache_dir):
  return np.array_equal(render(nv12, camera, geometry_dict, calibration, cache_dir=cache_dir),
                        cam.nv12_image(nv12, camera, geometry_dict, calibration, cache_dir=cache_dir))
