"""Download PhysicalAI-AV chunks (six cameras, egomotion, calibration) into LOCAL_DIR.
Usage (NVIDIA venv): nv_download.py LOCAL_DIR CHUNK [CHUNK ...]"""
import sys, time
import physical_ai_av
FEATURES = ['camera_front_wide_120fov', 'camera_front_tele_30fov', 'camera_cross_left_120fov', 'camera_cross_right_120fov',
            'camera_rear_left_70fov', 'camera_rear_right_70fov', 'egomotion', 'camera_intrinsics', 'sensor_extrinsics', 'vehicle_dimensions']
avdi = physical_ai_av.PhysicalAIAVDatasetInterface(local_dir=sys.argv[1], confirm_download_threshold_gb=1000.)
for chunk in map(int, sys.argv[2:]):
  for f in FEATURES:
    t = time.time()
    for attempt in range(5):
      try:
        avdi.download_chunk_features(chunk, features=[f]); break
      except Exception as e:  # noqa: BLE001
        print('retry', chunk, f, type(e).__name__, str(e)[:100], flush=True); time.sleep(20)
    print('done', chunk, f, round(time.time()-t), 's', flush=True)
print('all done', flush=True)
