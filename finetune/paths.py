"""Local paths of the fine-tuning tools.

Defaults describe the PC that built the students. Override a name with ALPAMAYO_<NAME> in the environment, or with
`finetune/paths.local.json` ({"footage": "..."}; not committed). See README.md "Expected layout".
"""
import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent   # finetune/
REPO = HERE.parent                       # the alpamayo2-edge checkout
TRAIN = REPO/'train'                     # the trainer package (alpamayo_train, train.py, sample.py, export.py)
_local = json.loads((HERE/'paths.local.json').read_text()) if (HERE/'paths.local.json').exists() else {}


def path(name, default):
  return Path(os.environ.get('ALPAMAYO_'+name.upper(), _local.get(name, default)))


# Recorded drive segments of the car: <route>--<segment>/{rlog.zst,fcamera.mp4,ecamera.mp4}.
FOOTAGE = path('footage', '/mnt/nvme-256/footage/DEVICE/realdata')
