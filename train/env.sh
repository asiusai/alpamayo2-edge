# Source this file: pinned PyTorch-ROCm training environment for the Alpamayo 2 Edge action expert.
TRAIN_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export TRAIN_ROOT
export PY=$TRAIN_ROOT/.venv/bin/python
# NVIDIA alpamayo2-edge 02f55a7 source (verified by alpamayo_train.model before any GPU use): ../pc/vendor/nvidia.
export PYTHONPATH=$TRAIN_ROOT:$(dirname "$TRAIN_ROOT")/pc/vendor/nvidia/src
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
