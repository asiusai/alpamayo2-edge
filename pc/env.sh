# Source this file: pinned interpreter and sources for the PC Alpamayo runtime.
PC_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export PC_ROOT
export PY=$PC_ROOT/vendor/nvidia/.venv/bin/python   # NVIDIA 02f55a7 checkout and venv (torch 2.8 CPU, av, scipy, pycapnp)
export PYTHONPATH=$PC_ROOT:$PC_ROOT/vendor/tinygrad:$PC_ROOT/vendor/opendbc
export PYTHONDONTWRITEBYTECODE=1
