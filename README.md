# NVIDIA Alpamayo 2 Edge on tinygrad

This repository holds Asius work on NVIDIA Alpamayo 2 Edge: a tinygrad demo, the car runtime on desktop GPUs, fine-tuning, and drive analysis.
NVIDIA's code and weights are not included. Download them from NVIDIA under their licences.

## Layout

| Path | Content | Status |
| --- | --- | --- |
| `run.py`, `model.py`, `models/`, `server.py`, `index.html`, `data/demo*` | First tinygrad port and browser demo on six-camera clips. BF16, stock sampling. | Demo. Not the car recipe. |
| [`pc/`](pc/README.md) | The car's selected runtime on two RX 7900 XTX GPUs (gfx1100, KFD). `pc/openpilot` is a snapshot of the car code. | **Current implementation** of the tinygrad runtime. The snapshot is older than the car code. |
| [`train/`](train/README.md) | LoRA trainer for the action expert (PyTorch-ROCm), sampler and export. | Current. |
| `finetune/` | Dataset builders, evaluators, export helpers and input tests. Documented in [`train/README.md`](train/README.md). | Current. |
| `analysis/` | Drive analysis on openpilot logs, including the drive-test score `lanes_score.py`. | Current. |
| [`kernels/`](kernels/REPORT.md) | GPU kernel review of 2026-11-23: report, per-kernel test loop, candidate kernel sources. | Reference. |

The car runs the model code of the openpilot fork branch `alpamayo` (`openpilot/selfdrive/alpamayo`), now at `d1edf7aa8`.
`pc/openpilot` is the `caa346c94` snapshot with later fine-tune pins. The update to the current car code (1 to 10 Euler steps, new expert layer,
three kernel speedups of 2026-11-23) is not done: the first attempt did not reproduce the plans on gfx1100. `pc/README.md` has the facts.

## Demo

Requires Python 3.12, Git, and FFmpeg.

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
DEV=AMD:LLVM python run.py
```

```text
usage: run.py [-h] [--checkpoint CHECKPOINT] [--demo DEMO] [--output OUTPUT]
              [--every EVERY] [--prepare-only] [--save-reference]
              [--model MODEL] [--limit LIMIT] [--steps {1,2,3,4,5,6,7,8,9,10}]

Run camera predictions, compare with stock references, and browse index.html.

options:
  -h, --help            show this help message and exit
  --checkpoint CHECKPOINT
                        Weights directory; downloads missing weights (default:
                        checkpoints/model)
  --demo DEMO           data/demo1: 6 cameras, 20s (default); demo2: 6
                        cameras, 252s; demo3: 2 cameras, 252s
  --output OUTPUT       Defaults to outputs/<demo>/<model>
  --every EVERY         Seconds between predictions (default: 1); 0.25 selects
                        every available window
  --prepare-only        Check video preparation without inference
  --save-reference      Save/resume stock ten-step predictions in the demo
                        folder
  --model MODEL         Default: model (stock tinygrad). Faster:
                        models.gfx1100. Extend either; select module or
                        module:Class
  --limit LIMIT         Stop after this many advancing predictions
  --steps {1,2,3,4,5,6,7,8,9,10}
                        Euler steps (default: 10); subclasses can use 3 for
                        speed at changed accuracy
```

```sh
python3 server.py
```

Open `http://localhost:8000/` or `http://<server-ip>:8000/` and select a run.
