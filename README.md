# NVIDIA Alpamayo 2 Edge on tinygrad

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
