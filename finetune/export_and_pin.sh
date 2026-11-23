#!/bin/bash
# Export a LoRA run to the car's FP8 cache and pin it as a reviewed fine-tune in the runtime (openpilot `alpamayo` commit + push).
# Usage: export_and_pin.sh RUN_NAME VIEWS   e.g. export_and_pin.sh t120-g3 120/30
set -euo pipefail
NAME=$1; VIEWS=$2
A=/mnt/nvme-500/alpamayo-train; E=$A/exports
F=$(cd "$(dirname "$0")" && pwd); REPO=$(dirname "$F")   # finetune/ and the alpamayo2-edge checkout
W=${OPENPILOT:-/home/john/alpamayo-work/alpamayo-w4-integration-20261118}   # openpilot fork worktree on branch alpamayo
[ -f $A/runs/$NAME/final/adapter.json ] || { echo "no final adapter for $NAME"; exit 1; }
cd $REPO/train && source env.sh
[ -d $E/$NAME ] || nice $PY export.py $A/runs/$NAME/final $E/$NAME | grep -E "fp8_retained"
cd $F && source $REPO/pc/env.sh
[ -f $E/$NAME-fp8/weights.bin ] || nice $PY prepare-fp8.py $E/$NAME $E/$NAME-fp8 | tail -1
git -C $W pull -q --ff-only origin alpamayo
python3 - "$NAME" "$VIEWS" "$W" "$REPO" <<'PY'
import json, hashlib, sys, re
from pathlib import Path
import numpy as np
name, views = sys.argv[1], sys.argv[2]
E = Path('/mnt/nvme-500/alpamayo-train/exports'); base = Path(sys.argv[-1])/'pc/assets/fp8-cache'
ia, ib = json.loads((base/'index.json').read_text()), json.loads((E/f'{name}-fp8/index.json').read_text())
key = 'records' if 'records' in ia else 'tensors'
ra = {r['name']+'|'+r['kind']: r for r in ia[key]}; rb = {r['name']+'|'+r['kind']: r for r in ib[key]}
ma, mb = np.memmap(base/'weights.bin', np.uint8, 'r'), np.memmap(E/f'{name}-fp8/weights.bin', np.uint8, 'r')
for k, r in ra.items():
  s = rb[k]
  assert (r['offset'], r['nbytes']) == (s['offset'], s['nbytes']), k
  if not np.array_equal(ma[r['offset']:r['offset']+r['nbytes']], mb[s['offset']:s['offset']+s['nbytes']]):
    assert re.match(r'expert\.expert\.layers\.\d+\.(self_attn\.[qkvo]_proj|mlp\.fc[12])\.weight\|', k), f'non-expert change: {k}'
p = Path(sys.argv[3])/'openpilot/selfdrive/alpamayo/selected/assets/artifacts.json'
a = json.loads(p.read_text())
a['fp8_finetunes'][name] = {'fp8_index_sha256': hashlib.sha256((E/f'{name}-fp8/index.json').read_bytes()).hexdigest(),
                            'fp8_weights_sha256': ib['sha256'], 'views': views}
text = json.dumps(a, indent=2).replace('''"shards": {
      "selected-bf16.safetensors": "a0dcb081369afb88ac5bcb82824803e5a99dd8981b8814a046bfdb583c8267a4"
    }''', '"shards": {"selected-bf16.safetensors": "a0dcb081369afb88ac5bcb82824803e5a99dd8981b8814a046bfdb583c8267a4"}')
p.write_text(text+'\n')
print('pinned', name, a['fp8_finetunes'][name])
PY
cd $W && git add openpilot/selfdrive/alpamayo/selected/assets/artifacts.json && git commit -q -m "Pin the $NAME expert fine-tune ($VIEWS views)

Only expert layer linears and their scales differ from the base FP8 cache." && git push -q origin HEAD:alpamayo && git log --oneline -1
cp $W/openpilot/selfdrive/alpamayo/selected/assets/artifacts.json $REPO/pc/openpilot/selfdrive/alpamayo/selected/assets/artifacts.json
echo "ready: install code (deploy_karel.sh) then deploy_karel_model.sh NAME=$NAME"
