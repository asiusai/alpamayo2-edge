"""Copy the 'kind' label (clean/ambiguous) from one sample set to the same-named samples of another (e.g. 180/60 -> 120/30).
Usage: copy_labels.py SOURCE_DIR TARGET_DIR"""
import json, sys
from collections import Counter
from pathlib import Path
src, dst = Path(sys.argv[1]), Path(sys.argv[2]); c = Counter()
for f in dst.glob('*.json'):
  s = src/f.name
  if not s.exists(): c['no source'] += 1; continue
  k = json.loads(s.read_text())['kind']; m = json.loads(f.read_text())
  if m['kind'] != k: m['kind'] = k; f.write_text(json.dumps(m)); c['changed'] += 1
  c[k] += 1
print(dict(c))
