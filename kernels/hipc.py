"""Compile a HIP kernel source for gfx1200 on the PC: hipc.py SOURCE.cpp OUT.hsaco [NAME=VALUE ...]"""
import sys, hashlib
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'pc/vendor/tinygrad'))
from tinygrad.runtime.support.compiler_amd import compile_hip
src = open(sys.argv[1]).read()
for d in sys.argv[3:]:
  src = f'#define {d.replace("=", " ")}\n' + src
b = compile_hip(src, 'gfx1200')
open(sys.argv[2], 'wb').write(b)
print(len(b), hashlib.sha256(b).hexdigest()[:16])
