t = Tensor.empty(4, 4, dtype=dtypes.bfloat16)
print(t.uop.op, t.uop.base.op, [u.op for u in t.uop.toposort()][:6])
try: print(t.uop.base.buffer.is_allocated())
except Exception as e: print('err', repr(e)[:200])
