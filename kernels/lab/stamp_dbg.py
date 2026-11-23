print(len(s.names), s.names[:5])
cn = s.compiled_names(); print(len(cn), cn[:12])
lin = s.jit.captured._linear
print(len(lin.src), [ (c.op, type(c.arg).__name__, type(getattr(c.arg,'aux',None)).__name__) for c in lin.src[:4]])
c = lin.src[0]; c = c.src[0] if c.op.name=='AFTER' else c
aux = c.arg.aux; print([f for f in dir(aux) if not f.startswith('_')]); print(aux.kernels[0][:3] if aux.kernels else None)
