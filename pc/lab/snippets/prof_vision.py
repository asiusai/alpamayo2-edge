exec(open('/home/john/alpamayo2-edge/pc/lab/snippets/prof.py').read().split("(x, cos, sin, max_tokens, length), _ = ARGS['prefill']")[0])
pairs = []
for i in range(2):
  pairs += [(Tensor.rand(1,720,768)*2-1).cast(dtypes.bfloat16).contiguous().realize(), (Tensor.randn(1,720,1152)*.02).cast(dtypes.bfloat16).contiguous().realize()]
profile('vision pair', lambda: Tensor.realize(*m.encode_group(*pairs)), runs=1, top=30)
