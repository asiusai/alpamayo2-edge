"""gfx1100 model with batched vision and per-shape WMMA tilings.

All new images with the same grid are encoded in one captured graph. Every vision operation is per row (linear,
norm, projector) or per image and head (attention), so features match per-image encoding bit for bit.

Each GEMM uses the fastest tiling that a sweep on the RX 7900 XTX found bit-identical to the original 4x4 tiling.
Tilings change the workgroup layout only; every output keeps its tensor-core accumulation order.
All Euler steps run in one captured graph.

Reasoning tokens are drafted from recent texts and checked up to 32 at a time; only tokens that match the model's own
greedy predictions are kept. Batched checking can round differently from serial decoding, so texts can differ from
models.gfx1100. Set ALPAMAYO_DRAFTS=0 for outputs bit-identical to models.gfx1100.
"""
import hashlib
import os
import numpy as np
import math
from functools import partial
from tinygrad import Device, Tensor, TinyJit, dtypes
from tinygrad.codegen.opt import Opt, OptOps
from tinygrad.codegen.opt.postrange import Scheduler
from tinygrad.helpers import Context, round_up
from tinygrad.uop.ops import AxisType, KernelInfo, Ops, UOp
from models.gfx1100 import Model as Gfx1100, decode_attention, gemv_kernel, pack, softmax_kernel


# (kind, small rows, columns, width) -> (upcast m, upcast n, local m, local n). Swept on demo3 shapes; all bit-exact.
TILES = {
    ('linear', False, 1152, 1152): (2, 4, 1, 2), ('linear', False, 4304, 1152): (2, 2, 1, 1),
    ('linear', False, 1152, 4304): (2, 4, 1, 1),
    ('linear', False, 2048, 2048): (4, 2, 1, 1), ('linear', False, 1024, 2048): (2, 4, 1, 2),
    ('linear', False, 9216, 2048): (4, 4, 2, 4), ('linear', False, 2048, 9216): (4, 2, 2, 2),
    ('linear', True, 2048, 2048): (1, 2, 1, 1), ('linear', True, 1024, 2048): (1, 2, 1, 1),
    ('linear', True, 9216, 2048): (4, 2, 1, 1), ('linear', True, 2048, 9216): (1, 2, 1, 1),
    ('linear', True, 512, 2048): (1, 2, 1, 1), ('linear', True, 1024, 512): (1, 2, 2, 2), ('linear', True, 3840, 512): (4, 2, 2, 2),
    ('linear', True, 512, 3840): (1, 2, 1, 1), ('linear', True, 512, 512): (1, 4, 1, 1), ('linear', True, 2048, 512): (4, 2, 1, 1),
    ('linear', False, 1152, 768): (2, 4, 1, 1), ('linear', False, 11520, 4608): (2, 4, 1, 2), ('linear', False, 2048, 11520): (2, 4, 1, 1),
    ('scores', False, 720, 72): (1, 2, 1, 1), ('values', False, 72, 720): (2, 2, 1, 1),
    ('scores', False, 'keys', 128): (2, 4, 1, 1), ('values', False, 128, 'keys'): (4, 2, 1, 1),
    ('scores', True, 'keys', 128): (2, 2, 1, 1), ('values', True, 128, 'keys'): (1, 2, 1, 1),
}


def tiling(kind, rows, columns, width):
    small = rows <= 64
    for key in ((kind, small, columns, width), (kind, small, 'keys', width), (kind, small, columns, 'keys')):
        if key in TILES:
            return TILES[key]
    return (4, 4, 1, 1)


def matmul_kernel(out, a, b, *extra, shape, tile, causal=False, scaled=False, bias=False, masked=False, masked2d=False):
    batch, heads, kv_heads, rows, columns, width = shape
    up_m, up_n, local_m, local_n = tile
    pr, pc, pw = (round_up(n, 16) for n in (rows, columns, width))
    head = UOp.range(batch*heads, 0)
    m, n, k = UOp.range(round_up(rows, 16*up_m*local_m), 1), UOp.range(round_up(columns, 16*up_n*local_n), 2), UOp.range(pw, 3)
    def index(h, row, length): return ((h*(length//16)+row//16)*(pw//16)+k//16)*256+(k%16//4)*64+(row%16)*4+k%4
    aa = a.flatten().index(index(head, m, pr).valid(m < rows))
    kh = (head//heads)*kv_heads+(head % heads)//(heads//kv_heads)
    bb = b.flatten().index(index(kh, n, pc).valid(n < columns))
    value = (aa*bb).cast(dtypes.float32).reduce(k, arg=Ops.ADD).cast(out.dtype)
    if scaled: value = value*UOp.const(width**-.5, out.dtype)
    if causal: value = (n <= m).where(value, UOp.const(-math.inf, out.dtype))
    if masked: value = extra[0].flatten().index(n.valid(n < columns)).where(value, UOp.const(-math.inf, out.dtype))
    if masked2d: value = extra[0].flatten().index((m*columns+n).valid((m < rows) & (n < columns))).where(value, UOp.const(-math.inf, out.dtype))
    if bias: value = value+extra[0].flatten().index(n.valid(n < columns))
    store = out.flatten().index(((head*rows+m)*columns+n).valid((m < rows) & (n < columns))).store(value)
    s = Scheduler(store.end(head, m, n).sink(arg=KernelInfo()), Device[Device.DEFAULT].renderer)
    s.convert_loop_to_global()
    ranges = s.apply_opt(Opt(OptOps.TC, 0, (-1, 0, 1)))
    axes = (ranges[1].axis_id, ranges[0].axis_id)
    for i, count in ((1, up_m), (0, up_n)):
        if count > 1: s.apply_opt(Opt(OptOps.SPLIT, s.rngs.index(ranges[i]), (count, AxisType.UPCAST)))
    for axis, count in zip(axes, (local_m, local_n)):
        if count == 1: continue
        rng = next(r for r in s.rngs if r.axis_type is AxisType.GLOBAL and r.axis_id == axis)
        s.apply_opt(Opt(OptOps.SPLIT, s.rngs.index(rng), (count, AxisType.LOCAL)))
    ast = s.get_optimized_ast(name_override=f'alpamayo_wmma_{up_m}{up_n}{local_m}{local_n}')
    # This pinned tinygrad revision leaves explicit custom ranges unclosed.
    if ast.ranges:
        assert len(ast.src) == 1 and ast.src[0].op is Ops.END
        ast = ast.replace(src=(ast.src[0].replace(src=ast.src[0].src+tuple(ast.ranges)),))
    return ast


def linear(x, w, packed_weight, bias=None):
    m, k = x.shape; n = w.shape[0]
    out = Tensor.empty(m, n, device=x.device, dtype=dtypes.bfloat16)
    extra = () if bias is None else (bias,)
    if m == 1: return Tensor.custom_kernel(out, x.contiguous(), w, *extra, fxn=gemv_kernel)[0]
    return Tensor.custom_kernel(out, pack(x.reshape(1, 1, m, k)), packed_weight, *extra,
                                fxn=partial(matmul_kernel, shape=(1, 1, 1, m, n, k), tile=tiling('linear', m, n, k), bias=bias is not None))[0]


def attention(q, k, v, causal=False, mask=None):
    if q.shape[1] == 1 and mask is not None: return decode_attention(q, k, v, mask)
    heads, rows, width = q.shape; kv_heads, columns = k.shape[:2]
    q, k, v = q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0)
    scores = Tensor.empty(1, heads, rows, columns, device=q.device, dtype=q.dtype)
    scores = Tensor.custom_kernel(scores, pack(q), pack(k), *(() if mask is None else (mask.contiguous(),)),
        fxn=partial(matmul_kernel, shape=(1, heads, kv_heads, rows, columns, width), tile=tiling('scores', rows, columns, width),
                    causal=causal, scaled=True, masked=mask is not None))[0]
    probs = Tensor.empty(1, heads, round_up(rows, 16), round_up(columns, 16), device=q.device, dtype=q.dtype)
    probs = Tensor.custom_kernel(probs, scores, fxn=softmax_kernel)[0]
    out = Tensor.empty(1, heads, rows, width, device=q.device, dtype=q.dtype)
    return Tensor.custom_kernel(out, probs, pack(v.transpose(-2, -1)),
        fxn=partial(matmul_kernel, shape=(1, heads, kv_heads, rows, width, columns), tile=tiling('values', rows, width, columns)))[0].squeeze(0)


ANCHOR = 135084


def rows_store_kernel(cache, value, index):
    heads, capacity, width = cache.shape
    rows = value.shape[1]
    i = UOp.range(heads*rows*width, 0)
    h, r, c = i//(rows*width), (i//width) % rows, i % width
    offset = (h*capacity+index.flatten().index(0)+r)*width+c
    return cache.flatten().index(offset).store(value.flatten().index(i)).end(i).sink(arg=KernelInfo(name='alpamayo_cache_rows'))


def store_rows(cache, value, index):
    return Tensor.custom_kernel(cache, value.contiguous(), index, fxn=rows_store_kernel)[0].realize()


def draft_attention(q, k, v, mask):
    """Several new queries against the KV cache; mask[r, j] says whether query r may read key j."""
    heads, rows, width = q.shape; kv_heads, columns = k.shape[:2]
    q, k, v = q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0)
    scores = Tensor.empty(1, heads, rows, columns, device=q.device, dtype=q.dtype)
    scores = Tensor.custom_kernel(scores, pack(q), pack(k), mask.contiguous(),
        fxn=partial(matmul_kernel, shape=(1, heads, kv_heads, rows, columns, width), tile=tiling('scores', rows, columns, width),
                    scaled=True, masked2d=True))[0]
    probs = Tensor.empty(1, heads, round_up(rows, 16), round_up(columns, 16), device=q.device, dtype=q.dtype)
    probs = Tensor.custom_kernel(probs, scores, fxn=softmax_kernel)[0]
    out = Tensor.empty(1, heads, rows, width, device=q.device, dtype=q.dtype)
    return Tensor.custom_kernel(out, probs, pack(v.transpose(-2, -1)),
        fxn=partial(matmul_kernel, shape=(1, heads, kv_heads, rows, width, columns), tile=tiling('values', rows, width, columns)))[0].squeeze(0)


def continuation(history, context, limit):
    """Recent text that continues the generated prefix: identical prefix first, then the longest matching suffix."""
    for previous in reversed(history):
        if len(previous) > len(context) and previous[:len(context)] == context:
            return previous[len(context):len(context)+limit]
    for width in range(min(8, len(context)), 0, -1):
        suffix = context[-width:]
        for previous in reversed(history):
            for end in range(len(previous)-1, width-1, -1):
                if previous[end-width:end] == suffix:
                    return previous[end:end+limit]
    return []


class Model(Gfx1100):
    def linear(self, x, name):
        weight = self.w(name+'.weight')
        bias = self.w(name+'.bias') if name+'.bias' in self.weights else None
        if x.shape[0] > 1 and name not in self.packed_weights:
            self.packed_weights[name] = pack(weight.reshape(1, 1, *weight.shape)).realize()
        return linear(x, weight, self.packed_weights.get(name), bias)

    def attention(self, q, k, v, causal=False, mask=None):
        return attention(q, k, v, causal, mask)

    def flow(self, noise, cache, cursor):
        # All Euler steps in one captured graph: no per-step time uploads, state realizes or graph launches.
        capacity = cache[0][0].shape[1]
        length = getattr(self, 'cache_length', capacity)
        slots = np.arange(capacity+64)
        mask = Tensor((slots < length) | (slots >= capacity), device=self.device).realize()
        from model import rotary_tables
        c, s = rotary_tables(np.broadcast_to(np.arange(64)+cursor, (3, 64)), self.device, dtypes.bfloat16)
        steps = self.euler_steps
        key = (capacity, steps)
        if key not in self.flow_graphs:
            times = np.linspace(0, 1, steps+1, dtype=np.float32)
            def run(x, mask, c, s, *kv):
                pairs = list(zip(kv[::2], kv[1::2]))
                for step in range(steps):
                    t = Tensor([float(times[step])], device=self.device)
                    velocity = self.compute_velocity(x, t, (c, s), pairs, mask)
                    x = (x+float(times[step+1]-times[step])*velocity).realize()
                return x
            self.flow_graphs[key] = TinyJit(run)
        x = Tensor(noise.reshape(64, 2), device=self.device).float().realize()
        return self.flow_graphs[key](x, mask, c, s, *[v for pair in cache for v in pair]).numpy()


    def draft_pass(self, tokens, index, step, cos, sin, *kv):
        """Run proposed tokens through the language model; return greedy predictions and normalized hidden states."""
        rows = tokens.shape[0]
        x = self.w('vlm.model.language_model.embed_tokens.weight')[tokens]
        offsets = step.reshape(1, 1)+Tensor.arange(rows).reshape(rows, 1)
        c, s = cos[offsets.reshape(-1)], sin[offsets.reshape(-1)]
        capacity = kv[0].shape[1]
        mask = Tensor.arange(capacity).reshape(1, capacity) <= (index.reshape(1, 1)+Tensor.arange(rows).reshape(rows, 1))
        from model import apply_rope
        prefix = 'vlm.model.language_model'
        for i in range(28):
            name = f'{prefix}.layers.{i}'
            z = self.norm(x, name+'.input_layernorm')
            q, k, v = [self.linear(z, name+'.self_attn.'+n+'_proj').reshape(rows, heads, 128).transpose(0, 1)
                       for n, heads in [('q', 16), ('k', 8), ('v', 8)]]
            q, k = apply_rope(q, c, s), apply_rope(k, c, s)
            keys, values = store_rows(kv[2*i], k, index), store_rows(kv[2*i+1], v, index)
            z = draft_attention(q, keys, values, mask).transpose(0, 1).reshape(rows, 2048)
            x = x+self.linear(z, name+'.self_attn.o_proj')
            z = self.linear(self.norm(x, name+'.post_attention_layernorm'), name+'.mlp.fc1').relu().square()
            x = (x+self.linear(z, name+'.mlp.fc2')).realize()
        hidden = self.norm(x, prefix+'.norm').realize()
        vocab = Tensor.arange(self.config['vlm_config']['text_config']['vocab_size']).to(self.device)
        allowed = ((vocab < 131072) | (vocab >= 135072)) & (vocab != 11)
        predicted = allowed.where(self.linear(hidden, 'vlm.lm_head'), -float('inf')).argmax(-1).realize()
        return predicted, hidden

    def first_token(self, x):
        if 'first' not in self.draft_graphs:
            self.draft_graphs['first'] = TinyJit(lambda h: self.pick_token(h)[0])
        return int(self.draft_graphs['first'](x).numpy().reshape(-1)[0])

    @Context(TC_OPT=2)
    def generate(self, ids, pixels, grids, max_tokens=256):
        from model import positions, rotary_tables
        pos, delta = positions(ids, grids)
        images = self.vision(pixels, grids)
        prefill_capacity = ((len(ids)+127)//128)*128
        padding = prefill_capacity-len(ids)
        padded_pos = np.concatenate((pos, np.broadcast_to(np.arange(len(ids), prefill_capacity)+delta, (3, padding))), axis=1)
        c, s = rotary_tables(padded_pos, self.device, dtypes.bfloat16)
        embeddings = self.embed_images(ids, images).pad(((0, padding), (0, 0))).contiguous().realize()
        capacity = ((len(ids)+max_tokens+255)//256)*256
        key = (prefill_capacity, capacity)
        if key not in self.prefill_graphs:
            self.prefill_graphs[key] = TinyJit(lambda x, c, s, n=capacity: self.prefill(x, c, s, n))
        x, cache = self.prefill_graphs[key](embeddings, c, s)
        cos, sin = rotary_tables(np.broadcast_to(np.arange(max_tokens)+len(ids)+delta, (3, max_tokens)), self.device, dtypes.bfloat16)
        x = x[len(ids)-1:len(ids)].clone().realize()
        kv = [v for pair in cache for v in pair]
        generated, index = [], len(ids)
        token = self.first_token(x)
        while len(generated) < max_tokens:
            step = index-len(ids)
            proposal = [token]+continuation(self.text_history, generated+[token], 31)[:max_tokens-step-1]
            if self.use_drafts and len(proposal) > 1 and step+32 <= max_tokens:
                rows = 16 if len(proposal) <= 16 else 32
                padded = proposal+[proposal[-1]]*(rows-len(proposal))
                graph_key = (capacity, rows)
                if graph_key not in self.draft_graphs:
                    self.draft_graphs[graph_key] = TinyJit(self.draft_pass)
                predicted, hidden = self.draft_graphs[graph_key](Tensor(np.array(padded, np.int32), device=self.device),
                    Tensor([index], device=self.device), Tensor([step], device=self.device), cos, sin, *kv)
                predicted = predicted.numpy().reshape(-1).astype(int).tolist()
                accepted = [proposal[0]]
                for i in range(len(proposal)-1):
                    if accepted[-1] == ANCHOR or predicted[i] != proposal[i+1]:
                        break
                    accepted.append(proposal[i+1])
                self.draft_stats['rounds'] += 1
                self.draft_stats['drafted'] += len(accepted)
                if ANCHOR in accepted:
                    accepted = accepted[:accepted.index(ANCHOR)+1]
                    generated += accepted
                    end = index+len(accepted)-1
                    self.cache_length = end+1
                    self.text_history = [h for h in self.text_history if h != generated][-63:]+[generated]
                    return generated, cache, end+delta+1
                generated += accepted
                index += len(accepted)
                token = predicted[len(accepted)-1]
                x = hidden[len(accepted)-1:len(accepted)].clone().realize()
                continue
            # No draft: the original six-token serial block from the current hidden state.
            count = min(self.decode_block_size, max_tokens-step)
            graph_key = (capacity, count)
            if graph_key not in self.next_graphs:
                self.next_graphs[graph_key] = TinyJit(lambda *a, n=count: self.token_block(*a, count=n))
            x, tokens = self.next_graphs[graph_key](x, Tensor([index], device=self.device).realize(),
                Tensor([step], device=self.device).realize(), cos, sin, *kv)
            for offset, value in enumerate(tokens.numpy().reshape(-1)):
                generated.append(int(value))
                self.draft_stats['serial'] += 1
                if int(value) == ANCHOR:
                    self.cache_length = index+offset+1
                    self.text_history = [h for h in self.text_history if h != generated][-63:]+[generated]
                    return generated, cache, index+offset+delta+1
            index += count
            token = self.first_token(x)
        raise RuntimeError('Trajectory anchor not generated within max_tokens')

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.draft_graphs, self.text_history = {}, []
        self.use_drafts = os.environ.get('ALPAMAYO_DRAFTS', '1') == '1'  # 0: bit-identical to models.gfx1100
        self.draft_stats = {'rounds': 0, 'drafted': 0, 'serial': 0}
        self.flow_graphs = {}
        self.batch_vision_graphs = {}
        self.copy_graphs = {}

    def encode_batch(self, pixels, emb):
        # pixels: (images, patches, 768) float32 on device; emb: (patches, 1152) bf16.
        images, patches = pixels.shape[:2]
        prefix = 'vlm.model.visual'
        x = pixels.cast(dtypes.bfloat16).reshape(images*patches, 768)
        x = (self.linear(x, prefix+'.embeddings.patch_embedding').reshape(images, patches, 1152)+emb).reshape(images*patches, 1152).realize()
        for i in range(27):
            p = f'{prefix}.encoder.layers.{i}'
            z = self.norm(x, p+'.layer_norm1', 1e-6, False)
            # Images become separate attention heads; no key crosses an image boundary.
            q, k, v = [self.linear(z, p+'.self_attn.'+n+'_proj').reshape(images, patches, 16, 72).permute(0, 2, 1, 3)
                       .reshape(images*16, patches, 72) for n in ('q', 'k', 'v')]
            h = attention(q, k, v).reshape(images, 16, patches, 72).permute(0, 2, 1, 3).reshape(images*patches, 1152)
            x = x+self.linear(h, p+'.self_attn.out_proj')
            z = self.linear(self.norm(x, p+'.layer_norm2', 1e-6, False), p+'.mlp.fc1').gelu()
            x = (x+self.linear(z, p+'.mlp.fc2')).realize()
        x = self.norm(x, prefix+'.post_layernorm', 1e-6, False)
        x = self.norm(x, 'vlm.model.projector.norm', 1e-6, False).reshape(-1, 4608)
        x = self.linear(x, 'vlm.model.projector.linear_fc1').gelu(approximate='none')
        return self.linear(x, 'vlm.model.projector.linear_fc2').realize()

    def vision(self, pixels, grids):
        keys, offset, pending = [], 0, {}
        for t, h, w in grids:
            h, w = int(h), int(w)
            assert t == 1, 'Image mode only'
            patches = pixels[offset:offset+h*w]
            # SHA-256 uses the CPU's SHA extensions: ~0.8 ms per image versus ~1.9 ms for BLAKE2b.
            digest = hashlib.sha256(np.asarray([t, h, w], dtype=np.int64).tobytes())
            digest.update(np.ascontiguousarray(patches))
            key = digest.digest()
            if key not in self.image_features and key not in pending:
                pending[key] = (h, w, patches)
            keys.append(key)
            offset += h*w
        assert offset == pixels.shape[0]
        by_grid = {}
        for key, (h, w, patches) in pending.items():
            by_grid.setdefault((h, w), []).append((key, patches))
        for (h, w), items in by_grid.items():
            count = len(items)
            batch = Tensor(np.stack([patches for _, patches in items]), device=self.device)
            graph_key = (count, h, w)
            if graph_key not in self.batch_vision_graphs:
                self.batch_vision_graphs[graph_key] = TinyJit(self.encode_batch)
            out = self.batch_vision_graphs[graph_key](batch, self.vision_position(h, w))
            # The captured output storage is reused by the next call; copy into storage owned by the cache.
            owned = Tensor(UOp.new_buffer(self.device, out.numel(), out.dtype)).reshape(out.shape)
            owned.uop.base.buffer.ensure_allocated()  # Explicit storage: an unallocated empty() target aliases on replay.
            if out.shape not in self.copy_graphs:
                self.copy_graphs[out.shape] = TinyJit(lambda dst, src: dst.assign(src).realize())
            self.copy_graphs[out.shape](owned, out)
            tokens = out.shape[0]//count
            for i, (key, _) in enumerate(items):
                self.image_features[key] = owned[i*tokens:(i+1)*tokens]
        chunks = []
        for key in keys:
            self.image_features.move_to_end(key)
            chunks.append(self.image_features[key])
        while len(self.image_features) > 128:
            self.image_features.popitem(last=False)
        return chunks[0].cat(*chunks[1:], dim=0)
