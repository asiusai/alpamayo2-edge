from functools import partial, reduce
import math
from tinygrad import Device, Tensor, dtypes
from tinygrad.codegen.opt import Opt, OptOps
from tinygrad.codegen.opt.postrange import Scheduler
from tinygrad.dtype import AddrSpace
from tinygrad.helpers import round_up
from tinygrad.uop.ops import AxisType, KernelInfo, Ops, UOp
from model import Model as BaseModel


def warp_reduce(value,maximum=False):
    shuffle=UOp.custom_function('llvm.amdgcn.ds.swizzle',dtype=dtypes.int32)
    for shift in (16,8,4,2,1):
        other=shuffle.call(value.bitcast(dtypes.int32),UOp.const(0x1f|(shift<<10),dtypes.int32)).bitcast(dtypes.float32)
        value=value.maximum(other) if maximum else value+other
    return value


def pack(x):
    batch,heads,rows,width=x.shape
    pr,pw=round_up(rows,16),round_up(width,16)
    return x.pad(((0,0),(0,0),(0,pr-rows),(0,pw-width))).reshape(batch,heads,pr//16,16,pw//16,4,4).permute(0,1,2,4,5,3,6).contiguous()


def matmul_kernel(out,a,b,*extra,shape,causal=False,scaled=False,bias=False,masked=False):
    batch,heads,kv_heads,rows,columns,width=shape
    pr,pc,pw=(round_up(n,16) for n in (rows,columns,width))
    head=UOp.range(batch*heads,0)
    m,n,k=UOp.range(round_up(rows,64),1),UOp.range(round_up(columns,64),2),UOp.range(pw,3)
    def index(h,row,length): return ((h*(length//16)+row//16)*(pw//16)+k//16)*256+(k%16//4)*64+(row%16)*4+k%4
    aa=a.flatten().index(index(head,m,pr).valid(m<rows))
    kh=(head//heads)*kv_heads+(head%heads)//(heads//kv_heads)
    bb=b.flatten().index(index(kh,n,pc).valid(n<columns))
    value=(aa*bb).cast(dtypes.float32).reduce(k,arg=Ops.ADD).cast(out.dtype)
    if scaled: value=value*UOp.const(width**-.5,out.dtype)
    if causal: value=(n<=m).where(value,UOp.const(-math.inf,out.dtype))
    if masked: value=extra[0].flatten().index(n.valid(n<columns)).where(value,UOp.const(-math.inf,out.dtype))
    if bias: value=value+extra[0].flatten().index(n.valid(n<columns))
    store=out.flatten().index(((head*rows+m)*columns+n).valid((m<rows)&(n<columns))).store(value)
    scheduler=Scheduler(store.end(head,m,n).sink(arg=KernelInfo()),Device[Device.DEFAULT].renderer)
    scheduler.convert_loop_to_global()
    ranges=scheduler.apply_opt(Opt(OptOps.TC,0,(-1,0,1)))
    for i in (1,0): scheduler.apply_opt(Opt(OptOps.SPLIT,scheduler.rngs.index(ranges[i]),(4,AxisType.UPCAST)))
    ast=scheduler.get_optimized_ast(name_override='alpamayo_wmma')
    # This pinned tinygrad revision leaves explicit custom ranges unclosed.
    if ast.ranges:
        assert len(ast.src)==1 and ast.src[0].op is Ops.END
        ast=ast.replace(src=(ast.src[0].replace(src=ast.src[0].src+tuple(ast.ranges)),))
    return ast


def gemv_kernel(out,x,w,*bias):
    columns,width=w.shape
    group=UOp.range((columns+3)//4,0,AxisType.GLOBAL)
    wave,lane=UOp.range(4,1,AxisType.LOCAL),UOp.range(32,-1,AxisType.WARP)
    row=group*4+wave
    r=UOp.range((width+31)//32,2,AxisType.LOOP)
    col=r*32+lane
    value=(x.flatten().index(col.valid(col<width)).cast(dtypes.float32)*
           w.flatten().index((row*width+col).valid((row<columns)&(col<width))).cast(dtypes.float32)).reduce(r,arg=Ops.ADD)
    value=warp_reduce(value).cast(out.dtype)
    if bias: value=value+bias[0].flatten().index(row.valid(row<columns))
    return out.flatten().index(row.valid((row<columns)&lane.eq(0))).store(value).end(group,wave,lane).sink(arg=KernelInfo(name='alpamayo_bf16_gemv',opts_to_apply=()))


def linear(x,w,packed_weight,bias=None):
    m,k=x.shape;n=w.shape[0]
    out=Tensor.empty(m,n,device=x.device,dtype=dtypes.bfloat16)
    extra=() if bias is None else (bias,)
    if m==1: return Tensor.custom_kernel(out,x.contiguous(),w,*extra,fxn=gemv_kernel)[0]
    return Tensor.custom_kernel(out,pack(x.reshape(1,1,m,k)),packed_weight,*extra,
                               fxn=partial(matmul_kernel,shape=(1,1,1,m,n,k),bias=bias is not None))[0]


def softmax_kernel(out,scores,packed=True):
    batch,heads,rows,columns=scores.shape
    row=UOp.range(batch*heads*rows,0,AxisType.GLOBAL)
    wave,lane=UOp.range(4,1,AxisType.LOCAL),UOp.range(32,-1,AxisType.WARP)
    cols=[wave*32+lane+i*128 for i in range((columns+127)//128)]
    values=[(c<columns).where(scores.flatten().index((row*columns+c).valid(c<columns)).cast(dtypes.float32),UOp.const(-math.inf,dtypes.float32)) for c in cols]
    maximum=warp_reduce(reduce(UOp.maximum,values),True)
    maxima=UOp.alloc((4,),dtypes.float32,addrspace=AddrSpace.LOCAL)
    store=maxima.index(wave.valid(lane.eq(0))).store(maximum)
    maximum=reduce(UOp.maximum,[maxima.after(store).index(i) for i in range(4)])
    exps=[((v-maximum)*math.log2(math.e)).exp2() for v in values]
    total=warp_reduce(sum(exps))
    sums=UOp.alloc((4,),dtypes.float32,addrspace=AddrSpace.LOCAL)
    store=sums.index(wave.valid(lane.eq(0))).store(total)
    inv=sum(sums.after(store).index(i) for i in range(4)).reciprocal()
    def index(c):
        if not packed:return row*columns+c
        return (((row//rows)*(round_up(rows,16)//16)+(row%rows)//16)*(round_up(columns,16)//16)+c//16)*256+(c%16//4)*64+(row%rows%16)*4+c%4
    stores=[out.flatten().index(index(c).valid(c<(round_up(columns,16) if packed else columns))).store((e*inv).cast(out.dtype)) for c,e in zip(cols,exps,strict=True)]
    return UOp.group(*stores).end(row,wave,lane).sink(arg=KernelInfo(name='alpamayo_softmax',opts_to_apply=()))


def attention(q,k,v,causal=False,mask=None):
    if q.shape[1]==1 and mask is not None:return decode_attention(q,k,v,mask)
    heads,rows,width=q.shape;kv_heads,columns=k.shape[:2]
    q,k,v=q.unsqueeze(0),k.unsqueeze(0),v.unsqueeze(0)
    scores=Tensor.empty(1,heads,rows,columns,device=q.device,dtype=q.dtype)
    scores=Tensor.custom_kernel(scores,pack(q),pack(k),*(() if mask is None else (mask.contiguous(),)),
        fxn=partial(matmul_kernel,shape=(1,heads,kv_heads,rows,columns,width),causal=causal,scaled=True,masked=mask is not None))[0]
    probs=Tensor.empty(1,heads,round_up(rows,16),round_up(columns,16),device=q.device,dtype=q.dtype)
    probs=Tensor.custom_kernel(probs,scores,fxn=softmax_kernel)[0]
    out=Tensor.empty(1,heads,rows,width,device=q.device,dtype=q.dtype)
    return Tensor.custom_kernel(out,probs,pack(v.transpose(-2,-1)),
        fxn=partial(matmul_kernel,shape=(1,heads,kv_heads,rows,width,columns)))[0].squeeze(0)


def decode_scores(out,q,k,mask):
    batch,heads,_,width=q.shape;kv_heads,keys=k.shape[1:3]
    tiles=(keys+3)//4
    group=UOp.range(batch*heads*tiles,0,AxisType.GLOBAL)
    wave,lane=UOp.range(4,1,AxisType.LOCAL),UOp.range(32,-1,AxisType.WARP)
    head,key=group//tiles,(group%tiles)*4+wave
    kh=(head//heads)*kv_heads+(head%heads)//(heads//kv_heads)
    value=sum(q.flatten().index(head*width+c).cast(dtypes.float32)*
              k.flatten().index(((kh*keys+key)*width+c).valid(key<keys)).cast(dtypes.float32) for c in (lane*4+j for j in range(4)))
    value=warp_reduce(value).cast(dtypes.bfloat16)*UOp.const(width**-.5,dtypes.bfloat16)
    value=mask.flatten().index(key.valid(key<keys)).where(value,UOp.const(-math.inf,dtypes.bfloat16))
    return out.flatten().index((head*keys+key).valid((key<keys)&lane.eq(0))).store(value).end(group,wave,lane).sink(arg=KernelInfo(name='alpamayo_decode_scores',opts_to_apply=()))


def decode_values(out,probs,v):
    batch,heads,parts,width=out.shape;kv_heads,keys=v.shape[1:3]
    group=UOp.range(batch*heads*parts,0,AxisType.GLOBAL)
    wave,lane=UOp.range(4,1,AxisType.LOCAL),UOp.range(32,-1,AxisType.WARP)
    head,part,col=group//parts,group%parts,wave*32+lane
    kh=(head//heads)*kv_heads+(head%heads)//(heads//kv_heads)
    r=UOp.range(128,2,AxisType.LOOP);key=part*128+r
    value=(probs.flatten().index((head*keys+key).valid(key<keys)).cast(dtypes.float32)*
           v.flatten().index(((kh*keys+key)*width+col).valid(key<keys)).cast(dtypes.float32)).reduce(r,arg=Ops.ADD)
    return out.flatten().index(group*width+col).store(value).end(group,wave,lane).sink(arg=KernelInfo(name='alpamayo_decode_values',opts_to_apply=()))


def decode_attention(q,k,v,mask):
    heads,_,width=q.shape;keys=k.shape[1]
    q,k,v=q.unsqueeze(0),k.unsqueeze(0),v.unsqueeze(0)
    scores=Tensor.empty(1,heads,1,keys,device=q.device,dtype=q.dtype)
    scores=Tensor.custom_kernel(scores,q.contiguous(),k.contiguous(),mask.contiguous(),fxn=decode_scores)[0]
    probs=Tensor.empty(scores.shape,device=q.device,dtype=q.dtype)
    probs=Tensor.custom_kernel(probs,scores,fxn=partial(softmax_kernel,packed=False))[0]
    partials=Tensor.empty(1,heads,(keys+127)//128,width,device=q.device,dtype=dtypes.float32)
    partials=Tensor.custom_kernel(partials,probs,v.contiguous(),fxn=decode_values)[0]
    return partials.sum(2,keepdim=True).cast(q.dtype).squeeze(0)


def norm_kernel(out,x,weight,*bias,rms,eps):
    width=x.shape[-1];rows=x.numel()//width
    row=UOp.range(rows,0,AxisType.GLOBAL)
    wave,lane=UOp.range(4,1,AxisType.LOCAL),UOp.range(32,-1,AxisType.WARP)
    cols=[wave*32+lane+i*128 for i in range((width+127)//128)]
    values=[x.flatten().index((row*width+c).valid(c<width)).cast(dtypes.float32) for c in cols]
    def total(value):
        value=warp_reduce(value)
        shared=UOp.alloc((4,),dtypes.float32,addrspace=AddrSpace.LOCAL)
        store=shared.index(wave.valid(lane.eq(0))).store(value)
        return sum(shared.after(store).index(i) for i in range(4))
    mean=UOp.const(0.,dtypes.float32) if rms else total(sum(values))/width
    centered=[(c<width).where(v-mean,UOp.const(0.,dtypes.float32)) for c,v in zip(cols,values)]
    inv=(total(sum(v*v for v in centered))/width+eps).rsqrt()
    stores=[]
    for c,v in zip(cols,centered):
        w=weight.flatten().index(c.valid(c<width))
        value=(v*inv).cast(out.dtype)*w if rms else v*inv*w.cast(dtypes.float32)+bias[0].flatten().index(c.valid(c<width)).cast(dtypes.float32)
        stores.append(out.flatten().index((row*width+c).valid(c<width)).store(value.cast(out.dtype)))
    return UOp.group(*stores).end(row,wave,lane).sink(arg=KernelInfo(name='alpamayo_norm',opts_to_apply=()))


def norm(x,weight,bias=None,eps=1e-5,rms=True):
    out=Tensor.empty(x.shape,device=x.device,dtype=x.dtype)
    return Tensor.custom_kernel(out,x.contiguous(),weight,*(() if bias is None else (bias,)),fxn=partial(norm_kernel,rms=rms,eps=eps))[0]


def cache_store_kernel(cache,value,index):
    heads,capacity,width=cache.shape
    i=UOp.range(heads*width,0)
    offset=(i//width*capacity+index.flatten().index(0))*width+i%width
    return cache.flatten().index(offset).store(value.flatten().index(i)).end(i).sink(arg=KernelInfo())


def update_cache(cache,value,index):
    return Tensor.custom_kernel(cache,value.contiguous(),index,fxn=cache_store_kernel)[0].realize()


class Model(BaseModel):
    decode_block_size=6
    tensor_cores=1

    def matmul(self,a,b):
        return a.cast(dtypes.float16).matmul(b.cast(dtypes.float16),dtype=dtypes.float32).cast(a.dtype)

    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        device=Device[self.device]
        if getattr(device,'arch',None)!='gfx1100' or 'LLVM' not in type(device.renderer).__name__:
            raise ValueError('models.gfx1100 requires a gfx1100 GPU with DEV=AMD:LLVM')
        self.packed_weights={}

    def linear(self,x,name):
        weight=self.w(name+'.weight')
        bias=self.w(name+'.bias') if name+'.bias' in self.weights else None
        if x.shape[0]>1 and name not in self.packed_weights:
            self.packed_weights[name]=pack(weight.reshape(1,1,*weight.shape)).realize()
        return linear(x,weight,self.packed_weights.get(name),bias)

    def attention(self,q,k,v,causal=False,mask=None):
        return attention(q,k,v,causal,mask)

    def norm(self,x,name,eps=1e-5,rms=True):
        return norm(x,self.w(name+'.weight'),None if rms else self.w(name+'.bias'),eps,rms)

    def update_cache(self,cache,value,index):
        return update_cache(cache,value,index)
