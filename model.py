# SPDX-License-Identifier: Apache-2.0
# Architecture adapted from NVIDIA Alpamayo 2 Edge and Hugging Face Cosmos3 Edge.
import json
import math
import hashlib
from collections import OrderedDict
from pathlib import Path
import numpy as np
from tinygrad import Tensor, TinyJit, Device, dtypes
from tinygrad.helpers import Context
from tinygrad.nn.state import safe_load


def positions(ids, grids):
    out = np.empty((3, len(ids)), np.int32)
    i = cursor = 0
    for t, h, w in grids:
        start = ids.index(19, i)
        out[:, i:start] = np.arange(cursor, cursor + start-i)
        cursor += start-i
        mesh = np.indices((int(t), int(h)//2, int(w)//2)).reshape(3, -1)
        end = start + mesh.shape[1]
        assert ids[start:end] == [19]*mesh.shape[1]
        out[:, start:end] = mesh + cursor
        cursor += int(mesh.max()) + 1
        i = end
    out[:, i:] = np.arange(cursor, cursor + len(ids)-i)
    return out, int(out.max()) + 1 - len(ids)


def rotary_tables(pos, device, dtype):
    freq = np.float32(1) / np.power(np.float32(1e8), np.arange(0, 128, 2, dtype=np.float32)/128)
    angles = pos[..., None].astype(np.float32)*freq
    mixed = angles[0].copy()
    mixed[:, 1:60:3] = angles[1, :, 1:60:3]
    mixed[:, 2:60:3] = angles[2, :, 2:60:3]
    mixed = np.concatenate((mixed, mixed), axis=-1)
    c = Tensor(np.cos(mixed), device=device).cast(dtype).realize()
    s = Tensor(np.sin(mixed), device=device).cast(dtype).realize()
    return c, s


def apply_rope(x, c, s):
    return x*c + (-x[..., 64:]).cat(x[..., :64], dim=-1)*s


class Model:
    euler_steps=10
    decode_block_size=1
    tensor_cores=0

    def __init__(self, checkpoint, device=None):
        root = Path(checkpoint)
        self.config = json.loads((root/'config.json').read_text())
        index = json.loads((root/'model.safetensors.index.json').read_text())['weight_map']
        self.weights = {}
        for filename in sorted(set(index.values())):
            self.weights.update(safe_load(root/filename))
        self.device, self.loaded = device or Device.DEFAULT, {}
        self.vision_graphs={}
        self.vision_positions={}
        self.expert_graphs={}
        self.next_graphs={}
        self.prefill_graphs={}
        self.causal_masks={}
        self.image_features=OrderedDict()

    def w(self, name):
        if name not in self.loaded:
            self.loaded[name] = self.weights[name].to(self.device).realize()
        return self.loaded[name]

    def predict(self,ids,pixels,grids,noise,prediction):
        with Context(TC=self.tensor_cores):
            tokens,cache,cursor=self.generate(ids,pixels,grids)
            actions=self.flow(noise,cache,cursor)
        return {'tokens':tokens,'actions':actions.tolist(),'xyz':self.trajectory(actions,prediction).tolist()}

    def trajectory(self,actions,prediction):
        """Decode the checkpoint's course-heading action space in metres."""
        cfg=self.config['expert_config']['action_space_cfg']
        dt=np.float32(cfg['dt'])
        history=np.asarray(prediction['ego_history_xyz'],np.float32).reshape(20,3)
        rotations=np.asarray(prediction['ego_history_rot'],np.float32).reshape(20,3,3)
        yaw=np.unwrap(np.arctan2(rotations[:,1,0],rotations[:,0,0]))
        n=len(history);i=np.arange(n-1)
        a=np.zeros((2*(n-1),n),np.float32)
        a[2*i,i]=np.cos(yaw[:-1]);a[2*i,i+1]=np.cos(yaw[1:])
        a[2*i+1,i]=np.sin(yaw[:-1]);a[2*i+1,i+1]=np.sin(yaw[1:])
        d=np.diff(np.eye(n,dtype=np.float32),n=3,axis=0)
        lhs=a.T@a+np.float32(cfg['v_lambda']/float(dt)**6)*(d.T@d)+np.float32(cfg['v_ridge'])*np.eye(n,dtype=np.float32)
        rhs=a.T@(np.diff(history[:,:2],axis=0)*np.float32(2/dt)).reshape(-1)
        v0=np.linalg.solve(lhs,rhs)[-1]
        chord=history[-1,:2]-history[-2,:2];heading=yaw[-2:].mean()
        along=np.cos(heading)*chord[0]+np.sin(heading)*chord[1]
        across=np.cos(heading)*chord[1]-np.sin(heading)*chord[0]
        theta0=np.arctan2(across*np.sign(along),abs(along))
        if abs(theta0)>.05:theta0=np.float32(0)
        theta0*=np.clip((np.linalg.norm(chord)/dt-.5)/.5,0,1)
        actions=np.asarray(actions,np.float32)
        accel=actions[:,0]*np.float32(cfg['accel_std'])+np.float32(cfg['accel_mean'])
        curvature=actions[:,1]*np.float32(cfg['curvature_std'])+np.float32(cfg['curvature_mean'])
        velocity=np.concatenate(([v0],v0+np.cumsum(accel*dt)))
        theta=np.concatenate(([theta0],theta0+np.cumsum(curvature*velocity[:-1]*dt)+np.cumsum(curvature*accel*np.float32(.5*dt*dt))))
        xyz=np.zeros((len(actions),3),np.float32)
        for axis,fn in enumerate((np.cos,np.sin)):
            xyz[:,axis]=np.cumsum(velocity[:-1]*fn(theta[:-1])*dt*.5)+np.cumsum(velocity[1:]*fn(theta[1:])*dt*.5)
        xyz[:,2]=history[-1,2]
        return xyz

    def norm(self, x, name, eps=1e-5, rms=True):
        f = x.float()
        if rms:
            return (f*(f.square().mean(-1, keepdim=True)+eps).rsqrt()).cast(x.dtype)*self.w(name+'.weight')
        f = f-f.mean(-1, keepdim=True)
        return (f*(f.square().mean(-1, keepdim=True)+eps).rsqrt()*self.w(name+'.weight').float()+self.w(name+'.bias').float()).cast(x.dtype)

    def vision_position(self,h,w):
        if (h,w) in self.vision_positions:return self.vision_positions[h,w]
        learned=self.w('vlm.model.visual.embeddings.position_embedding.weight').float().numpy().reshape(16,16,-1)
        def axis(n):
            scale=16/n
            centers=(np.arange(n,dtype=np.float32)+.5)*scale
            distance=np.abs((np.arange(16,dtype=np.float32)[None,:]+.5)-centers[:,None])
            weights=np.maximum(1-distance/max(scale,1),0)
            return weights/weights.sum(axis=1,keepdims=True)
        emb=np.einsum('ih,hwc,jw->ijc',axis(h),learned,axis(w),optimize=True)
        emb=emb.reshape(h//2,2,w//2,2,-1).transpose(0,2,1,3,4).reshape(h*w,-1).astype(np.float32)
        self.vision_positions[h,w]=Tensor(emb,device=self.device).cast(dtypes.bfloat16).realize()
        return self.vision_positions[h,w]

    def vision(self, pixels, grids):
        chunks, offset = [], 0
        for frame, (t, h, w) in enumerate(grids):
            h, w = int(h), int(w)
            assert t == 1, 'Image mode only'
            patches=pixels[offset:offset+h*w]
            digest=hashlib.blake2b(digest_size=20)
            digest.update(np.asarray([t,h,w],dtype=np.int64).tobytes())
            digest.update(patches.tobytes())
            key=digest.digest()
            if key not in self.image_features:
                x=Tensor(patches,device=self.device).cast(dtypes.bfloat16).contiguous().realize()
                self.image_features[key]=self.vision_frame(x,self.vision_position(h,w))
            self.image_features.move_to_end(key)
            x=self.image_features[key]
            while len(self.image_features)>128:self.image_features.popitem(last=False)
            offset += h*w
            chunks.append(x)
            print(f'vision {frame+1}/{len(grids)}', flush=True)
        assert offset == pixels.shape[0]
        return chunks[0].cat(*chunks[1:], dim=0)

    def encode_image(self, pixels, emb):
        prefix = 'vlm.model.visual'
        x = (self.linear(pixels, prefix+'.embeddings.patch_embedding')+emb).realize()
        for i in range(27):
            p = f'{prefix}.encoder.layers.{i}'
            z = self.norm(x, p+'.layer_norm1', 1e-6, False)
            q, k, v = [self.linear(z, p+'.self_attn.'+n+'_proj').reshape(-1, 16, 72).transpose(0, 1) for n in ('q', 'k', 'v')]
            x = x+self.linear(self.attention(q, k, v).transpose(0, 1).reshape(-1, 1152), p+'.self_attn.out_proj')
            z = self.linear(self.norm(x, p+'.layer_norm2', 1e-6, False), p+'.mlp.fc1').gelu()
            x = (x+self.linear(z, p+'.mlp.fc2')).realize()
        x = self.norm(x, prefix+'.post_layernorm', 1e-6, False)
        x = self.norm(x, 'vlm.model.projector.norm', 1e-6, False).reshape(-1, 4608)
        x = self.linear(x, 'vlm.model.projector.linear_fc1').gelu(approximate='none')
        return self.linear(x, 'vlm.model.projector.linear_fc2').realize()

    def transformer(self, x, pos, cache=None, expert=False, attention_mask=None):
        prefix = 'expert.expert' if expert else 'vlm.model.language_model'
        kv = []
        for i in range(28):
            p = f'{prefix}.layers.{i}'
            x,pair=self.layer(x,pos,p,None if cache is None else cache[i],attention_mask,causal=not expert)
            kv.append(pair)
        return self.norm(x, prefix+'.norm').realize(), kv

    def layer(self,x,pos,name,cache=None,mask=None,index=None,causal=False):
        z=self.norm(x,name+'.input_layernorm')
        q,k,v=[self.linear(z,name+'.self_attn.'+n+'_proj').reshape(-1,heads,128).transpose(0,1)
               for n,heads in [('q',16),('k',8),('v',8)]]
        q,k=apply_rope(q,*pos),apply_rope(k,*pos)
        if cache is not None:
            if index is None:k,v=cache[0].cat(k,dim=1),cache[1].cat(v,dim=1)
            else:k,v=self.update_cache(cache[0],k,index),self.update_cache(cache[1],v,index)
        k,v=k.realize(),v.realize()
        z=self.attention(q,k,v,causal=causal,mask=mask).transpose(0,1).reshape(x.shape[0],2048)
        x=x+self.linear(z,name+'.self_attn.o_proj')
        z=self.linear(self.norm(x,name+'.post_attention_layernorm'),name+'.mlp.fc1').relu().square()
        return (x+self.linear(z,name+'.mlp.fc2')).realize(),(k,v)

    def embed_images(self, ids, images):
        embeddings = self.w('vlm.model.language_model.embed_tokens.weight')[Tensor(ids, device=self.device)]
        spans, offset, i = [], 0, 0
        while i < len(ids):
            end = i+1
            image = ids[i] == 19
            while end < len(ids) and (ids[end] == 19) == image: end += 1
            spans.append(images[offset:offset+end-i] if image else embeddings[i:end])
            if image: offset += end-i
            i = end
        return spans[0].cat(*spans[1:], dim=0)

    @Context(TC_OPT=2)
    def flow(self, noise, cache, cursor):
        capacity=cache[0][0].shape[1]
        length=getattr(self,'cache_length',capacity)
        slots=np.arange(capacity+64)
        self.expert_mask=Tensor((slots<length)|(slots>=capacity),device=self.device).realize()
        self.expert_rotary=rotary_tables(np.broadcast_to(np.arange(64)+cursor,(3,64)),self.device,dtypes.bfloat16)
        x = Tensor(noise.reshape(64, 2), device=self.device).float()
        steps=self.euler_steps
        times = np.linspace(0, 1, steps+1, dtype=np.float32)
        for step in range(steps):
            t = Tensor([float(times[step])], device=self.device).realize()
            velocity = self.velocity(x.realize(), t, cache)
            x = (x+float(times[step+1]-times[step])*velocity).realize()
            print(f'flow {step+1}/{steps}', flush=True)
        return x.numpy()

    def compute_velocity(self, x, t, pos, cache, attention_mask=None):
        freq = Tensor(np.logspace(0, 2, 10, dtype=np.float32), device=self.device).cast(dtypes.bfloat16).float()
        def fourier(value):
            a = value.unsqueeze(-1)*freq*2*math.pi
            return a.sin().cat(a.cos(), dim=-1)*math.sqrt(2)
        features = fourier(x[:, 0]).cat(fourier(x[:, 1]), fourier(t.expand(64)), dim=-1).cast(dtypes.bfloat16)
        p = 'expert.action_in_proj.encoder.trunk'
        z = self.linear(features, p+'.0').silu()
        z = self.linear(self.norm(z, p+'.2'), p+'.3').silu()
        z = self.linear(self.norm(z, p+'.5'), p+'.6')
        z = self.norm(z, 'expert.action_in_proj.norm', rms=False)
        z, _ = self.transformer(z, pos, cache, expert=True, attention_mask=attention_mask)
        return self.linear(z, 'expert.action_out_proj').float().realize()


    def pick_token(self,x):
        vocab=Tensor.arange(self.config['vlm_config']['text_config']['vocab_size']).to(self.device)
        allowed=((vocab<131072)|(vocab>=135072)) & (vocab!=11)
        token=allowed.where(self.linear(x,'vlm.lm_head'),-float('inf')).argmax(-1).realize()
        embedding=self.w('vlm.model.language_model.embed_tokens.weight')[token].contiguous().realize()
        return token,embedding

    def linear(self,x,name):
        y=self.matmul(x,self.w(name+'.weight').T)
        return y+self.w(name+'.bias') if name+'.bias' in self.weights else y

    def matmul(self,a,b):
        return a.float().matmul(b.float(),dtype=dtypes.float32).cast(a.dtype)

    def attention(self,q,k,v,causal=False,mask=None):
        if causal:
            nq,nk=q.shape[1],k.shape[1]
            if (nq,nk) not in self.causal_masks:
                self.causal_masks[nq,nk]=Tensor(np.arange(nk)[None,:]<=np.arange(nq)[:,None]+nk-nq,device=self.device).realize()
            mask=self.causal_masks[nq,nk] if mask is None else mask & self.causal_masks[nq,nk]
        if q.shape[0]!=k.shape[0]:
            repeats=q.shape[0]//k.shape[0]
            k,v=k.repeat_interleave(repeats,dim=0),v.repeat_interleave(repeats,dim=0)
        score=self.matmul(q,k.transpose(-1,-2))*q.shape[-1]**-.5
        if mask is not None:score=mask.where(score,-float('inf'))
        return self.matmul(score.float().softmax(-1).cast(q.dtype),v)

    def vision_frame(self,pixels,emb):
        key=pixels.shape
        if key not in self.vision_graphs:self.vision_graphs[key]=TinyJit(self.encode_image)
        # Captured output storage is reused; retained images need their own copies.
        return self.vision_graphs[key](pixels,emb).clone().realize()

    def velocity(self,x,t,cache):
        capacity=cache[0][0].shape[1]
        if capacity not in self.expert_graphs:
            self.expert_graphs[capacity]=TinyJit(lambda a,b,c,s,mask,*kv:
                self.compute_velocity(a,b,(c,s),list(zip(kv[::2],kv[1::2])),mask))
        return self.expert_graphs[capacity](x,t,*self.expert_rotary,self.expert_mask,*[v for pair in cache for v in pair])

    def update_cache(self,cache,value,index):
        update=(Tensor.arange(cache.shape[1]).to(self.device)==index).reshape(1,cache.shape[1],1)
        cache.assign(update.where(value,cache)).realize()
        return cache

    def token_block(self,x,index,step,cos,sin,*kv,count=6):
        tokens=[]
        for offset in range(count):
            token,embedding=self.pick_token(x)
            x=self.decode(embedding,index+offset,cos[step+offset].contiguous(),sin[step+offset].contiguous(),list(zip(kv[::2],kv[1::2])))
            tokens.append(token)
        return x,Tensor.cat(*tokens).realize()

    def prefill(self,x,c,s,capacity):
        result,cache=self.transformer(x,(c,s))
        length=x.shape[0]
        cache=[(k.pad(((0,0),(0,capacity-length),(0,0))).contiguous().realize(),
                v.pad(((0,0),(0,capacity-length),(0,0))).contiguous().realize()) for k,v in cache]
        return result,cache

    def decode(self,x,index,c,s,cache):
        capacity=cache[0][0].shape[1]
        visible=Tensor.arange(capacity).to(self.device)<=index
        prefix='vlm.model.language_model'
        for i in range(28):
            p=f'{prefix}.layers.{i}'
            x,_=self.layer(x,(c,s),p,cache[i],visible,index)
        return self.norm(x,prefix+'.norm').realize()

    @Context(TC_OPT=2)
    def generate(self,ids,pixels,grids,max_tokens=256):
        pos,delta=positions(ids,grids)
        images=self.vision(pixels,grids)
        prefill_capacity=((len(ids)+127)//128)*128
        padding=prefill_capacity-len(ids)
        padded_pos=np.concatenate((pos,np.broadcast_to(np.arange(len(ids),prefill_capacity)+delta,(3,padding))),axis=1)
        c,s=rotary_tables(padded_pos,self.device,dtypes.bfloat16)
        embeddings=self.embed_images(ids,images).pad(((0,padding),(0,0))).contiguous().realize()
        capacity=((len(ids)+max_tokens+255)//256)*256
        key=(prefill_capacity,capacity)
        if key not in self.prefill_graphs:
            self.prefill_graphs[key]=TinyJit(lambda x,c,s,n=capacity:self.prefill(x,c,s,n))
        x,cache=self.prefill_graphs[key](embeddings,c,s)
        x=x[:len(ids)]
        cos,sin=rotary_tables(np.broadcast_to(np.arange(max_tokens)+len(ids)+delta,(3,max_tokens)),self.device,dtypes.bfloat16)
        x=x[-1:].clone().realize()
        generated=[]
        for step in range(0,max_tokens,self.decode_block_size):
            count=min(self.decode_block_size,max_tokens-step)
            key=(capacity,count)
            if key not in self.next_graphs:self.next_graphs[key]=TinyJit(lambda *a,n=count:self.token_block(*a,count=n))
            index=len(ids)+step
            x,tokens=self.next_graphs[key](x,Tensor([index],device=self.device).realize(),Tensor([step],device=self.device).realize(),cos,sin,*[v for pair in cache for v in pair])
            for offset,value in enumerate(tokens.numpy().reshape(-1)):
                token=int(value)
                generated.append(token)
                print(f'token {step+offset}: {token}',flush=True)
                if token==135084:
                    self.cache_length=index+offset+1
                    return generated,cache,index+offset+delta+1
        raise RuntimeError('Trajectory anchor not generated within max_tokens')
