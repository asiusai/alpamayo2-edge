"""Run camera predictions, compare with stock references, and browse index.html."""
import argparse
import importlib
import hashlib
import json
import math
import os
import subprocess
import time
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
from PIL import Image
from tinygrad import Device
from tokenizers import Tokenizer
from huggingface_hub import hf_hub_download
from model import Model

NAMES = {0: 'Front left camera', 1: 'Front camera', 2: 'Front right camera',
         3: 'Rear left camera', 5: 'Rear right camera', 6: 'Front telephoto camera'}


@lru_cache(maxsize=4)
def tokenizer(checkpoint):
    return Tokenizer.from_file(str(Path(checkpoint)/'tokenizer.json'))


class FrameCache:
    """Decode selected route frames once; keep the original Pillow resize math."""
    def __init__(self,directory,manifest,predictions):
        self.frames={}
        root=Path(directory)
        def camera_frames(camera):
            key=str(camera['index'])
            needed={stamp for p in predictions for stamp in p['camera_timestamps_us'][key]}
            indices=[i for i,stamp in enumerate(camera['timestamps_us']) if stamp in needed]
            # Compact adjacent indices so FFmpeg does not evaluate thousands of terms.
            ranges=[]
            for i in indices:
                if ranges and i==ranges[-1][1]+1:ranges[-1][1]=i
                else:ranges.append([i,i])
            expression='+'.join(f'between(n\\,{a}\\,{b})' for a,b in ranges)
            command=['ffmpeg','-v','error','-threads','2','-i',str(root/camera['file']),'-map','0:v:0',
                     '-vf','select='+expression,'-fps_mode','passthrough','-frames:v',str(len(indices)),
                     '-f','rawvideo','-pix_fmt','rgb24','-threads','1','-']
            process=subprocess.Popen(command,stdout=subprocess.PIPE)
            size=camera['height']*camera['width']*3
            h,w=resize_shape(camera['height'],camera['width'],*camera['pixel_bounds'])
            result={}
            try:
                for i in indices:
                    raw=process.stdout.read(size)
                    if len(raw)!=size:raise ValueError(f'Truncated camera video: {camera["file"]}')
                    image=Image.frombytes('RGB',(camera['width'],camera['height']),raw)
                    result[i]=np.asarray(image.resize((w,h),Image.Resampling.BICUBIC))
                if process.wait()!=0:raise RuntimeError(f'FFmpeg failed: {camera["file"]}')
            finally:
                process.stdout.close()
                if process.poll() is None:process.kill();process.wait()
            return camera['index'],result
        with ThreadPoolExecutor(max_workers=3) as pool:
            for key,frames in pool.map(camera_frames,manifest['cameras']):self.frames[key]=frames


def decode(path, width, height, count, indices=None):
    select = [] if indices is None else ['-vf', 'select='+ '+'.join(f'eq(n\\,{i})' for i in indices), '-fps_mode', 'passthrough']
    result = subprocess.run(['ffmpeg', '-v', 'error', '-i', str(path), '-map', '0:v:0', *select,
                             '-frames:v', str(count), '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-threads', '1', '-'],
                            check=True, stdout=subprocess.PIPE)
    frames = np.frombuffer(result.stdout, np.uint8)
    if frames.size != count*height*width*3:
        raise ValueError(f'{path}: video frames do not match the manifest')
    return frames.reshape(count, height, width, 3)


def resize_shape(h, w, lower, upper):
    rh, rw = round(h/32)*32, round(w/32)*32
    if rh*rw > upper:
        scale = math.sqrt(h*w/upper)
        rh, rw = math.floor(h/scale/32)*32, math.floor(w/scale/32)*32
    elif rh*rw < lower:
        scale = math.sqrt(lower/(h*w))
        rh, rw = math.ceil(h*scale/32)*32, math.ceil(w*scale/32)*32
    return rh, rw


def prepare(directory, checkpoint, manifest, prediction, frame_cache=None):
    root = Path(directory)
    context = prediction
    cameras = sorted(manifest['cameras'], key=lambda c: c['index'])
    if [c['index'] for c in cameras] not in (sorted(NAMES),[1,6]):
        raise ValueError('Demo must contain six cameras or the two front cameras (1, 6)')
    selected = {}
    for camera in cameras:
        timestamps = np.asarray(camera['timestamps_us'])
        requested = np.asarray(prediction['camera_timestamps_us'][str(camera['index'])])
        indices = np.searchsorted(timestamps, requested, side='right')-1
        if np.any(indices < 0) or not np.array_equal(timestamps[indices], requested):
            raise ValueError('Prediction capture timestamps are absent from the video manifest')
        if 'valid_frames' in camera and not np.asarray(camera['valid_frames'])[indices].all():
            raise ValueError('Prediction history includes a missing-frame placeholder')
        selected[camera['index']] = (indices, timestamps[indices])
    last_timestamp = selected[cameras[-1]['index']][1][-1]
    prompt = '\n<|im_start|>system\nYou are a driving assistant that generates safe and accurate actions.<|im_end|>\n<|im_start|>user\n'
    pixels, grids = [], []
    for camera in cameras:
        index = camera['index']
        indices, timestamps = selected[index]
        frames = ([frame_cache.frames[index][int(i)] for i in indices] if frame_cache else
                  decode(root/camera['file'], camera['width'], camera['height'], len(indices), indices))
        prompt += NAMES[index]+': '
        for frame, timestamp in zip(frames, timestamps, strict=True):
            if frame_cache:
                h,w=frame.shape[:2]
            else:
                h, w = resize_shape(*frame.shape[:2], *camera['pixel_bounds'])
                frame = np.asarray(Image.fromarray(frame).resize((w, h), Image.Resampling.BICUBIC))
            normalized = (frame.astype(np.float32)/255-.5)/.5
            gh, gw = h//16, w//16
            patches = normalized.reshape(gh//2, 2, 16, gw//2, 2, 16, 3).transpose(0, 3, 1, 4, 2, 5, 6).reshape(gh*gw, 768)
            pixels.append(patches)
            grids.append([1, gh, gw])
            prompt += f'<t={(timestamp-last_timestamp)/1e6:z.1f}s><|vision_start|>'+ '<|image_pad|>'*(gh*gw//4)+'<|vision_end|>'
    prompt += '\n<|traj_history_start|>'+'<|traj_history|>'*57+'<|traj_history_end|>\n'
    prompt += 'Navigation input is: '+context['navigation']+'\n'
    prompt += 'Your task: output the chain-of-thought reasoning of the driving process, then output the future trajectory.<|im_end|>\n<|im_start|>assistant\n<think></think>'
    ids = tokenizer(str(checkpoint)).encode(prompt, add_special_tokens=False).ids
    history = np.asarray(context['ego_history_xyz'], dtype=np.float32).reshape(20, 3)
    limits = np.array([4, 4, 10], np.float32)
    history_ids = np.clip(np.rint((np.diff(history, axis=0)+limits)/(2*limits)*999), 0, 999).astype(np.int32).reshape(-1)+131072
    slots = [i for i, token in enumerate(ids) if token == 135087]
    if len(slots) != 57:
        raise ValueError(f'Expected 57 history placeholders, got {len(slots)}')
    for i, token in zip(slots, history_ids, strict=True): ids[i] = int(token)
    return ids, np.concatenate(pixels), np.asarray(grids, np.int64), np.asarray(manifest['initial_noise'], np.float32).reshape(1, 64, 2)


def ensure_checkpoint(directory):
    """Download missing files from the pinned checkpoint; reuse local files offline."""
    directory=Path(directory).expanduser()
    directory.mkdir(parents=True,exist_ok=True)
    def fetch(name):
        path=directory/name
        if not path.is_file() or path.stat().st_size==0:
            print(f'Downloading nvidia/Alpamayo2-Edge/{name}',flush=True)
            hf_hub_download('nvidia/Alpamayo2-Edge',name,
                            revision='b92b6aa326bd3e2235db28f248ad5a30b5890e64',local_dir=directory)
    for name in ('config.json','tokenizer.json','model.safetensors.index.json','LICENSE'):fetch(name)
    index=json.loads((directory/'model.safetensors.index.json').read_text())
    for name in sorted(set(index['weight_map'].values())):
        if Path(name).name!=name:raise ValueError('Checkpoint index contains an invalid shard filename')
        fetch(name)
    return directory


def checkpoint_digest(directory):
    index=json.loads((directory/'model.safetensors.index.json').read_text())
    names=['config.json','tokenizer.json','model.safetensors.index.json',*sorted(set(index['weight_map'].values()))]
    digest=hashlib.sha256()
    for name in names:
        digest.update(name.encode())
        with (directory/name).open('rb') as source:
            for block in iter(lambda:source.read(8*1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def input_digest(ids,pixels,grids,noise,prediction):
    digest=hashlib.sha256()
    for value in (np.asarray(ids,np.int64),pixels,grids,noise,
                  np.asarray(prediction['ego_history_xyz'],np.float32),np.asarray(prediction['ego_history_rot'],np.float32)):
        value=np.ascontiguousarray(value)
        digest.update(str((value.shape,value.dtype.str)).encode());digest.update(value.tobytes())
    return digest.hexdigest()


def save_json(path,value):
    temporary=path.with_name(path.name+f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    temporary.replace(path)


def compare_prediction(result,reference):
    action_delta=np.asarray(result['actions'])-np.asarray(reference['actions'])
    distance=np.linalg.norm(np.asarray(result['xyz'])-np.asarray(reference['xyz']),axis=-1)
    return {'action_rmse':float(np.sqrt(np.mean(action_delta**2))),
            'action_max_error':float(np.abs(action_delta).max()),'ade_m':float(distance.mean()),
            'fde_m':float(distance[-1]),'max_error_m':float(distance.max()),'tokens_match':result['tokens']==reference['tokens']}


def index_outputs():
    """Publish a small run catalog for the static playback app."""
    root=Path(__file__).resolve().parent
    runs=[]
    for path in sorted((root/'outputs').rglob('setup.json')):
        setup=json.loads(path.read_text())
        directory=path.parent.relative_to(root).as_posix()
        summary_path=path.parent/'comparison.json'
        summary=json.loads(summary_path.read_text()) if summary_path.exists() else {}
        runs.append({'id':path.parent.relative_to(root/'outputs').as_posix(),'directory':directory,
                     'demo':setup.get('demo'),'model':setup.get('model'),'setup':setup,'summary':summary,
                     'updated':summary_path.stat().st_mtime if summary_path.exists() else path.stat().st_mtime})
    save_json(root/'outputs/index.json',{'runs':sorted(runs,key=lambda r:r['updated'],reverse=True)})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,default=Path(__file__).resolve().parent/'checkpoints/model',help='Weights directory; downloads missing weights (default: checkpoints/model)')
    parser.add_argument('--demo',type=Path,default=Path('data/demo1'),help='data/demo1: 6 cameras, 20s (default); demo2: 6 cameras, 252s; demo3: 2 cameras, 252s')
    parser.add_argument('--output',type=Path,help='Defaults to outputs/<demo>/<model>')
    parser.add_argument('--every',type=float,default=1.,help='Seconds between predictions (default: 1); 0.25 selects every available window')
    parser.add_argument('--prepare-only',action='store_true',help='Check video preparation without inference')
    parser.add_argument('--save-reference',action='store_true',help='Save/resume stock ten-step predictions in the demo folder')
    parser.add_argument('--model',default='model',help='Default: model (stock tinygrad). Faster: models.gfx1100. Extend either; select module or module:Class')
    parser.add_argument('--limit',type=int,help='Stop after this many advancing predictions')
    parser.add_argument('--steps',type=int,choices=range(1,11),default=10,help='Euler steps (default: 10); subclasses can use 3 for speed at changed accuracy')
    args=parser.parse_args()
    if args.save_reference and (args.prepare_only or args.steps!=10):parser.error('--save-reference requires inference with ten steps')
    if args.limit is not None and args.limit<1:parser.error('--limit must be positive')
    module,separator,name=args.model.partition(':')
    name=name if separator else 'Model'
    if not module or not name:parser.error('--model must be a module or module:Class')
    if args.output is None:args.output=Path('outputs')/args.demo.name/(module.rsplit('.',1)[-1]+('' if name=='Model' else '-'+name))
    if args.every<.25: parser.error('--every must be at least 0.25 seconds')
    manifest=json.loads((args.demo/'route.json').read_text())
    if not manifest['predictions']: parser.error('Demo has no valid prediction windows')
    args.checkpoint=ensure_checkpoint(args.checkpoint)
    anchors=[]
    for p in manifest['predictions']:
        if args.every==.25 or not anchors or p['time_us']-anchors[-1]['time_us']>=args.every*1e6-1000: anchors.append(p)
    if anchors[-1]!=manifest['predictions'][-1]: anchors.append(manifest['predictions'][-1])
    if args.limit is not None:anchors=anchors[:args.limit]
    args.output.mkdir(parents=True,exist_ok=True)
    model=None
    if not args.prepare_only:
        model_class=getattr(importlib.import_module(module),name)
        if not isinstance(model_class,type) or not issubclass(model_class,Model):parser.error('--model must extend model.Model')
        model=model_class(args.checkpoint,Device.DEFAULT)
        if args.save_reference and type(model) is not Model:
            parser.error('--save-reference requires the default model.Model class')
        if type(model) is Model and args.steps!=10:
            parser.error('The stock model uses ten Euler steps; select a subclass to test fewer steps')
    if model is not None:model.euler_steps=args.steps
    reference_path=args.demo/'reference.json'
    reference=json.loads(reference_path.read_text()) if reference_path.exists() else None
    metadata=None
    if not args.prepare_only:
        metadata={'checkpoint_sha256':checkpoint_digest(args.checkpoint),
                  'route_sha256':hashlib.sha256((args.demo/'route.json').read_bytes()).hexdigest()}
        if reference:
            for key,value in metadata.items():
                if reference['metadata'].get(key)!=value:raise ValueError(f'Stock reference {key} differs; use matching checkpoint and demo inputs')
        if args.save_reference:
            metadata.update(implementation='tinygrad stock',
                            source_sha256=hashlib.sha256(Path(importlib.import_module(Model.__module__).__file__).read_bytes()).hexdigest(),
                            weights='original BF16',accumulation='FP32',steps=10,text_sampling='greedy',
                            decode_block_size=1,inputs='same prepared tensors and initial noise as the candidate')
            if reference and reference['metadata']!=metadata:raise ValueError('Saved stock reference settings differ')
            if reference is None:reference={'metadata':metadata,'predictions':[]}
    references={} if reference is None else {p['time_us']:p for p in reference['predictions']}
    if args.save_reference:anchors=[p for p in anchors if p['time_us'] not in references]
    if not anchors:
        print('All selected reference predictions already exist',flush=True)
        return
    started=time.monotonic()
    frame_cache=FrameCache(args.demo,manifest,anchors)
    setup={'video_decode_s':time.monotonic()-started,'euler_steps':args.steps,'camera_count':len(manifest['cameras'])}
    setup['model']=None if model is None else args.model
    setup['device']=Device.DEFAULT
    setup['demo']=args.demo.resolve().relative_to(Path(__file__).resolve().parent).as_posix()
    setup['stock_reference']=bool(args.save_reference)
    if metadata:setup.update({key:metadata[key] for key in ('checkpoint_sha256','route_sha256')})
    setup['every']=args.every
    if model is not None:
        sources={Path(importlib.import_module(cls.__module__).__file__) for cls in type(model).__mro__ if cls is not object}
        setup['source_sha256']=hashlib.sha256(b''.join(path.read_bytes() for path in sorted(sources))).hexdigest()
    save_json(args.output/'setup.json',setup)
    print(f'Video preparation: {setup["video_decode_s"]:.2f}s',flush=True)
    tokenizer=Tokenizer.from_file(str(args.checkpoint/'tokenizer.json'))
    results=[dict(r) for r in reference['predictions']] if args.save_reference else []
    if args.save_reference:
        by_time={p['time_us']:p for p in manifest['predictions']}
        for result in results:result['recorded_xyz']=by_time[result['time_us']].get('reference_xyz')
    save_json(args.output/'predictions.json',results)
    save_json(args.output/'comparison.json',{'predictions':len(results),'complete':False})
    index_outputs()
    for i,p in enumerate(anchors):
        started=time.monotonic()
        ids,pixels,grids,noise=prepare(args.demo,args.checkpoint,manifest,p,frame_cache)
        if args.prepare_only:
            print(f'Prepared {i+1}/{len(anchors)} at {p["time_us"]/1e6:.2f}s: {pixels.shape}',flush=True)
            continue
        fingerprint=input_digest(ids,pixels,grids,noise,p)
        stock=references.get(p['time_us'])
        if stock and stock['input_sha256']!=fingerprint:raise ValueError('Stock reference input tensors differ from this run')
        result=model.predict(ids,pixels,grids,noise,p)
        if not np.isfinite(result['actions']).all() or not np.isfinite(result['xyz']).all():raise RuntimeError('Non-finite model output')
        result.update(time_us=p['time_us'],seconds=time.monotonic()-started,input_sha256=fingerprint,warmup=i<2,
                      text=tokenizer.decode(result['tokens'],skip_special_tokens=True))
        if args.save_reference:
            reference['predictions'].append(dict(result))
            reference['predictions'].sort(key=lambda r:r['time_us'])
            reference['coverage']={'saved':len(reference['predictions']),'available':len(manifest['predictions'])}
            save_json(reference_path,reference)
        elif stock:
            result['comparison']=compare_prediction(result,stock)
            result['stock']={key:stock[key] for key in ('xyz','actions','text')}
        else:
            result['comparison']=None
        result['recorded_xyz']=p.get('reference_xyz')
        results.append(result)
        results.sort(key=lambda r:r['time_us'])
        save_json(args.output/'predictions.json',results)
        compared=[r['comparison'] for r in results if r.get('comparison')]
        summary={'predictions':len(results),'compared':len(compared),'missing_reference':0 if args.save_reference else len(results)-len(compared),
                 'median_seconds':float(np.median([r['seconds'] for r in results]))}
        warm=[r['seconds'] for j,r in enumerate(results) if not r.get('warmup',j<2)]
        if warm:
            summary.update(warm_predictions=len(warm),warm_median_seconds=float(np.median(warm)),warm_p95_seconds=float(np.percentile(warm,95)))
        if compared:
            summary.update(ade_m=float(np.mean([r['ade_m'] for r in compared])),
                           fde_m=float(np.mean([r['fde_m'] for r in compared])),
                           action_rmse=float(np.sqrt(np.mean([r['action_rmse']**2 for r in compared]))),
                           token_match_rate=float(np.mean([r['tokens_match'] for r in compared])))
        save_json(args.output/'comparison.json',summary)
        index_outputs()
        print(f'Prediction {i+1}/{len(anchors)} at {p["time_us"]/1e6:.2f}s: {result["seconds"]:.3f}s',flush=True)
    if not args.prepare_only:
        if 'warm_median_seconds' in summary:print(f'Warm latency: median {summary["warm_median_seconds"]:.3f}s, p95 {summary["warm_p95_seconds"]:.3f}s (first two excluded)',flush=True)
        if compared:print(f'Stock comparison: {len(compared)}/{len(results)} windows, ADE {summary["ade_m"]:.3f}m, FDE {summary["fde_m"]:.3f}m, action RMSE {summary["action_rmse"]:.5f}',flush=True)
        elif not args.save_reference:print('No matching stock reference predictions yet',flush=True)
        with (args.output/'history.jsonl').open('a') as history:history.write(json.dumps({'setup':setup,**summary})+'\n')
        summary['complete']=True
        save_json(args.output/'comparison.json',summary)
        index_outputs()
    print(f'Finished {len(anchors)} '+('input windows' if args.prepare_only else 'predictions. Open index.html'),flush=True)


if __name__=='__main__': main()
