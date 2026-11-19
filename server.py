"""Serve playback with seekable, browser-compatible camera videos."""
import argparse
import json
import os
import re
import subprocess
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT=Path(__file__).resolve().parent
LOCKS={}


def playback(source):
    target=ROOT/'outputs/playback'/source.relative_to(ROOT/'data')
    with LOCKS.setdefault(source,threading.Lock()):
        route=source.parent/'route.json'
        if target.exists() and target.stat().st_mtime>max(source.stat().st_mtime,route.stat().st_mtime):return target
        camera=next(c for c in json.loads(route.read_text())['cameras'] if c['file']==source.name)
        times=camera['timestamps_us'];valid=camera.get('valid_frames',[True]*len(times))
        # Ignore sub-5ms padding caused by capture-clock jitter, but preserve real gaps.
        drop=[i for i in range(len(times)-1) if not valid[i] and valid[i+1] and times[i+1]-times[i]<5000]
        select='select=not('+ '+'.join(f'eq(n\\,{i})' for i in drop)+'),' if drop else ''
        target.parent.mkdir(parents=True,exist_ok=True)
        temporary=target.with_suffix('.tmp.mp4')
        subprocess.run(['ffmpeg','-v','error','-y','-threads','2','-copyts','-i',str(source),
                        '-vf',select+'scale=768:432','-an','-c:v','libx264','-threads','2','-preset','fast',
                        '-profile:v','baseline','-pix_fmt','yuv420p','-crf','20','-g','20','-bf','0',
                        '-fps_mode','vfr','-enc_time_base','1:1000000','-video_track_timescale','1000000',
                        '-movflags','+faststart',str(temporary)],check=True)
        temporary.replace(target)
    return target


class Handler(SimpleHTTPRequestHandler):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,directory=str(ROOT),**kwargs)

    def translate_path(self,path):
        urlpath=unquote(urlsplit(path).path)
        if urlpath.startswith('/playback/'):
            source=(ROOT/'data'/urlpath.removeprefix('/playback/')).resolve()
            if source.is_relative_to(ROOT/'data') and source.is_file() and source.suffix=='.mp4':return str(playback(source))
        return super().translate_path(path)

    def end_headers(self):
        self.send_header('Accept-Ranges','bytes')
        self.send_header('Cache-Control','no-cache')
        super().end_headers()

    def send_head(self):
        self.remaining=None
        path=Path(self.translate_path(self.path))
        match=re.fullmatch(r'bytes=(\d*)-(\d*)',self.headers.get('Range',''))
        if not match or not any(match.groups()) or not path.is_file():return super().send_head()
        try:source=path.open('rb')
        except OSError:return self.send_error(404)
        size=os.fstat(source.fileno()).st_size
        first,last=match.groups()
        start=int(first) if first else max(0,size-int(last))
        end=min(size-1,int(last)) if first and last else size-1
        if start>end or start>=size:
            source.close()
            self.send_response(416)
            self.send_header('Content-Range',f'bytes */{size}')
            self.send_header('Content-Length','0')
            self.end_headers()
            return None
        self.send_response(206)
        self.send_header('Content-Type',self.guess_type(str(path)))
        self.send_header('Content-Range',f'bytes {start}-{end}/{size}')
        self.remaining=end-start+1
        self.send_header('Content-Length',str(self.remaining))
        self.end_headers()
        source.seek(start)
        return source

    def copyfile(self,source,output):
        try:
            if self.remaining is None:return super().copyfile(source,output)
            while self.remaining:
                data=source.read(min(self.remaining,256*1024))
                if not data:break
                output.write(data)
                self.remaining-=len(data)
        except (BrokenPipeError,ConnectionResetError):pass


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port',type=int,default=8000,help='Listen on 0.0.0.0 (default: 8000)')
    args=parser.parse_args()
    with ThreadingHTTPServer(('0.0.0.0',args.port),Handler) as server:
        print(f'Playback: http://localhost:{args.port}/',flush=True)
        try:server.serve_forever()
        except KeyboardInterrupt:pass
