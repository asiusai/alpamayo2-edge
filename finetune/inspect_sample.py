import json, sys, glob, numpy as np
from PIL import Image, ImageDraw
base = sys.argv[1]  # path prefix without -k
out = sys.argv[2]
clean = np.load(base+'-0.npz'); metas = {k: json.load(open(f'{base}-{k}.json')) for k in (0, 1, 2)}
k = max((1, 2), key=lambda k: abs(metas[k]['dy_m'])+20*abs(metas[k]['dpsi_rad']))
rec = np.load(f'{base}-{k}.npz'); m = metas[k]
img = lambda z, i: z['image_frames'][i, 0].transpose(1, 2, 0)
top = np.concatenate([img(clean, 4), img(clean, 9)], 1); bot = np.concatenate([img(rec, 4), img(rec, 9)], 1)
grid = Image.fromarray(np.concatenate([top, bot], 0)).resize((1536//2*1, 864//2*1))
# top-down: x forward (up), y left (left)
plot = Image.new('RGB', (400, 432), (20, 20, 20)); d = ImageDraw.Draw(plot)
def pt(x, y): return (200-y*25, 420-x*1.5)  # 25 px/m lateral, 1.5 px/m forward
c, s = np.cos(m['dpsi_rad']), np.sin(m['dpsi_rad'])
hx, hy = clean['target_xyz'][:, 0], clean['target_xyz'][:, 1]
# human path expressed in the shifted frame
rx, ry = c*hx+s*(hy-m['dy_m']), -s*hx+c*(hy-m['dy_m'])
d.line([pt(x, y) for x, y in zip(rx, ry)], fill=(120, 120, 255), width=2)
d.line([pt(x, y) for x, y in zip(rec['target_xyz'][:, 0], rec['target_xyz'][:, 1])], fill=(255, 200, 0), width=2)
hist = rec['ego_history_xyz'][0]; d.line([pt(x, y) for x, y in zip(hist[:, 0], hist[:, 1])], fill=(200, 200, 200), width=2)
d.text((5, 5), f"dy {m['dy_m']:+.2f} m (left+)  dpsi {np.degrees(m['dpsi_rad']):+.1f} deg", fill=(255, 255, 255))
d.text((5, 20), 'blue: human path  yellow: target  grey: history', fill=(255, 255, 255))
canvas = Image.new('RGB', (grid.width+plot.width, max(grid.height, plot.height)), (0, 0, 0)); canvas.paste(grid, (0, 0)); canvas.paste(plot, (grid.width, 0))
canvas.save(out); print(out, m)
