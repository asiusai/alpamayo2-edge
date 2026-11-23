# Input sensitivity: PC prediction on each rebuilt Karel input versus the same input with +-1 LSB pixel noise.
# Also saves the rebuilt inputs as NPZ (for exact-input runs on other runtimes).
exec(open('/home/john/alpamayo2-edge/pc/tools/karel_parity.py').read().split('plans = [Path(p)')[0])
out = PARITY_DIR/'samples'; out.mkdir(exist_ok=True)
plans = [Path(p) for p in json.loads((PARITY_DIR/'plans.json').read_text())['plans']]
rng = np.random.default_rng(0); rows = []
for path in plans:
  plan = json.loads(path.read_text()); sample, _ = build(plan)
  np.savez_compressed(out/f'{path.parent.name}-{path.stem}.npz', **{k: (np.array(v) if k == 'nav_text' else v) for k, v in sample.items()})
  base = rt.predict(sample)
  noisy = dict(sample); noise = rng.integers(-1, 2, sample['image_frames'].shape)
  noisy['image_frames'] = np.clip(sample['image_frames'].astype(np.int16)+noise, 0, 255).astype(np.uint8)
  pert = rt.predict(noisy)
  c = compare(base, pert); rows.append(c)
  print(path.stem, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in c.items()}, flush=True)
ade = np.array([r['ade_m'] for r in rows]); fde = np.array([r['fde_m'] for r in rows]); hd = np.array([r['heading_max_deg'] for r in rows])
print(f"+-1 LSB: ADE median {np.median(ade):.3f} p90 {np.percentile(ade, 90):.3f} max {ade.max():.3f}; FDE median {np.median(fde):.3f} p90 {np.percentile(fde,90):.3f}; "
      f"heading median {np.median(hd):.2f} p90 {np.percentile(hd,90):.2f}; tokens equal {sum(r['tokens_equal'] for r in rows)}/{len(rows)}")
