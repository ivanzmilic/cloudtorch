# Step 6 — running & using the cube inversion (`fit_cube.py`)

Deployment of the step‑1…5 stack: a fitted PCA background ⊕ two absorbing clouds,
fit by a Fourier neural field with explicit regularisation. One configurable,
resumable script. **No new physics** — it reuses `pca_background`, `pinn_model`,
`composite_model`, `regularizers` unchanged. With the baked‑in defaults it
reproduces the tuned step‑5 notebook; the full cube is reached by config alone.

## Environment
- Python + CUDA torch: `/home/milic/miniconda3/envs/pt/bin/python` (torch 2.11, cuda ✓).
- Data path auto‑detects (`data.path` list); add yours there or via `--set`.

## Quick start
```bash
PY=/home/milic/miniconda3/envs/pt/bin/python
cd /home/milic/codes/cloudtorch/develop

$PY fit_cube.py --out run1 --report          # defaults == step-5 notebook (64x64, 16 frames)
$PY fit_cube.py --dump-config cfg.json        # write the full default config to edit
$PY fit_cube.py --config cfg.json --out run2   # run an edited config
```

## Configuring (everything is in the config; three ways to set it)
1. **Edit a file:** `--dump-config cfg.json`, edit, `--config cfg.json`.
2. **Inline overrides:** `--set data.crop=32 train.n_iters=2000 field.width=128`
   (values parse as JSON: `null`, `true`, numbers, `[0,0]`).
3. **Flags:** `--out DIR`, `--device cuda|cpu|auto`.
Precedence: defaults → `--config` → `--set` → flags. The **resolved** config (with
the scaled σ) is written to `DIR/config_used.json`.

Key knobs: `data.crop`/`data.n_t` (`null` = full field / all frames),
`train.batch` (`null` = full batch), `train.n_iters`, and the whole `reg` block
(smoothness `w_xy/w_t`, hinges, `no_emission`, and the `anchor` weight — the main
background lever). The Fourier bandwidths **scale with the domain automatically**
(`field.sigma_xy_ref`/`sigma_t_ref` at `crop_ref`/`n_t_ref`), so the tuning
transfers as you grow the crop/frames — no manual σ retune.

## Outputs (in `--out` dir)
- `params.npz` — `c (Nt,Cy,Cx,6)`, `p_cloud (Nt,Cy,Cx,10)` (order in `cloud_names`,
  incl. frozen `loga`), plus `wl, lam, Iqs, ts, ys, xs, cfg`.
- `checkpoint.pt` — self‑contained (field + optimiser + scheduler + RNG + config +
  PCA basis); written every `io.ckpt_every` iters and at the end.
- `config_used.json`; with `--report`: `loss / obs_vs_fit / emission_hist / spectra` PNGs.
- Console `[metrics]`: RMS (% of mean I) and the background emission diagnostic.

## Resume / evaluate‑only
```bash
$PY fit_cube.py --resume run1/checkpoint.pt                 # continue to n_iters
$PY fit_cube.py --resume run1/checkpoint.pt --no-train --report   # just eval + figures
```
Resume restores everything and continues the cosine schedule to the original
`n_iters`. (To train *longer*, start a fresh run with a larger `train.n_iters`.)

## Scale to the full cube
```bash
$PY fit_cube.py --set data.crop=null data.n_t=null train.batch=8192 \
                train.n_iters=30000 --out full --report
```
`crop=null`/`n_t=null` take the whole field/all frames; σ rescales itself; pick
`train.batch` to fit GPU memory (the regulariser is mesh‑free, so minibatching is
exact). Then **tune `reg`** at that scale (weights are relative to the χ² term).

## Using the results
```python
import numpy as np
d = np.load("run1/params.npz", allow_pickle=True)
vlos1 = d["p_cloud"][..., 2]        # (Nt,Cy,Cx) LOS velocity of cloud 1, km/s
S1    = d["p_cloud"][..., 0]        # source function, dtau1=[...,1], dv1=[...,3]
print(d["cloud_names"], d["c"].shape)   # background PCA coeffs c: (Nt,Cy,Cx,6)
```
`p_cloud` gives the physical cloud parameters directly. To **re‑synthesise**
spectra or the background, the checkpoint is self‑contained: reload it (it carries
the PCA basis + field weights) and call the same forward, or just run
`--resume … --no-train --report`.
