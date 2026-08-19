# GPU-server handoff — steps 1–6 (step 6 deployment DONE)

Self-contained note for picking up `develop/` on the GPU server. General project
status lives in `HANDOFF.md`; this file is the practical "how to run it bigger on
CUDA" guide. Model = fitted PCA background + two absorbing clouds, fit by a
space–time neural field (PINN), per `develop/instructions.md`.

--------------------------------------------------------------------------------
## TL;DR
Steps 1–6 are built. Step 5 (`regularizers.py` + `step_5_reg_test.ipynb`) added explicit
penalties — smoothness + soft bounds + a **no-emission** cap — plus a later **background
anchor** (`background_anchor`, `w·‖c−c_ref‖²` toward the filament-free profile) that fixes a
*second* degeneracy the ≤1 cap misses: the background filling its H-α core up to continuum
while the clouds over-absorb. **Step 6** (`fit_cube.py` + `STEP6_HOWTO.md`) turns the whole
stack into one configurable, resumable **CUDA CLI** that fits the full `(t,y,x,λ)` cube —
validated end-to-end on real data. What's left is science, not code: **scan the REG weights
at full scale** (`reg.anchor.weight` is the key knob).

--------------------------------------------------------------------------------
## Files that matter
- `pca_background.py` — step 1 basis (`fit_background_basis`, `.reconstruct`, `.project`). K=6.
- `background_model.py` — step 2 (`background(pca,c)=mu+c@V`).
- `composite_model.py` — step 3 forward (`composite_synth`, `chi2_per_pixel`). `log a` frozen path used.
- `pinn_model.py` — step 4 field (`ParameterField`, `FourierFeatures`, `grid_coords`). 14 outputs (6 bkg + 8 cloud; `log a` frozen).
- **`regularizers.py`** — step 5. `regularization_loss`, `expand_group_config`, `smoothness_penalty`, `hinge_bounds`, `no_emission`, `DEFAULT_REG`, `FIELD_NAMES/FIELD_GROUPS`.
- **`step_5_reg_test.ipynb`** — the runnable notebook. Cells: config → `REG` config → load+basis → field → grad check → **train (χ²+reg)** → **emission diagnostic** → maps/spectra/residuals → cost extrapolation.
- **`fit_cube.py`** — step 6 deployment CLI (start here for full runs). `STEP6_HOWTO.md` = the run guide.
- Read alongside: `step_5_outline.pdf` + `step_6_outline.pdf` (approved plans), `HANDOFF.md` (full history + step-4 caveat).

--------------------------------------------------------------------------------
## Environment & data
- Env (GPU server): `/home/milic/miniconda3/envs/pt/bin/python` (torch 2.11 + CUDA ✓).
  (The laptop env was `.../envs/ml` torch 2.3.0, CPU.)
- Data: `mihi_all_data.npz` (~5.8 GB, uncompressed → memory-mappable), key `data`
  shape `(t, y, x, lambda)`, key `wav`. The notebook auto-detects the path from
  `_CANDS`; **add the server path** to that list in the config cell:
  ```python
  _CANDS = ['/PATH/ON/SERVER/mihi_all_data.npz',
            '/dat/milic/MiHI_filament/mihi_all_data.npz',
            '/home/milic/data/MiHI_halpha_filament/mihi_all_data.npz']
  ```
- `DEVICE` already resolves to `'cuda' if torch.cuda.is_available() else 'cpu'`.

--------------------------------------------------------------------------------
## Run step 5 as-is (sanity check first)
Open `step_5_reg_test.ipynb`, fix the data path, Run-All. Default config is the
laptop-scale 40×40×4 crop. On CUDA this should be fast. Confirm:
- loss curve: `data` (χ²) descends toward the noise floor; `reg` stays sub-dominant;
- **emission diagnostic**: `background peak / continuum` median ≈ 1.0–1.2 (not ~5);
- `vlos` maps show spatial/temporal structure (not washed out).

--------------------------------------------------------------------------------
## Scale up (config cell knobs)
Enlarge in the first config cell, everything else follows:
- `CROP` (spatial), `N_T` / `T0` (frames) — grow toward the full field/cube.
- `WIDTH`, `DEPTH`, `N_FREQ` — network capacity (step-4 sweet spot: n_freq=128,
  width=256, depth=4, `SIGMA_XY=4`, `SIGMA_T=2`, `S_MAX=1.0`, `DV_FLOOR=8`).
- `N_ITERS`, `LR` (step-4: lr=5e-3, ≥1500 iters, cosine-annealed).
- **`BATCH`**: set to an int to minibatch over coordinates (default `None`=full batch).
  The regulariser is **mesh-free**, so minibatching is fine — the train loop already
  builds a clean grad-tracking leaf per batch:
  `cb = coords[idx].detach().requires_grad_(True)`. Use this for the full cube.

For the **full cube**: normalise `t` over the true time range (see the note in
`grid_coords`), pick `BATCH` to fit GPU memory, and raise `N_ITERS` accordingly.

--------------------------------------------------------------------------------
## Reg weights — CALIBRATE at your scale (most important knob)
`REG` weights are **relative to the χ² data term**, which scales with L (wavelengths)
and batch/reduction. The baked defaults (`no_emission weight=50`, `bkg w_xy=w_t=0.2`)
were calibrated for a **12×12×2 full-batch CPU crop**. When you change crop size, L,
or switch to minibatch, **rescale**:
1. Run once with `REG` all-off (set every weight to 0) → note the `data` χ² magnitude
   and the reg-off background peak (should reproduce the ~5× pathology).
2. Turn on `no_emission` first; raise its `weight` until the emission diagnostic
   median drops to ≈1.0–1.2, watching that `data` χ² rises only a few %.
3. Add `bkg` smoothness (`w_xy=w_t`) at ~0.5–2× the no-emission scale to kill any
   residual high-freq spikes; keep cloud smoothness light (`1e-3`).
4. Cloud hinges (`vlos` ±60, `dv` lo=8, `S/dtau` lo=0) are cheap guardrails — leave on.
The success target is unchanged: **background loses emission while χ² stays near the
noise floor and `vlos` gains structure.** Low RMS alone is NOT success (step-4 lesson).

--------------------------------------------------------------------------------
## GPU-specific pitfalls (verified, but watch)
- **`coords.requires_grad_(True)` is mandatory** — the smoothness penalty differentiates
  the fields w.r.t. coordinates. The notebook sets it; if you rebuild coords, keep it.
- Device flow is clean: `field.to(device)` → `coords` on device → `pca.reconstruct`
  and `cmt.model_synth_2clouds_givenbck` both move their constants onto the param
  device automatically. No manual `.cuda()` on `xl` needed.
- Minor: `cmt` re-creates `ll/ll0` from numpy and copies them host→device **every**
  forward call. Negligible now; if the physics forward becomes the bottleneck at scale,
  hoist those two wavelength tensors to a cached device tensor.
- `loga_clamp=None` is used in the notebook (the field freezes `log a`); keep it.
- `smoothness_penalty` does one extra backward **through the MLP only** (not Voigt),
  looping over the ≤14 active fields — cheap, but if profiling flags it, zero-weight
  the fields you don't smooth (already skipped when weight==0).

--------------------------------------------------------------------------------
## Verification checklist
- [ ] Loss finite throughout; `data` → noise floor; `reg` sub-dominant at convergence.
- [ ] Emission diagnostic: peak/continuum median ≈ 1.0–1.2, few % of pixels >1.05.
- [ ] `vlos1/vlos2` maps physically structured; `S, dtau, dv` within sane ranges.
- [ ] χ² penalty for turning reg on is small (target ≲10 %).

--------------------------------------------------------------------------------
## Step 6 — DONE (`fit_cube.py` + `STEP6_HOWTO.md`)
The notebook is now a script: JSON config + `--set`/`--config` overrides (defaults == the
tuned notebook), `--device`, `--resume` (self-contained checkpoints: field+opt+sched+RNG+
config+PCA), `--report` (diagnostic PNGs). Saves `params.npz` (`c (Nt,Cy,Cx,6)`,
`p_cloud (Nt,Cy,Cx,10)` incl. frozen loga). Fourier σ **auto-scale with the domain**
(`sigmas_for`, σ∝crop/frames) so the tuning transfers to the full field. Full cube:

    PY=/home/milic/miniconda3/envs/pt/bin/python
    $PY fit_cube.py --set data.crop=null data.n_t=null train.batch=8192 \
                    train.n_iters=30000 --out full --report

Validated end-to-end on real data (fresh + resume + eval-only, deterministic, no warnings).
Remaining: tune `REG` at full scale (weights are relative to the χ² term).

The background **anchor** (once an optional lever) is now implemented and on by default
(`reg.anchor.weight=0.1`); `S` was capped via `S_MAX=0.7` rather than switched to softplus.
Relaxing `S` to softplus+hinge stays an untried lever if `S` saturation returns.
