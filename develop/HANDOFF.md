# develop/ — handoff / current status

H-alpha filament inversion: model = fitted PCA background + two clouds, per
`develop/instructions.md`. Workflow: a half-page `step_N_outline.pdf` is reviewed
before coding each step.

## Where we are (steps 1–6 built)
- **Step 1** `pca_background.py` — `PCABackground`: global all-pixel PCA of the last
  10 raw frames; **K=6**. `fit_background_basis`, `.reconstruct(c)`, `.project(spec)`.
- **Step 2** `background_model.py` — `init_background_coeffs`, `background(pca,c)=mu+c@V`.
- **Step 3** `composite_model.py` — `composite_synth` (background → two-cloud transfer)
  + `chi2_per_pixel`; per-pixel, strategy-agnostic. `log a` free with a freeze switch.
- **Step 4** `pinn_model.py` + `step_4_pinn_test.ipynb` — MLP + anisotropic Fourier
  neural field `f_theta(x,y,t)` predicting 14 params (6 bkg + 8 cloud; `log a` frozen);
  range transforms (sigmoid S, softplus dtau, softplus+floor dv). Tuned config
  (`sigma_xy=4, sigma_t=2, lr=5e-3, >=1500 iters, S_MAX=1.0`) baked into the notebook,
  which also auto-detects the data path.

- **Step 5** `regularizers.py` + `step_5_reg_test.ipynb` — explicit penalties on the
  step-4 PINN: `L = chi2 + R_smooth + R_bound`. `smoothness_penalty` (autograd
  coordinate derivatives ∂x,∂y,∂t per field — mesh-free, minibatch-safe),
  `hinge_bounds` (soft one-sided quadratic ReLU² bounds, generalises
  `utils.regu_min/max`), `no_emission` (upper hinge on the background *spectrum*,
  caps it at continuum ≈1). Grouped `REG` config (`bkg/S/dtau/vlos/dv/no_emission`)
  → `expand_group_config` → per-field (P=14) weight/bound vectors; the field's
  frozen `loga` columns are excluded. `regularization_loss(c,p_cloud,coords,pca,...,c_ref=)`
  returns `{smooth,bound,no_emission,anchor,total}`. **Requires `coords.requires_grad_(True)`.**
  Later addition: `background_anchor` (`w·‖c−c_ref‖²`) pulls the background toward the
  filament-free reference — the fix for the *core-filling* degeneracy the ≤1 cap can't see
  (background fills its H-α core up to continuum while the clouds over-absorb). Notebook now
  tuned at 64×64×16: domain-scaled Fourier σ (σ∝crop/frames), `bkg w=0.2`, `S_MAX=0.7`,
  `anchor weight=0.1` (key knob), 5k iters, batch 4096.

- **Step 6** `fit_cube.py` + `STEP6_HOWTO.md` — deployment: the step-1–5 stack as one
  configurable, resumable CLI fitting the full `(t,y,x,λ)` cube on GPU. JSON config +
  `--set`/`--config` overrides (defaults == the tuned notebook), `--resume`, self-contained
  checkpoints (field+opt+sched+RNG+config+PCA), `params.npz` output (`c`,`p_cloud`), optional
  `--report`. Fourier σ auto-scale with the domain (`sigmas_for`). Validated end-to-end on
  real data (fresh + resume + eval-only, deterministic).

## STEP 5 RESULT — the degeneracy is broken
Calibrated on a 12×12×2 real-data crop (full batch, 200 iters): **REG off** reproduces
the step-4 pathology (background peak **~4.9× / 5.8×** continuum, median/max);
**REG on** (`no_emission weight=50`, `bkg w_xy=w_t=0.2`) collapses it to **~1.1× / 1.2×**
while the data χ² rises only ~6% (1.94→2.05) — still at the noise floor. So the
emission-spike degeneracy is suppressed at negligible fit cost. Weights are calibrated
to this data scale; **rescale with the data-term magnitude** if crop size / batching
changes. Update: the notebook now runs 64×64×16 with domain-scaled σ + the background
anchor (step 6). A *second* degeneracy was then found — the background filling its core UP
to continuum (≤1, so invisible to the no-emission cap) while clouds over-absorb — and fixed
with `background_anchor`. REG weights still need scanning at full scale.

## KEY FINDING (read before trusting step_4_results.pdf)
On the real data the **per-pixel intensity fit is at the photon-noise floor**
(median RMS ~= noise, 96% of pixels within 1.5x noise). BUT the **background/cloud
decomposition is degenerate and unphysical**: the fitted background develops large
**emission peaks** (up to ~8x continuum on off-filament pixels) that the clouds then
absorb. Low RMS != good physics. `step_4_results.pdf` over-states quality; treat it
together with this caveat.

Contributing cause: the background (6 free PCA coeffs/pixel) is unconstrained, and on
small crops the network is over-parameterised (~267k weights vs N_pix*14, e.g. 22k for
a 40x40 frame — 12x). Fourier smoothness limits spatial frequency, not physicality.

## Freeze-background test (done) — naive freezing is NOT the fix
Single frame, same config: free bkg RMS **2.35%**; frozen at per-pixel projection
**7.97%**; frozen at a single mean profile **8.37%** (cloud v runs to +/-200 km/s, dv
to 330 — it blows up). Freezing at `pca.project(obs)` double-counts the absorption
(that reference already contains the line), so the clouds over-absorb; a single mean
can't track spatial variation. The background is genuinely load-bearing.

## Correct fix — DONE (background anchor)
Anchor the background to the **true filament-free quiet-Sun** profile — NOT the observed
projection — and forbid emission:
- reference = last-10-frame (filament-free) background per pixel, or skglm
  `bdata = new_fit - new_extrafit` (in `mihi_rvm_reconstruction_averages_skglm.npz`);
- add a **no-emission bound** (background <= continuum ~ 1) so it can't grow peaks the
  clouds cancel;
- keep the background flexible but with a prior pulling it toward that reference.
**Implemented** as `regularizers.py::background_anchor`, wired into the notebook and
`fit_cube.py` (step 6). Reference = the filament-free last-10-frame mean per pixel, projected
(`c_ref`); the no-emission cap stays on alongside it. Full-cube GPU run is now
`fit_cube.py --set data.crop=null data.n_t=null` (see `STEP6_HOWTO.md`).

## How to run
- **GPU server (current):** `/home/milic/miniconda3/envs/pt/bin/python` (torch 2.11 + CUDA);
  data `/dat/milic/MiHI_filament/mihi_all_data.npz`. Step 6: `fit_cube.py` (see
  `STEP6_HOWTO.md`). The notebook `step_5_reg_test.ipynb` also runs here.
- **Laptop (original):** `/home/milic/miniconda3/envs/ml/bin/python` (torch 2.3.0, CPU only);
  data `/home/milic/data/MiHI_halpha_filament/mihi_all_data.npz`. Voigt forward dominates
  CPU cost (~0.8 s/iter/10k px).
- Data: 5.8 GB, uncompressed → memory-mappable; key `data` (t,y,x,λ), key `wav`.
- Note: this repo does NOT contain the Claude chat or memory notes (those live in
  `~/.claude/` on the original machine).
