# develop/ — handoff / current status

H-alpha filament inversion: model = fitted PCA background + two clouds, per
`develop/instructions.md`. Workflow: a half-page `step_N_outline.pdf` is reviewed
before coding each step.

## Where we are (steps 1–4 built)
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

## In progress / next
- Running a **freeze-background** test (background fixed to per-pixel projection, or to a
  single mean profile, vs free) to decide the fix.
- Likely fix: **constrain the background** — freeze at / strongly prior toward the
  filament-free quiet-Sun profile, add a **no-emission bound** (background <= continuum),
  and/or reduce background DOF. The clean reference background is the filament-free
  last-10-frame profile (projecting the *observed* data is itself slightly absorbed).
- Then **step 5**: full cube on GPU — same code, enlarge config, `device=cuda`, minibatch.

## How to run (this machine)
- Env: `/home/milic/miniconda3/envs/ml/bin/python` (torch 2.3.0, CPU only).
- Data: `/home/milic/data/MiHI_halpha_filament/mihi_all_data.npz` (5.8 GB, uncompressed
  → memory-mappable). The physics (Voigt) forward dominates CPU cost (~0.8 s/iter/10k px).
- Note: this repo does NOT contain the Claude chat or memory notes (those live in
  `~/.claude/` on the original machine).
