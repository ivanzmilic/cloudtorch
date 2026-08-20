"""
step_3a_given_test.py -- Step 3a smoke test (see docs/step_3a_outline.pdf).

Exercises the GIVEN-background path end to end on a small real-data crop:
  1. load the fit window (fit_cube.load_data) + the aligned given background
     (given_background.load_given_background);
  2. check I_in aligns with obs (same shape, same rows) and carries NO free params;
  3. forward composite_synth_given + chi2_per_pixel, confirm gradients reach the
     cloud parameters (and only those);
  4. a short unregularised smoke fit -> chi2 drops;
  5. compare against the step-3 FITTED-PCA-background fit on the identical crop.

Run on the GPU server:
    /home/milic/miniconda3/envs/pt/bin/python step_3a_given_test.py
"""

import copy

import torch

from fit_cube import DEFAULT_CFG, deep_merge, load_data, resolve_device, set_seed
from composite_model import (init_cloud_params, combine_cloud_params,
                             composite_synth, composite_synth_given, chi2_per_pixel)
from background_model import init_background_coeffs
from given_background import load_given_background, default_bkg_path


def smoke_fit(step_fn, obs_pt, xl, extra, n_iters=300, lr=5e-2, seed=0):
    """Fit cloud params (+ optional background coeffs) with plain Adam; return the
    (chi2_initial, chi2_final, rms_percent). `step_fn(p_cloud) -> syn`."""
    set_seed(seed)
    device = obs_pt.device
    n_pix = obs_pt.shape[0]
    cloud = init_cloud_params(n_pix, device=device, freeze_loga=True)
    trainable = [p for p in cloud.values() if p.requires_grad] + list(extra)
    opt = torch.optim.Adam(trainable, lr=lr)

    chi0 = None
    for it in range(n_iters):
        opt.zero_grad()
        syn = step_fn(combine_cloud_params(cloud))
        _, loss = chi2_per_pixel(obs_pt, syn, reduce_mean=True)
        if chi0 is None:
            chi0 = float(loss.detach())
        loss.backward()
        opt.step()
    with torch.no_grad():
        syn = step_fn(combine_cloud_params(cloud))
        _, chi = chi2_per_pixel(obs_pt, syn, reduce_mean=True)
        rms = float(((obs_pt - syn) ** 2).mean().sqrt()) * 100.0
    return chi0, float(chi), rms


def main():
    # --- tiny crop, a couple of frames -- runs in seconds on the GPU -----------
    cfg = deep_merge(DEFAULT_CFG, {"data": {"crop": 12, "n_t": 2, "t0": 13}})
    device = resolve_device(cfg)
    print("[cfg] device=%s crop=%s n_t=%s" % (device, cfg["data"]["crop"], cfg["data"]["n_t"]))

    obs_pt, c_ref, pca, xl, meta = load_data(cfg, device)
    print("[data] obs %s | Iqs=%.4f | ts=%s" % (tuple(obs_pt.shape), meta["Iqs"], meta["ts"]))

    # --- load the given background, aligned to the same window -----------------
    print("[bkg ] path=%s" % default_bkg_path(cfg["data"]["path"][0]))
    I_in = load_given_background(cfg, meta, device=device)
    assert I_in.shape == obs_pt.shape, (I_in.shape, obs_pt.shape)
    assert not I_in.requires_grad, "given background must carry NO free parameters"
    print("[bkg ] I_in %s | mean=%.4f | requires_grad=%s"
          % (tuple(I_in.shape), float(I_in.mean()), I_in.requires_grad))

    # --- absolute-level check (the "live trap"): same Iqs level, and the given ---
    # background must sit ABOVE the observed core (the clouds supply the absorption).
    obs_cont, bkg_cont = float(obs_pt[:, -20:].mean()), float(I_in[:, -20:].mean())
    obs_core, bkg_core = float(obs_pt[:, 180:220].mean()), float(I_in[:, 180:220].mean())
    print("[bkg ] red-continuum obs=%.4f I_in=%.4f | core obs=%.4f I_in=%.4f"
          % (obs_cont, bkg_cont, obs_core, bkg_core))
    assert 0.85 < bkg_cont / obs_cont < 1.15, "given background mis-normalised (Iqs?)"
    assert bkg_core > obs_core, "given background must sit above the observed core"

    # --- forward + gradient check (grads reach the clouds, not the background) --
    cloud = init_cloud_params(obs_pt.shape[0], device=device, freeze_loga=True)
    p_cloud = combine_cloud_params(cloud)
    syn = composite_synth_given(xl, I_in, p_cloud)
    assert syn.shape == obs_pt.shape and torch.isfinite(syn).all()
    _, loss = chi2_per_pixel(obs_pt, syn, reduce_mean=True)
    loss.backward()
    g = {k: (None if v.grad is None else float(v.grad.abs().mean())) for k, v in cloud.items()}
    free_ok = all(g[k] is not None and torch.isfinite(torch.tensor(g[k]))
                  for k in cloud if cloud[k].requires_grad)
    frozen_ok = all(cloud[k].grad is None for k in ("c1_loga", "c2_loga"))
    print("[grad] free-cloud grads finite=%s | loga frozen (no grad)=%s" % (free_ok, frozen_ok))
    print("[grad] |dchi2/dp| vlos1=%.3g dtau1=%.3g S1=%.3g" % (g["c1_vlos"], g["c1_tau"], g["c1_S"]))
    assert free_ok and frozen_ok

    # --- smoke fits: given background vs. step-3 fitted PCA background ----------
    g0, gf, grms = smoke_fit(lambda pc: composite_synth_given(xl, I_in, pc),
                             obs_pt, xl, extra=[])
    coeffs = init_background_coeffs(pca, obs_pt)          # step-3 fitted background
    f0, ff, frms = smoke_fit(lambda pc: composite_synth(xl, pca, coeffs, pc),
                             obs_pt, xl, extra=[coeffs])

    print("\n  fit (300 it)      chi2_init -> chi2_final     RMS%%")
    print("  given background  %8.4f  -> %8.4f      %5.2f" % (g0, gf, grms))
    print("  fitted PCA (st3)  %8.4f  -> %8.4f      %5.2f" % (f0, ff, frms))
    assert gf < g0, "given-background fit did not reduce chi2"
    print("\n[PASS] step 3a given-background path works end to end.")


if __name__ == "__main__":
    main()
