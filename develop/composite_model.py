"""
composite_model.py -- Steps 3 & 3a of the develop/ plan (see instructions.md).

The composite forward model: two absorbing clouds stacked on a background, as a
single autograd-differentiable per-pixel model,

        I_in  = mu + c @ V                         (step-2 fitted PCA background)
        I_out = cloud_2( cloud_1( I_in ) )         (existing two-cloud transfer)

The background is a user-selectable MODE with two sibling entry points sharing the
same physics tail (`_compose`):

  * step 3  -- `composite_synth(x, pca, coeffs, p_cloud)`  : I_in fitted from the
    PCA basis (6 free coefficients per pixel);
  * step 3a -- `composite_synth_given(x, I_in, p_cloud)`   : I_in SUPPLIED as a
    fixed, pre-calculated tensor (the skglm quiet-Sun profile
    bdata = new_fit - new_extrafit), 0 background free parameters.

Because the given-background path fixes I_in, the background/cloud degeneracy that
steps 5-6 fought (emission spikes / core-filling the clouds cancel) cannot arise
there: only the cloud parameters are free.

The atomic unit is ONE pixel -> ONE spectrum; everything is simply batched over an
arbitrary set of pixels (rows), with NO coupling between them. That per-pixel
independence is deliberate: this layer commits to no inversion strategy and adds
no regularisation. A classical spatially-regularised fit or a PINN can wrap the
identical model + loss later -- only the parameter source and the regulariser
differ.

Parameters per pixel: 6 background coefficients c + 10 cloud numbers
[S, dtau, vlos, dv, log a] x 2 (= 16). log a is a FREE parameter here, with a
per-quantity freeze switch retained (make_param(..., grad=)) and an optional
clamp to a safe range, because free damping historically drove the Voigt profile
to NaNs.

Reuses cloud_model_torch (parent folder) and background_model (this folder); it
re-derives nothing.
"""

import os
import sys

import torch
import torch.nn as nn

# cloud_model_torch.py lives one level up (the repo root); make it importable
# whether or not the notebook has already put the parent on sys.path.
_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

import cloud_model_torch as cmt          # noqa: E402  (needs the sys.path tweak above)
from background_model import background   # noqa: E402

__all__ = [
    "make_param", "init_cloud_params", "combine_cloud_params",
    "composite_synth", "composite_synth_given", "chi2_per_pixel",
]

# Layout of the stacked cloud-parameter tensor expected by
# cloud_model_torch.model_synth_2clouds_givenbck:
#   [S1, dtau1, vlos1, dv1, loga1, S2, dtau2, vlos2, dv2, loga2]
_CLOUD_KEYS = ["c1_S", "c1_tau", "c1_vlos", "c1_dv", "c1_loga",
               "c2_S", "c2_tau", "c2_vlos", "c2_dv", "c2_loga"]
_LOGA_COLS = (4, 9)

# Defaults mirror the existing main_gpu notebook.
_CLOUD_INIT = {
    "c1_S": 0.3, "c1_tau": 4.5, "c1_vlos": -20.0, "c1_dv": 12.0, "c1_loga": -4.0,
    "c2_S": 0.3, "c2_tau": 4.5, "c2_vlos":  20.0, "c2_dv": 12.0, "c2_loga": -4.0,
}


# ====================================================================
def make_param(init_value, shape, grad=True, device=None, dtype=torch.float32):
    """A filled nn.Parameter of the given shape (the notebook make_param idiom).
    `grad=False` freezes the quantity (requires_grad off) -- this is the freeze
    switch used for log a."""
    p = torch.full(tuple(shape), float(init_value), dtype=dtype, device=device)
    return nn.Parameter(p, requires_grad=grad)


# ====================================================================
def init_cloud_params(n_pix, device=None, freeze_loga=False, init=None):
    """Build the 10 per-pixel cloud quantities as nn.Parameters.

    log a is free by default (freeze_loga=False); set freeze_loga=True to hold
    both damping parameters fixed (the old behaviour). Returns an ordered dict
    {name: nn.Parameter}; stack it with combine_cloud_params.
    """
    vals = dict(_CLOUD_INIT)
    if init:
        vals.update(init)
    grads = {k: True for k in _CLOUD_KEYS}
    grads["c1_loga"] = grads["c2_loga"] = not freeze_loga
    return {k: make_param(vals[k], (n_pix,), grad=grads[k], device=device)
            for k in _CLOUD_KEYS}


# ====================================================================
def combine_cloud_params(params):
    """Stack the cloud-parameter dict into the (n_pix, 10) tensor the two-cloud
    routine expects, in the canonical column order."""
    return torch.stack([params[k] for k in _CLOUD_KEYS], dim=1)


# ====================================================================
def _clamp_loga(p_cloud, loga_clamp):
    """Return p_cloud with the two log a columns clamped to [lo, hi]; autograd
    -friendly (no in-place ops). Pass loga_clamp=None to disable."""
    if loga_clamp is None:
        return p_cloud
    lo, hi = loga_clamp
    cols = list(torch.unbind(p_cloud, dim=1))
    for j in _LOGA_COLS:
        cols[j] = cols[j].clamp(lo, hi)
    return torch.stack(cols, dim=1)


# ====================================================================
def _compose(x, I_in, p_cloud, loga_clamp):
    """Shared physics tail of both background modes: clamp log a, then push the
    incident intensity I_in (n_pix, L) through the two-cloud transfer.

    I_in is treated as-is -- whatever produced it (fitted PCA background or a
    supplied fixed tensor). Differentiable w.r.t. p_cloud, and w.r.t. I_in when
    I_in carries a graph (it does not in the given-background mode)."""
    p_cloud = _clamp_loga(p_cloud, loga_clamp)
    return cmt.model_synth_2clouds_givenbck(x, p_cloud, I_in)


# ====================================================================
def composite_synth(x, pca, coeffs, p_cloud, loga_clamp=(-6.0, 0.0)):
    """Step 3 -- composite forward model: two clouds on the FITTED PCA background.

    Parameters
    ----------
    x : [wavelength_array, line_center_array]
        As cloud_model_torch expects (numpy arrays; it casts to float32).
    pca : PCABackground
        The fixed step-1 basis.
    coeffs : (n_pix, K) tensor
        Background coefficients c (typically an nn.Parameter).
    p_cloud : (n_pix, 10) tensor
        Stacked cloud parameters (combine_cloud_params).
    loga_clamp : (lo, hi) or None
        Clamp on the two log a columns for stability (default [-6, 0]).

    Returns
    -------
    (n_pix, L) synthetic spectra. Differentiable w.r.t. BOTH coeffs and p_cloud.
    """
    I_in = background(pca, coeffs)                       # (n_pix, L)
    return _compose(x, I_in, p_cloud, loga_clamp)


# ====================================================================
def composite_synth_given(x, I_in, p_cloud, loga_clamp=(-6.0, 0.0)):
    """Step 3a -- composite forward model on a GIVEN (pre-calculated) background.

    Identical physics to composite_synth, but the incident intensity I_in is
    SUPPLIED directly as a fixed tensor rather than reconstructed from the PCA
    basis -- there are NO background free parameters. Use it with the
    pre-calculated skglm quiet-Sun profile (bdata = new_fit - new_extrafit; see
    given_background.load_given_background), which must already share the fit
    window's lambda-crop and Iqs normalisation with the observed data.

    Parameters
    ----------
    x : [wavelength_array, line_center_array]
        As cloud_model_torch expects.
    I_in : (n_pix, L) tensor
        The fixed incident background, one spectrum per pixel. Typically a plain
        (grad-free) tensor; the fit only moves p_cloud.
    p_cloud : (n_pix, 10) tensor
        Stacked cloud parameters (combine_cloud_params).
    loga_clamp : (lo, hi) or None
        Clamp on the two log a columns for stability (default [-6, 0]).

    Returns
    -------
    (n_pix, L) synthetic spectra. Differentiable w.r.t. p_cloud.
    """
    return _compose(x, I_in, p_cloud, loga_clamp)


# ====================================================================
def chi2_per_pixel(obs, syn, weight=None, reduce_mean=False):
    """Per-pixel chi-square = sum over wavelength of squared residuals.

    obs, syn : (n_pix, L) tensors.
    weight   : optional (L,) or (n_pix, L) inverse-variance-like weights.
    reduce_mean : if True, also return the scalar mean over pixels.

    Returns the per-pixel vector (n_pix,); or (vector, scalar_mean) when
    reduce_mean is True. The vector is the natural interface for a PINN's
    coordinate sampling; the scalar mean is what a plain optimiser descends.
    """
    r2 = (obs - syn) ** 2
    if weight is not None:
        r2 = r2 * weight
    chi2 = r2.sum(dim=-1)                                # (n_pix,)
    if reduce_mean:
        return chi2, chi2.mean()
    return chi2
