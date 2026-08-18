"""
background_model.py -- Step 2 of the develop/ plan (see instructions.md).

A differentiable, K-parameter-per-pixel background model built on the fixed PCA
dictionary { mu, V } from step 1 (pca_background.PCABackground). Each pixel's
background is a linear combination of the K eigenprofiles,

        b = mu + c @ V ,        c.shape == (n_pixels, K),   V.shape == (K, L)

so the whole background field is described by only K free numbers per pixel
(K = 6 for the MiHI filament). This replaces the fixed, per-spectrum skglm
background (new_fit - new_extrafit) with a compact model that step 3 optimises
jointly with the two clouds; the returned spectra are the incoming intensity
I_in fed to cloud_model_torch.cloud(...).

This module deliberately contains NO optimiser: step 2 is "model + init only".
It provides two functions, mirroring the entry points of cloud_model_torch.py:

    init_background_coeffs(pca, obs)  -> nn.Parameter (n_pixels, K)   [warm start]
    background(pca, coeffs)           -> Tensor      (n_pixels, L)    [forward model]

Both reuse PCABackground.project / .reconstruct rather than re-deriving anything.
"""

import torch
import torch.nn as nn

__all__ = ["init_background_coeffs", "background"]


# ====================================================================
def _flatten_spectra(obs):
    """Turn observed spectra into a 2-D (n_pixels, L) tensor.

    Accepts (n_pixels, L) already, a cube (..., L) which is flattened over every
    axis except the last, or a single spectrum (L,).
    """
    x = obs if isinstance(obs, torch.Tensor) else torch.as_tensor(obs)
    if x.ndim > 2:
        x = x.reshape(-1, x.shape[-1])
    elif x.ndim == 1:
        x = x[None, :]
    return x


# ====================================================================
def init_background_coeffs(pca, obs, k=None, requires_grad=True,
                           device=None, dtype=torch.float32):
    """Warm-start the per-pixel background coefficients by projecting the observed
    spectra onto the basis: c0 = (obs - mu) @ V^T.

    Because `obs` (tofit) still contains the filament absorption, c0 is the best
    background-subspace approximation of the data -- slightly biased low at the
    absorbed wavelengths, but a good starting point that step 3's joint fit then
    refines as the clouds take up the absorption.

    Parameters
    ----------
    pca : PCABackground
        The fixed basis from step 1.
    obs : array or tensor
        Observed spectra, (n_pixels, L) or a cube (..., L). MUST already live in
        the same lambda-crop and Iqs normalisation as the basis.
    k : int or None
        Number of coefficients (defaults to pca.n_components, i.e. K).
    requires_grad, device, dtype :
        Standard nn.Parameter placement/precision controls.

    Returns
    -------
    nn.Parameter of shape (n_pixels, k) -- the trainable background coefficients.
    """
    x = _flatten_spectra(obs).to(dtype)
    c0 = pca.project(x, k=k)                       # (n_pixels, k)
    c0 = c0.detach().clone().to(device=device, dtype=dtype)
    return nn.Parameter(c0, requires_grad=requires_grad)


# ====================================================================
def background(pca, coeffs):
    """Forward background model: b = mu + c @ V.

    coeffs : (n_pixels, k) tensor, typically an nn.Parameter from
             init_background_coeffs. Returns (n_pixels, L); differentiable w.r.t.
             coeffs (mu, V are constants moved onto coeffs' device/dtype).
    """
    return pca.reconstruct(coeffs)
