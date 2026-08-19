"""
pinn_model.py -- Step 4 of the develop/ plan (see instructions.md).

A simple MLP neural field with Fourier-feature encoding that predicts the
per-pixel model parameters as a smooth function of coordinates (x, y, t), trained
THROUGH the step-3 physics. Optimising the network weights (not per-pixel values)
regularises the maps in space and time; the Fourier bandwidths (sigma_xy, sigma_t)
cap how rough the maps can get.

The field predicts 14 numbers per coordinate: 6 background PCA coefficients c +
8 cloud values [S, dtau, vlos, dv] x 2. log a is FROZEN at (-4, -5) and inserted.

Physical ranges are enforced by smooth output transforms (not by abs() in the
cloud model, which would create a sign degeneracy / gradient kinks that hurt a
smooth field):
    S    = S_MAX * sigmoid(.)          in [0, S_MAX]
    dtau = softplus(.)                 >= 0
    dv   = DV_FLOOR + softplus(.)      >= DV_FLOOR   (the ~thermal floor step-3 penalised)
    vlos = identity                    (sign carries the Doppler direction)
    c    = identity
The affine output head is anchored so that at init (small head weights) the field
predicts the step-3 inits everywhere; the background bias can be set to the mean
projected coefficients so the background starts at the field-average, not at mu.

Everything downstream -- composite_model.composite_synth / chi2_per_pixel -- is
reused unchanged. Nothing here runs an optimiser or loads data.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from composite_model import composite_synth, chi2_per_pixel   # reused unchanged

__all__ = ["FourierFeatures", "ParameterField", "grid_coords",
           "composite_synth", "chi2_per_pixel"]

# physical init values (step-3), per cloud: S, dtau, vlos, dv
_CLOUD_INIT = dict(S1=0.2, t1=4.5, v1=-20.0, d1=12.0,
                   S2=0.6, t2=4.5, v2=20.0,  d2=12.0)
_LOGA_FROZEN = (-4.0, -5.0)
# pre-activation output scales (sensitivity of each channel to the MLP output)
_SCALE = [1.0] * 6 + [1.5, 1.0, 20.0, 1.0,   1.5, 1.0, 20.0, 1.0]


def _logit(y):
    return math.log(y / (1.0 - y))


def _inv_softplus(y):
    # y = softplus(x) = ln(1 + e^x)  ->  x = ln(e^y - 1)
    return math.log(math.expm1(y))


# ====================================================================
class FourierFeatures(nn.Module):
    """Fixed anisotropic Gaussian random Fourier features (Tancik et al.).

    gamma(v) = [sin(2 pi B v), cos(2 pi B v)], B ~ N(0,1) scaled per axis by
    `sigmas` -- space (sigma_xy) and time (sigma_t) get independent bandwidths.
    """

    def __init__(self, d_in=3, n_freq=64, sigmas=(4.0, 4.0, 2.0),
                 include_input=False, seed=0):
        super().__init__()
        if len(sigmas) != d_in:
            raise ValueError(f"sigmas must have length d_in={d_in}, got {len(sigmas)}")
        g = torch.Generator().manual_seed(int(seed))
        B = torch.randn(n_freq, d_in, generator=g)
        B = B * torch.as_tensor(sigmas, dtype=torch.float32)[None, :]
        self.register_buffer("B", B)
        self.include_input = bool(include_input)
        self.d_in = d_in
        self.n_out = 2 * n_freq + (d_in if include_input else 0)

    def forward(self, coords):
        proj = (2.0 * math.pi) * (coords @ self.B.t())
        feats = torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)
        if self.include_input:
            feats = torch.cat([coords, feats], dim=-1)
        return feats


# ====================================================================
class ParameterField(nn.Module):
    """Coordinate MLP: (x, y, t) -> (background coeffs c (N,6), p_cloud (N,10)).

    p_cloud columns are [S,dtau,vlos,dv,loga] x 2 with loga frozen & inserted,
    and S/dtau/dv passed through range transforms. Feed straight to
    composite_synth(x, pca, c, p_cloud).
    """

    def __init__(self, d_in=3, n_freq=64, sigmas=(4.0, 4.0, 2.0),
                 width=192, depth=4, include_input=False,
                 loga=_LOGA_FROZEN, s_max=0.8, dv_floor=8.0,
                 bkg_bias=None, head_std=1e-3, seed=0):
        super().__init__()
        if not s_max > max(_CLOUD_INIT["S1"], _CLOUD_INIT["S2"]):
            raise ValueError("s_max must exceed the initial S values")
        if not dv_floor < min(_CLOUD_INIT["d1"], _CLOUD_INIT["d2"]):
            raise ValueError("dv_floor must be below the initial dv values")
        self.s_max = float(s_max)
        self.dv_floor = float(dv_floor)

        self.ff = FourierFeatures(d_in, n_freq, sigmas, include_input, seed)
        layers, d = [], self.ff.n_out
        for _ in range(depth):
            layers += [nn.Linear(d, width), nn.SiLU()]
            d = width
        self.mlp = nn.Sequential(*layers)
        self.head = nn.Linear(width, 14)
        nn.init.normal_(self.head.weight, std=head_std)   # start at the anchored bias
        nn.init.zeros_(self.head.bias)

        ci = _CLOUD_INIT
        bkg = [0.0] * 6 if bkg_bias is None else \
            list(torch.as_tensor(bkg_bias, dtype=torch.float32).flatten()[:6])
        bias_pre = bkg + [
            _logit(ci["S1"] / self.s_max), _inv_softplus(ci["t1"]),
            ci["v1"], _inv_softplus(ci["d1"] - self.dv_floor),
            _logit(ci["S2"] / self.s_max), _inv_softplus(ci["t2"]),
            ci["v2"], _inv_softplus(ci["d2"] - self.dv_floor),
        ]
        self.register_buffer("bias_pre", torch.tensor(bias_pre, dtype=torch.float32))
        self.register_buffer("scale",    torch.tensor(_SCALE,   dtype=torch.float32))
        self.register_buffer("loga",     torch.tensor(loga,     dtype=torch.float32))

    def forward(self, coords):
        raw = self.bias_pre + self.scale * self.head(self.mlp(self.ff(coords)))   # (N,14)
        c = raw[:, 0:6]
        S1 = self.s_max * torch.sigmoid(raw[:, 6]);  t1 = F.softplus(raw[:, 7])
        v1 = raw[:, 8];                              d1 = self.dv_floor + F.softplus(raw[:, 9])
        S2 = self.s_max * torch.sigmoid(raw[:, 10]); t2 = F.softplus(raw[:, 11])
        v2 = raw[:, 12];                             d2 = self.dv_floor + F.softplus(raw[:, 13])
        N = coords.shape[0]
        la1 = raw.new_full((N,), float(self.loga[0]))     # frozen constants (no grad)
        la2 = raw.new_full((N,), float(self.loga[1]))
        p_cloud = torch.stack([S1, t1, v1, d1, la1, S2, t2, v2, d2, la2], dim=1)   # (N,10)
        return c, p_cloud


# ====================================================================
def grid_coords(nt, ny, nx, device=None, dtype=torch.float32):
    """Normalised (x, y, t) coordinates in [-1, 1] for a regular (nt, ny, nx) grid,
    flattened in the SAME (t, y, x) order as data.reshape(nt,ny,nx,L).reshape(-1,L).
    Returns (nt*ny*nx, 3), columns (x, y, t). For the full-cube run, normalise t
    over the true time range instead of this local window."""
    def ax(n):
        if n == 1:
            return torch.zeros(1, dtype=dtype, device=device)
        return torch.linspace(-1.0, 1.0, n, dtype=dtype, device=device)

    tt, yy, xx = torch.meshgrid(ax(nt), ax(ny), ax(nx), indexing="ij")
    return torch.stack([xx.reshape(-1), yy.reshape(-1), tt.reshape(-1)], dim=1)
