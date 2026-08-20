"""
pinn_model.py -- Step 4 of the develop/ plan (see instructions.md).

A simple MLP neural field with Fourier-feature encoding that predicts the
per-pixel model parameters as a smooth function of coordinates (x, y, t), trained
THROUGH the step-3 physics. Optimising the network weights (not per-pixel values)
regularises the maps in space and time; the Fourier bandwidths (sigma_xy, sigma_t)
cap how rough the maps can get.

The field predicts n_bkg + n_free numbers per coordinate -- FLEXIBLE on both axes:
  * n_bkg  background coefficients c (0 for the given-background mode of step 3a,
    K for a fitted PCA background of step 3/4);
  * n_free cloud pre-activations, one per UN-FROZEN cloud parameter.
The 10 cloud slots are [S, dtau, vlos, dv, log a] x 2 (canonical p_cloud order).
Any slot can be FROZEN to its init via `freeze=(...)`: a frozen slot emits no output
column and is inserted as its (spatially/temporally uniform) init constant -- the
mechanism the model has always used for log a, now generalised to any parameter.
Default freeze = (log a1, log a2), giving n_free = 8 (the step-4 behaviour).

Physical ranges are enforced by smooth output transforms on the FREE slots (not by
abs() in the cloud model, which would create a sign degeneracy / gradient kinks that
hurt a smooth field):
    S    = S_MAX * sigmoid(.)          in [0, S_MAX]
    dtau = softplus(.)                 >= 0
    dv   = DV_FLOOR + softplus(.)      >= DV_FLOOR   (the ~thermal floor step-3 penalised)
    vlos = identity                    (sign carries the Doppler direction)
    log a= identity                    (usually frozen; clamp it if freed)
    c    = identity
The affine output head is anchored so that at init (small head weights) the field
predicts the step-3 inits everywhere; the background bias can be set to the mean
projected coefficients so the background starts at the field-average, not at mu.

Everything downstream -- composite_model.composite_synth (fitted background) or
composite_synth_given (fixed background) / chi2_per_pixel -- is reused unchanged.
Nothing here runs an optimiser or loads data.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from composite_model import (composite_synth, composite_synth_given,   # reused unchanged
                             chi2_per_pixel)

__all__ = ["FourierFeatures", "ParameterField", "grid_coords",
           "composite_synth", "composite_synth_given", "chi2_per_pixel"]

_LOGA_FROZEN = (-4.0, -5.0)

# The 10 cloud slots, in the canonical p_cloud column order expected by
# composite_model (matches _CLOUD_KEYS): [S,dtau,vlos,dv,loga] x 2.
_SLOT_NAMES = ["S1", "dtau1", "vlos1", "dv1", "loga1",
               "S2", "dtau2", "vlos2", "dv2", "loga2"]
# physical init values (step-3 anchors); a FROZEN slot is held here everywhere.
_SLOT_INIT = {"S1": 0.2, "dtau1": 4.5, "vlos1": -20.0, "dv1": 12.0, "loga1": _LOGA_FROZEN[0],
              "S2": 0.6, "dtau2": 4.5, "vlos2":  20.0, "dv2": 12.0, "loga2": _LOGA_FROZEN[1]}
# range-transform kind per slot (applied to FREE slots only).
_SLOT_KIND = {"S1": "S", "dtau1": "softplus", "vlos1": "identity", "dv1": "dv", "loga1": "identity",
              "S2": "S", "dtau2": "softplus", "vlos2": "identity", "dv2": "dv", "loga2": "identity"}
# pre-activation output scale (sensitivity of each free channel to the MLP output).
_SLOT_SCALE = {"S1": 1.5, "dtau1": 1.0, "vlos1": 20.0, "dv1": 1.0, "loga1": 1.0,
               "S2": 1.5, "dtau2": 1.0, "vlos2": 20.0, "dv2": 1.0, "loga2": 1.0}


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
    """Coordinate MLP: (x, y, t) -> (background coeffs c (N, n_bkg), p_cloud (N,10)).

    The head width is FLEXIBLE = n_bkg + n_free:
      * n_bkg  background columns c  -- 0 for the given-background mode (step 3a; feed
        p_cloud to composite_synth_given), or K for a fitted PCA background (step 3/4;
        feed c, p_cloud to composite_synth). Set via `n_bkg` or inferred from `bkg_bias`.
      * n_free cloud columns -- one per UN-FROZEN cloud slot. `freeze=(...)` names slots
        (from _SLOT_NAMES) to hold at their init; a frozen slot emits no column and is
        inserted as its uniform init constant. Default freeze = (loga1, loga2) -> n_free=8.

    p_cloud is always the full (N,10) in canonical column order; free slots pass through
    their range transform (S sigmoid, dtau/dv softplus(+floor), vlos/loga identity).
    In given mode, c has shape (N, 0); use `_, pc = field(coords)`.
    """

    def __init__(self, d_in=3, n_freq=64, sigmas=(4.0, 4.0, 2.0),
                 width=192, depth=4, include_input=False,
                 loga=_LOGA_FROZEN, s_max=0.8, dv_floor=8.0,
                 bkg_bias=None, n_bkg=None, freeze=("loga1", "loga2"), init=None,
                 head_std=1e-3, seed=0):
        super().__init__()
        self.s_max = float(s_max)
        self.dv_floor = float(dv_floor)

        # --- slot init values: base anchors, loga override (back-compat), then `init` ---
        slot_init = dict(_SLOT_INIT)
        slot_init["loga1"], slot_init["loga2"] = float(loga[0]), float(loga[1])
        for k, v in (init or {}).items():
            if k not in _SLOT_INIT:
                raise ValueError("unknown cloud slot in init: %r" % (k,))
            slot_init[k] = float(v)

        # --- which slots are free vs frozen (held at init) ---
        frozen = set(freeze)
        bad = frozen - set(_SLOT_NAMES)
        if bad:
            raise ValueError("unknown cloud slot(s) in freeze: %s" % sorted(bad))
        self._is_free = [name not in frozen for name in _SLOT_NAMES]
        free = [name for name in _SLOT_NAMES if name not in frozen]

        # guards apply only to FREE slots that use a bounded transform
        free_S = [slot_init[n] for n in free if _SLOT_KIND[n] == "S"]
        if free_S and not s_max > max(free_S):
            raise ValueError("s_max must exceed the initial S values of free S slots")
        free_dv = [slot_init[n] for n in free if _SLOT_KIND[n] == "dv"]
        if free_dv and not dv_floor < min(free_dv):
            raise ValueError("dv_floor must be below the initial dv values of free dv slots")

        self.ff = FourierFeatures(d_in, n_freq, sigmas, include_input, seed)
        layers, d = [], self.ff.n_out
        for _ in range(depth):
            layers += [nn.Linear(d, width), nn.SiLU()]
            d = width
        self.mlp = nn.Sequential(*layers)

        # --- number of background columns: explicit n_bkg wins, else infer from bkg_bias ---
        if n_bkg is None:
            if bkg_bias is None:
                n_bkg, bkg = 6, [0.0] * 6
            else:
                bkg = list(torch.as_tensor(bkg_bias, dtype=torch.float32).flatten())
                n_bkg = len(bkg)
        else:
            n_bkg = int(n_bkg)
            if bkg_bias is None:
                bkg = [0.0] * n_bkg
            else:
                bkg = list(torch.as_tensor(bkg_bias, dtype=torch.float32).flatten())
                if len(bkg) != n_bkg:
                    raise ValueError("bkg_bias length %d != n_bkg %d" % (len(bkg), n_bkg))
        self.n_bkg = n_bkg

        # --- head: n_bkg background + n_free cloud pre-activations ---
        bias_pre = list(bkg) + [self._slot_bias(n, slot_init[n]) for n in free]
        scale = [1.0] * n_bkg + [_SLOT_SCALE[n] for n in free]
        self.head = nn.Linear(width, n_bkg + len(free))
        nn.init.normal_(self.head.weight, std=head_std)   # start at the anchored bias
        nn.init.zeros_(self.head.bias)
        self.register_buffer("bias_pre", torch.tensor(bias_pre, dtype=torch.float32))
        self.register_buffer("scale",    torch.tensor(scale,    dtype=torch.float32))
        # frozen slots are inserted as these python-float constants (canonical order)
        self._slot_init = [slot_init[n] for n in _SLOT_NAMES]
        # kept only so the state_dict key set is unchanged from the pre-refactor model
        # (a fit_cube checkpoint stays resumable); not used by forward.
        self.register_buffer("loga", torch.tensor(
            [slot_init["loga1"], slot_init["loga2"]], dtype=torch.float32))

    def _slot_bias(self, name, v):
        """Pre-activation bias for a free slot that maps to its init v under the transform."""
        kind = _SLOT_KIND[name]
        if kind == "S":
            return _logit(v / self.s_max)
        if kind == "softplus":
            return _inv_softplus(v)
        if kind == "dv":
            return _inv_softplus(v - self.dv_floor)
        return v                                          # identity (vlos, loga)

    def _transform(self, name, x):
        """Range transform applied to a free slot's pre-activation column."""
        kind = _SLOT_KIND[name]
        if kind == "S":
            return self.s_max * torch.sigmoid(x)
        if kind == "softplus":
            return F.softplus(x)
        if kind == "dv":
            return self.dv_floor + F.softplus(x)
        return x                                          # identity (vlos, loga)

    def forward(self, coords):
        raw = self.bias_pre + self.scale * self.head(self.mlp(self.ff(coords)))   # (N, n_bkg+n_free)
        N = coords.shape[0]
        c = raw[:, :self.n_bkg]                           # (N, n_bkg); (N,0) in given mode
        free = raw[:, self.n_bkg:]
        cols, j = [], 0
        for i, name in enumerate(_SLOT_NAMES):
            if self._is_free[i]:
                cols.append(self._transform(name, free[:, j])); j += 1
            else:
                cols.append(raw.new_full((N,), self._slot_init[i]))   # frozen: no grad
        p_cloud = torch.stack(cols, dim=1)                # (N,10) canonical order
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
