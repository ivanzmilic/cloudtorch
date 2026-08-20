"""
regularizers.py -- Step 5 of the develop/ plan (see instructions.md).

Explicit, configurable penalties added to the step-4 PINN loss to tame the
background/cloud degeneracy found in step 4 (the fitted background grew emission
spikes that the clouds then cancelled -- low RMS, unphysical decomposition).
The loss becomes

        L = chi2  +  R_smooth  +  R_bound .

Because the parameters are a *field* p_k = f_theta(x, y, t), the natural place to
regularise is coordinate space:

  * R_smooth  -- penalise the TRUE coordinate derivatives (autograd) of each
                 field, per parameter and per axis (space vs time). Mesh-free and
                 exact, so it survives random-minibatch sampling (needed for the
                 step-6 full-cube run). One extra backward through the MLP only
                 (not the Voigt physics), so it is mild.
  * R_bound   -- per-parameter SOFT hinges  w * <ReLU(l - p)^2> + w * <ReLU(p - u)^2>:
                 zero inside [l, u], quadratic outside, gradient continuous at the
                 edge. Lets a parameter roam freely inside its physical range and
                 only pushes back when it tries to leave (unlike a hard transform,
                 which distorts the interior). Generalises utils.regu_min/regu_max.
  * no_emission -- the load-bearing physical bound: an upper hinge on the
                 reconstructed background SPECTRUM b = mu + c@V, capping it at the
                 continuum (~1). Costs nothing for absorption (b < 1); it only
                 fights the emission peaks the clouds were cancelling.

Everything is a pure function of the field outputs (c, p_cloud), the (grad-tracking)
coords, and the fixed PCA basis; nothing here runs an optimiser or touches data.
composite_synth / chi2_per_pixel supply the data term, unchanged.

--------------------------------------------------------------------------------
Field layout (the P=14 differentiable, meaningful quantities; loga is frozen and
excluded because its coordinate derivative is identically zero):

    col   0  1  2  3  4  5   6    7     8     9    10    11    12    13
    name  c0 c1 c2 c3 c4 c5  S1  dtau1 vlos1 dv1  S2   dtau2 vlos2 dv2

Groups (for the human-facing REG config): 'bkg' (the 6 c), 'S', 'dtau', 'vlos',
'dv' (each pairs the two clouds), plus the special 'no_emission'.
"""

import torch
import torch.nn.functional as F

__all__ = [
    "FIELD_NAMES", "FIELD_GROUPS", "DEFAULT_REG",
    "assemble_fields", "smoothness_penalty", "hinge_bounds", "no_emission",
    "background_anchor", "expand_group_config", "regularization_loss",
]

# ---- field layout -------------------------------------------------------------
# p_cloud columns [S1,dtau1,vlos1,dv1,loga1, S2,dtau2,vlos2,dv2,loga2] -> keep,
# dropping the two frozen loga columns (4, 9) whose coordinate derivative is 0.
_CLOUD_KEEP = [0, 1, 2, 3, 5, 6, 7, 8]
_CLOUD_NAMES = ["S1", "dtau1", "vlos1", "dv1", "S2", "dtau2", "vlos2", "dv2"]


def field_layout(n_bkg=6):
    """(names, groups, P) of the P = n_bkg + 8 differentiable fields
    [c_0..c_{n_bkg-1}, S1,dtau1,vlos1,dv1, S2,dtau2,vlos2,dv2] for a given number of
    background columns. n_bkg=6 is the fitted-PCA-background layout (step 5); n_bkg=0
    is the given-background mode (step 5a: cloud fields only, no 'bkg'/no_emission/
    anchor). Cloud groups are simply offset by n_bkg."""
    names = ["c%d" % i for i in range(n_bkg)] + _CLOUD_NAMES
    o = n_bkg
    groups = {"bkg":  list(range(n_bkg)),
              "S":    [o + 0, o + 4], "dtau": [o + 1, o + 5],
              "vlos": [o + 2, o + 6], "dv":   [o + 3, o + 7]}
    return names, groups, len(names)


# default (fitted-background) layout, unchanged from before
FIELD_NAMES, FIELD_GROUPS, P = field_layout(6)

# Default config -- a *starting point* for the weight scan, NOT tuned values.
# The whole job of step 5 is balancing these against the chi2 scale (see notebook);
# absolute magnitudes here are placeholders. The intent (per the outline): smooth
# the background HARD in space and time + cap it at the continuum; smooth the clouds
# LIGHTLY and give them physical hinges.
DEFAULT_REG = {
    "bkg":  dict(w_xy=1e-1, w_t=1e-1),
    "S":    dict(w_xy=1e-3, w_t=1e-3, lo=0.0,   w_lo=1e-2),
    "dtau": dict(w_xy=1e-3, w_t=1e-3, lo=0.0,   w_lo=1e-2),
    "vlos": dict(w_xy=1e-3, w_t=1e-3, lo=-60.0, hi=60.0, w_lo=1e-2, w_hi=1e-2),
    "dv":   dict(w_xy=1e-3, w_t=1e-3, lo=8.0,   w_lo=1e-2),
    "no_emission": dict(weight=1.0, cap=1.0),
}


# ====================================================================
def assemble_fields(c, p_cloud):
    """Concatenate the field outputs into the canonical (N, P) tensor whose columns
    are FIELD_NAMES: the 6 background coeffs c followed by the 8 non-frozen cloud
    quantities. Differentiable; carries whatever graph c / p_cloud carry."""
    return torch.cat([c, p_cloud[:, _CLOUD_KEEP]], dim=1)


# ====================================================================
def smoothness_penalty(fields, coords, w_xy, w_t):
    """Autograd coordinate-derivative smoothness, per field and per axis.

    R = sum_k [ w_xy[k] * <(d p_k/dx)^2 + (d p_k/dy)^2>  +  w_t[k] * <(d p_k/dt)^2> ]

    fields : (N, P) tensor, output of assemble_fields, depending on `coords`.
    coords : (N, 3) tensor with requires_grad=True, columns (x, y, t). Because each
             output row depends only on its own coord row (the MLP is applied
             row-wise), the gradient of a column's sum recovers the per-pixel
             coordinate derivative -- so this is exact on arbitrary minibatches.
    w_xy, w_t : (P,) per-field weights (0 disables that field / axis).

    Returns a scalar. Uses create_graph=True so the penalty backpropagates into the
    network weights (second-order through the MLP only, not the physics).
    """
    if not coords.requires_grad:
        raise ValueError("smoothness_penalty needs coords with requires_grad=True")
    w_xy = torch.as_tensor(w_xy, dtype=fields.dtype, device=fields.device)
    w_t  = torch.as_tensor(w_t,  dtype=fields.dtype, device=fields.device)
    total = fields.new_zeros(())
    n = fields.shape[0]
    for k in range(fields.shape[1]):
        if float(w_xy[k]) == 0.0 and float(w_t[k]) == 0.0:
            continue
        g = torch.autograd.grad(fields[:, k].sum(), coords,
                                create_graph=True, retain_graph=True)[0]   # (N,3)
        if float(w_xy[k]) != 0.0:
            total = total + w_xy[k] * (g[:, 0] ** 2 + g[:, 1] ** 2).sum() / n
        if float(w_t[k]) != 0.0:
            total = total + w_t[k] * (g[:, 2] ** 2).sum() / n
    return total


# ====================================================================
def hinge_bounds(fields, lo, hi, w_lo, w_hi):
    """Per-field soft one-sided quadratic hinges (generalised utils.regu_min/max).

    R = sum_k [ w_lo[k] * <ReLU(lo[k] - p_k)^2>  +  w_hi[k] * <ReLU(p_k - hi[k])^2> ]

    lo, hi : (P,) bounds; entries may be NaN (or the weight 0) to disable that side.
    w_lo, w_hi : (P,) weights. Returns a scalar (mean over pixels).
    """
    lo   = torch.as_tensor(lo,   dtype=fields.dtype, device=fields.device)
    hi   = torch.as_tensor(hi,   dtype=fields.dtype, device=fields.device)
    w_lo = torch.as_tensor(w_lo, dtype=fields.dtype, device=fields.device)
    w_hi = torch.as_tensor(w_hi, dtype=fields.dtype, device=fields.device)
    total = fields.new_zeros(())
    for k in range(fields.shape[1]):
        if float(w_lo[k]) != 0.0 and torch.isfinite(lo[k]):
            total = total + w_lo[k] * F.relu(lo[k] - fields[:, k]).pow(2).mean()
        if float(w_hi[k]) != 0.0 and torch.isfinite(hi[k]):
            total = total + w_hi[k] * F.relu(fields[:, k] - hi[k]).pow(2).mean()
    return total


# ====================================================================
def no_emission(pca, c, cap=1.0, weight=1.0):
    """Upper hinge on the reconstructed background spectrum b = mu + c@V, forbidding
    emission above the continuum `cap` (~1 in Iqs units):

        R = weight * < ReLU(b - cap)^2 >   (mean over pixels and wavelength).

    Zero wherever the background stays in absorption (b <= cap); it only penalises
    the emission peaks the clouds were cancelling. Differentiable w.r.t. c.
    """
    if weight == 0.0:
        return c.new_zeros(())
    b = pca.reconstruct(c)                              # (N, L)
    return weight * F.relu(b - cap).pow(2).mean()


# ====================================================================
def background_anchor(c, c_ref, weight=1.0):
    """L2 anchor pulling the background coeffs c toward a reference vector c_ref --
    the filament-free last-N-frame background, projected onto the basis:

        R = weight * < ||c - c_ref||^2 >   (sum over K coeffs, mean over pixels).

    This is the load-bearing fix for the OTHER background degeneracy: the clouds
    can be fed a background whose H-alpha core has been filled UP toward the
    continuum (bright/flat, no absorption) -- which stays <= 1, so the no_emission
    cap never catches it. Anchoring c toward the filament-free reference keeps the
    background at its true quiet-Sun profile (with its absorption line) and forces
    the clouds to supply the filament absorption. Use the last-N-frame background,
    NOT the observed projection (which already contains the absorption -- the
    freeze test showed that route fails).

    c     : (N, K) field background coefficients.
    c_ref : (N, K) or (K,) reference coefficients (broadcast over pixels).
    """
    if weight == 0.0:
        return c.new_zeros(())
    c_ref = torch.as_tensor(c_ref, dtype=c.dtype, device=c.device)
    return weight * (c - c_ref).pow(2).sum(dim=-1).mean()


# ====================================================================
def expand_group_config(reg=None, n_bkg=6):
    """Turn a human-facing grouped REG dict into per-field (P,) weight/bound
    vectors for the P = n_bkg + 8 layout. Missing entries default to 0 weight and NaN
    (disabled) bounds, so a partial config just turns terms on. Pass n_bkg=0 for the
    given-background mode (cloud-only). Returns a dict with keys w_xy, w_t, lo, hi,
    w_lo, w_hi (each a (P,) tensor) plus ('no_weight','no_cap','anchor_weight').
    """
    reg = DEFAULT_REG if reg is None else reg
    _, groups, P = field_layout(n_bkg)
    w_xy = torch.zeros(P); w_t = torch.zeros(P)
    lo = torch.full((P,), float("nan")); hi = torch.full((P,), float("nan"))
    w_lo = torch.zeros(P); w_hi = torch.zeros(P)
    for group, cfg in reg.items():
        if group in ("no_emission", "anchor"):
            continue
        if group not in groups:
            raise KeyError("unknown REG group %r; expected one of %s"
                           % (group, list(groups) + ["no_emission", "anchor"]))
        for k in groups[group]:
            w_xy[k] = cfg.get("w_xy", 0.0)
            w_t[k]  = cfg.get("w_t", 0.0)
            if "lo" in cfg:
                lo[k] = cfg["lo"]
            if "hi" in cfg:
                hi[k] = cfg["hi"]
            w_lo[k] = cfg.get("w_lo", 0.0)
            w_hi[k] = cfg.get("w_hi", 0.0)
    ne = reg.get("no_emission", {})
    an = reg.get("anchor", {})
    return dict(w_xy=w_xy, w_t=w_t, lo=lo, hi=hi, w_lo=w_lo, w_hi=w_hi,
                no_weight=float(ne.get("weight", 0.0)), no_cap=float(ne.get("cap", 1.0)),
                anchor_weight=float(an.get("weight", 0.0)))


# ====================================================================
def regularization_loss(c, p_cloud, coords, pca=None, reg=None, weights=None, c_ref=None):
    """Assemble the full regulariser R = R_smooth + R_bound + no_emission + anchor
    for one field evaluation. Add its ['total'] to the chi2 data term.

    c, p_cloud : field(coords) outputs. In the given-background mode c is (N, 0), so
                 only R_smooth + R_bound on the cloud fields survive.
    coords     : the SAME (N,3) tensor fed to the field, with requires_grad=True.
    pca        : the fixed PCA basis (for no_emission / anchor). None (given mode) or
                 a config with those terms off -> both are skipped.
    reg        : grouped REG config (defaults to DEFAULT_REG).
    weights    : optional pre-expanded config from expand_group_config(reg, n_bkg) --
                 pass it to avoid re-expanding every iteration in the training loop.
    c_ref      : optional (N,K) or (K,) reference background coeffs for the anchor
                 (the filament-free last-N-frame background, projected). MUST be
                 sliced to the SAME pixels as c when minibatching. If None the
                 anchor term is 0 regardless of its weight.

    Returns a dict {'smooth','bound','no_emission','anchor','total'} of scalars.
    """
    w = expand_group_config(reg, n_bkg=c.shape[1]) if weights is None else weights
    fields = assemble_fields(c, p_cloud)
    r_smooth = smoothness_penalty(fields, coords, w["w_xy"], w["w_t"])
    r_bound  = hinge_bounds(fields, w["lo"], w["hi"], w["w_lo"], w["w_hi"])
    r_noem   = (no_emission(pca, c, cap=w["no_cap"], weight=w["no_weight"])
                if pca is not None else c.new_zeros(()))
    r_anchor = (background_anchor(c, c_ref, weight=w["anchor_weight"])
                if c_ref is not None else c.new_zeros(()))
    return {"smooth": r_smooth, "bound": r_bound, "no_emission": r_noem,
            "anchor": r_anchor, "total": r_smooth + r_bound + r_noem + r_anchor}
