"""
pca_background.py -- Step 1 of the develop/ plan (see instructions.md).

Build a compact, differentiable PCA background dictionary {mu, V} from a set of
background-dominated spectra -- typically the last N snapshots of the
(t, y, x, lambda) cube. The CALLER supplies the spectra, so the exact same code
works whether you feed raw frames or already-reconstructed backgrounds
(e.g. new_fit - new_extrafit): this module does not decide that for you.

Downstream (step 2) the background of each pixel is modelled with ~K free
coefficients c as

        b = mu + c @ V ,        c.shape == (n_pixels, K),   V.shape == (K, L)

which is linear and autograd-friendly, so it plugs straight into the cloud model
as the incoming intensity I_in.

This module ONLY builds the fixed basis and the two helpers reconstruct()/
project() that use it. It never fits clouds and never runs an optimiser.

All torch. The eigen-solve is done in float64 for stability; the stored basis is
float32 by default. No I/O side effects beyond optional save()/load().

Typical use (to be exercised in the notebook, not here):

    from pca_background import fit_background_basis
    # bkg : whatever spectra you consider "background", e.g. cube[-10:] or a
    #       reconstructed-background cube of shape (t, y, x, lambda)
    pca = fit_background_basis(bkg, n_snapshots=10, n_components=10,
                               wl_slice=slice(100, 500), normalization=Iqs)
    I_in = pca.reconstruct(c)          # c : (n_pixels, K) nn.Parameter -> I_in (n_pixels, L)
    c0   = pca.project(observed_bkg)   # initialise c from data
"""

import numpy as np
import torch

__all__ = ["PCABackground", "fit_background_basis"]


# ====================================================================
def _to_slice(wl_slice):
    """Normalise the wl_slice argument to something usable for indexing the
    last (wavelength) axis. Accepts None, a slice, a (start, stop[, step])
    tuple/list, or an explicit 1-D index array/tensor."""
    if wl_slice is None or isinstance(wl_slice, slice):
        return wl_slice
    if isinstance(wl_slice, (tuple, list)) and len(wl_slice) in (2, 3) \
            and all(isinstance(v, (int, np.integer)) or v is None for v in wl_slice):
        return slice(*wl_slice)
    # otherwise treat it as an explicit index array
    return torch.as_tensor(wl_slice, dtype=torch.long)


# ====================================================================
def _as_spectra(data, n_snapshots, wl_slice, normalization, dtype=torch.float64):
    """Turn the user-supplied `data` into a 2-D spectra matrix X of shape
    (M, L), in `dtype`.

    - ndim >= 3 : axis 0 is time; keep the last `n_snapshots` frames (all of
                  them if n_snapshots is None) and flatten every axis except the
                  last one, which is wavelength.  So (t, y, x, L) -> (N*y*x, L)
                  and (t, pix, L) -> (N*pix, L).
    - ndim == 2 : already (M, L); used as-is (n_snapshots is ignored).

    `wl_slice` (optional) crops the wavelength axis; `normalization` (optional,
    a scalar or broadcastable tensor) divides the spectra -- pass the SAME
    constant you use to normalise the fit data so the basis lives in fit units.
    """
    x = torch.as_tensor(data)
    x = x.to(dtype)

    if x.ndim >= 3:
        if n_snapshots is not None and x.shape[0] > n_snapshots:
            x = x[-n_snapshots:]
        L = x.shape[-1]
        x = x.reshape(-1, L)
    elif x.ndim == 2:
        pass  # (M, L) already
    else:
        raise ValueError(
            f"data must be 2-D (M, L) or >=3-D (t, ..., L); got shape {tuple(x.shape)}"
        )

    sl = _to_slice(wl_slice)
    if sl is not None:
        if isinstance(sl, slice):
            x = x[:, sl]
        else:  # explicit index array
            x = x.index_select(1, sl.to(x.device))

    if normalization is not None:
        norm = torch.as_tensor(normalization, dtype=x.dtype, device=x.device)
        x = x / norm

    return x


# ====================================================================
class PCABackground:
    """A fixed PCA background basis {mu, V} plus reconstruct()/project().

    Attributes
    ----------
    mean_ : (L,) tensor
        Mean spectrum mu.  Kept explicitly -- the background is mu + c @ V, not
        a zero-mean expansion.
    components_full : (L, L) tensor
        ALL principal directions, one per row, ordered by descending variance.
        The active basis V is simply the first `n_components` rows, so K can be
        changed after the fact (set_n_components) without re-solving.
    eigenvalues_full, explained_variance_ratio_full : (L,) tensors
        Variance carried by each component, and its fraction of the total.
    n_components : int
        Size K of the active basis.
    meta : dict
        Bookkeeping (n_snapshots, n_samples, wl_slice, normalization, centered).
    """

    # ----------------------------------------------------------------
    def __init__(self, mean, components_full, eigenvalues_full,
                 n_components, meta=None):
        self.mean_ = mean
        self.components_full = components_full
        self.eigenvalues_full = eigenvalues_full
        total = torch.clamp(eigenvalues_full.sum(), min=1e-30)
        self.explained_variance_ratio_full = eigenvalues_full / total
        self.n_components = int(n_components)
        self.meta = dict(meta or {})

    # ----------------------------------------------------------------
    @classmethod
    def fit(cls, data, n_snapshots=10, n_components=10, wl_slice=None,
            normalization=None, subsample=None, center=True,
            solve_dtype=torch.float64, store_dtype=torch.float32,
            device=None, seed=0):
        """Build the basis from `data`.

        Parameters
        ----------
        data : array or tensor
            The spectra you consider background.  Either (M, L) spectra or a
            >=3-D cube (t, ..., L) whose last `n_snapshots` frames are used.
            YOU choose what goes in here (raw frames vs reconstructed background).
        n_snapshots : int or None
            How many trailing frames of a cube to use (ignored for 2-D input).
        n_components : int
            Size K of the active basis.  All L components are computed and stored
            regardless, so this is cheap to change later.
        wl_slice : slice / (start, stop[, step]) / index array / None
            Optional wavelength crop, applied to the last axis.
        normalization : scalar / tensor / None
            Optional divisor (e.g. the Iqs continuum constant used by the fit).
        subsample : int or None
            If set and < M, randomly keep this many spectra (seeded) to bound
            memory/time of the covariance.
        center : bool
            Subtract the mean spectrum before the decomposition (recommended).
        solve_dtype, store_dtype : torch dtypes
            Precision of the eigen-solve and of the stored basis.
        device : torch device or None
            Where to place the stored basis (default: CPU).
        seed : int
            RNG seed for `subsample`.
        """
        X = _as_spectra(data, n_snapshots, wl_slice, normalization, dtype=solve_dtype)

        # drop non-finite spectra
        finite = torch.isfinite(X).all(dim=1)
        if not bool(finite.all()):
            X = X[finite]
        if X.shape[0] < 2:
            raise ValueError("need at least 2 finite spectra to build a basis")

        # optional random subsample of the rows
        if subsample is not None and subsample < X.shape[0]:
            g = torch.Generator().manual_seed(int(seed))
            idx = torch.randperm(X.shape[0], generator=g)[:subsample]
            X = X[idx]

        M, L = X.shape

        # mean-centre
        if center:
            mean = X.mean(dim=0)
        else:
            mean = torch.zeros(L, dtype=X.dtype, device=X.device)
        Xc = X - mean

        # covariance is only L x L (L ~ few hundred), so eigh is cheap and stable
        C = (Xc.t() @ Xc) / (M - 1)
        C = 0.5 * (C + C.t())  # symmetrise against round-off
        evals, evecs = torch.linalg.eigh(C)  # ascending; evecs columns are eigenvectors

        # order by descending variance; rows of `components` are the PCs
        evals = torch.flip(evals, dims=[0])
        components = torch.flip(evecs, dims=[1]).t().contiguous()
        evals = torch.clamp(evals, min=0.0)  # kill tiny negative round-off

        # deterministic sign: largest-magnitude entry of each component is positive
        row_max = components.abs().argmax(dim=1)
        signs = torch.sign(components[torch.arange(L, device=components.device), row_max])
        signs[signs == 0] = 1.0
        components = components * signs[:, None]

        meta = dict(n_snapshots=n_snapshots, n_samples=int(M),
                    wl_slice=wl_slice, normalization=normalization,
                    centered=bool(center))

        return cls(
            mean=mean.to(device=device, dtype=store_dtype),
            components_full=components.to(device=device, dtype=store_dtype),
            eigenvalues_full=evals.to(device=device, dtype=store_dtype),
            n_components=n_components,
            meta=meta,
        )

    # ---- active basis views -----------------------------------------
    @property
    def components_(self):
        """Active basis V : (K, L)."""
        return self.components_full[: self.n_components]

    # convenient aliases
    @property
    def V(self):
        return self.components_

    @property
    def mu(self):
        return self.mean_

    @property
    def explained_variance_ratio_(self):
        return self.explained_variance_ratio_full[: self.n_components]

    def set_n_components(self, k):
        """Change K in place (0 < k <= L). No re-solve needed."""
        k = int(k)
        if not (0 < k <= self.components_full.shape[0]):
            raise ValueError(f"k must be in 1..{self.components_full.shape[0]}")
        self.n_components = k
        return self

    def cumulative_variance(self):
        """Cumulative explained-variance fraction over all components -- use this
        in the notebook to pick K from a cut-off (e.g. 0.99)."""
        return torch.cumsum(self.explained_variance_ratio_full, dim=0)

    # ---- the two differentiable helpers -----------------------------
    def reconstruct(self, c):
        """Background from coefficients: b = mu + c @ V.

        c : (..., K) tensor (typically an nn.Parameter of shape (n_pixels, K)).
        Returns (..., L). Differentiable w.r.t. c; mu and V are constants that
        are moved onto c's device/dtype automatically.
        """
        k = c.shape[-1]
        V = self.components_full[:k].to(device=c.device, dtype=c.dtype)
        mu = self.mean_.to(device=c.device, dtype=c.dtype)
        return mu + c @ V

    def project(self, spec, k=None):
        """Least-squares coefficients of a spectrum onto the basis:
        c = (spec - mu) @ V^T  (exact, since V is orthonormal). Useful to
        initialise the coefficients from observed background spectra.

        spec : (..., L) tensor. Returns (..., k) with k = n_components by default.
        """
        k = self.n_components if k is None else int(k)
        V = self.components_full[:k].to(device=spec.device, dtype=spec.dtype)
        mu = self.mean_.to(device=spec.device, dtype=spec.dtype)
        return (spec - mu) @ V.t()

    # ---- housekeeping ----------------------------------------------
    def to(self, device=None, dtype=None):
        """Move/cast the stored basis in place."""
        self.mean_ = self.mean_.to(device=device, dtype=dtype)
        self.components_full = self.components_full.to(device=device, dtype=dtype)
        self.eigenvalues_full = self.eigenvalues_full.to(device=device, dtype=dtype)
        self.explained_variance_ratio_full = \
            self.explained_variance_ratio_full.to(device=device, dtype=dtype)
        return self

    def save(self, path):
        """Persist the full basis so it can be reused without re-solving."""
        torch.save(
            dict(mean=self.mean_,
                 components_full=self.components_full,
                 eigenvalues_full=self.eigenvalues_full,
                 n_components=self.n_components,
                 meta=self.meta),
            path,
        )

    @classmethod
    def load(cls, path, map_location="cpu"):
        d = torch.load(path, map_location=map_location, weights_only=False)
        return cls(mean=d["mean"], components_full=d["components_full"],
                   eigenvalues_full=d["eigenvalues_full"],
                   n_components=d["n_components"], meta=d.get("meta", {}))

    def __repr__(self):
        L = self.components_full.shape[1]
        cum = float(self.cumulative_variance()[self.n_components - 1])
        return (f"PCABackground(L={L}, K={self.n_components}, "
                f"M={self.meta.get('n_samples', '?')}, "
                f"cum_var@K={cum:.4f})")


# ====================================================================
def fit_background_basis(data, n_snapshots=10, n_components=10, **kwargs):
    """Thin functional wrapper around PCABackground.fit -- matches the name in
    step_1_outline. Returns a PCABackground whose .mu, .V and
    .explained_variance_ratio_ are the mean spectrum, the K eigenprofiles and
    their variance fractions."""
    return PCABackground.fit(data, n_snapshots=n_snapshots,
                             n_components=n_components, **kwargs)
