"""
given_background.py -- Step 3a of the develop/ plan (see instructions.md).

Loads the *pre-calculated* background used by composite_model.composite_synth_given:
the skglm quiet-Sun profile

        bdata = new_fit - new_extrafit                  (the incident intensity I_in)

from `mihi_rvm_reconstruction_averages_skglm.npz`, which lives in the SAME directory
as the data cube and has the SAME (t, y, x, lambda) shape. `new_extrafit` is the
extra (filament) absorption; removing it leaves the smooth profile behind the
filament -- exactly the I_in the two-cloud transfer needs. See docs/cloudtorch_summary.

The one hard requirement is ALIGNMENT: the given background must be carved with the
identical border + lambda crop, the identical (ts, ys, xs) fit window, and divided
by the identical Iqs as the observed data -- otherwise the transfer
I_out = I_in e^-tau + S(1 - e^-tau) sits at the wrong absolute level. This module
reuses the run's own `cfg["data"]` crop and `meta` (from fit_cube.load_data), so the
alignment is guaranteed by construction rather than reproduced by hand.

The reconstruction npz is stored UNCOMPRESSED, so its members are memory-mapped and
only the fit-window slice is paged in -- no 5.8 GB full-array load. (A compressed
file falls back to a plain np.load of one member at a time.)

This module does NO fitting and defines NO parameters: the given background is a
fixed tensor with zero degrees of freedom.
"""

import os
import zipfile

import numpy as np
import numpy.lib.format as _nf
import torch

__all__ = ["load_given_background", "load_given_background_window", "default_bkg_path"]

# The reconstruction file sits next to the data cube; these are its member keys.
_BKG_FILENAME = "mihi_rvm_reconstruction_averages_skglm.npz"
_KEY_FIT, _KEY_EXTRA = "new_fit", "new_extrafit"


# ====================================================================
def default_bkg_path(data_path):
    """Default given-background path: the reconstruction file in the SAME directory
    as the (resolved) data cube."""
    return os.path.join(os.path.dirname(data_path), _BKG_FILENAME)


# ====================================================================
def _read_npy_header(fileobj):
    """(shape, fortran, dtype) of an open .npy stream, leaving the read cursor at
    the first array byte (so fileobj.tell() then gives the header length)."""
    ver = _nf.read_magic(fileobj)
    if ver == (1, 0):
        return _nf.read_array_header_1_0(fileobj)
    if ver == (2, 0):
        return _nf.read_array_header_2_0(fileobj)
    # numpy>=1.9 private helper handles any version; use it if the above miss.
    return _nf._read_array_header(fileobj, ver)          # pragma: no cover


def _member_memmap(path, name):
    """Memory-map one member of an UNCOMPRESSED (stored) .npz as a read-only array.
    `name` is the logical key (NpzFile-style, no extension); the archive member is
    `name + '.npy'`. Returns None if the member is compressed (caller falls back)."""
    arcname = name + ".npy"
    z = zipfile.ZipFile(path)
    try:
        zi = z.getinfo(arcname)
        if zi.compress_type != zipfile.ZIP_STORED:
            return None                                  # compressed -> can't mmap
        # Absolute offset of the member's raw bytes = local-header end.
        with open(path, "rb") as fh:
            fh.seek(zi.header_offset)
            local = fh.read(30)                          # fixed local file header
            name_len = int.from_bytes(local[26:28], "little")
            extra_len = int.from_bytes(local[28:30], "little")
        member_off = zi.header_offset + 30 + name_len + extra_len
        # The member is itself a .npy: skip its header to reach the array data.
        with z.open(arcname) as f:
            shape, fortran, dtype = _read_npy_header(f)
            npy_hdr_len = f.tell()
    finally:
        z.close()
    return np.memmap(path, dtype=dtype, mode="r", offset=member_off + npy_hdr_len,
                     shape=shape, order="F" if fortran else "C")


def _member_sliced(path, name, YS, XS, lam, ts, ys, xs):
    """Return np.copy of member[name] carved to the fit window, in the SAME order
    load_data uses: border+lambda crop -> frame pick (ts) -> spatial crop (ys,xs)."""
    mm = _member_memmap(path, name)
    if mm is None:                                       # compressed fallback
        with np.load(path, allow_pickle=True) as f:
            if name not in f.files:
                raise KeyError("%r not in %s (has %s)" % (name, path, f.files))
            mm = f[name]
    sub = mm[:, YS, XS, lam][ts][:, ys, xs, :]           # ts (list) -> materialised
    out = np.ascontiguousarray(sub, dtype=np.float32).copy()
    del mm, sub
    return out


# ====================================================================
def load_given_background_window(bkg_path, ts, YS, XS, lam, ys, xs, Iqs,
                                 device=None, dtype=torch.float32):
    """Core loader from explicit slices (used directly by the step-3a notebook, and
    by the cfg/meta wrapper below). Carves bdata = (new_fit - new_extrafit) / Iqs to
    the SAME window as the observed data, in load_data's crop order.

    Parameters
    ----------
    bkg_path : str
        Path to the reconstruction .npz (default_bkg_path derives it from the data).
    ts : int or sequence of int
        Frame index/indices (a single int is allowed for a one-frame fit).
    YS, XS, lam : slice
        The border + wavelength crop applied FIRST (as in fit_cube.load_data).
    ys, xs : slice
        A further spatial crop applied within the bordered field (slice(None) for
        the whole bordered field, e.g. the single-frame notebook).
    Iqs : float
        The SAME red-continuum normalisation used for the observed data.
    device, dtype :
        Placement / precision of the returned tensor.

    Returns
    -------
    I_in : (nt*Cy*Cx, L) tensor
        The fixed incident background, rows in (t, y, x) order (matches obs_pt /
        grid_coords). A plain grad-free tensor -- feed straight to
        composite_synth_given.
    shape : tuple
        The (nt, Cy, Cx, L) shape before flattening, for an alignment check.
    """
    if not os.path.exists(bkg_path):
        raise FileNotFoundError("given background not found: %s" % bkg_path)
    ts = [int(ts)] if np.isscalar(ts) else list(ts)

    fit = _member_sliced(bkg_path, _KEY_FIT, YS, XS, lam, ts, ys, xs)
    extra = _member_sliced(bkg_path, _KEY_EXTRA, YS, XS, lam, ts, ys, xs)
    bdata = (fit - extra) / float(Iqs)                   # (nt, Cy, Cx, L)
    del fit, extra

    I_in = torch.from_numpy(bdata.reshape(-1, bdata.shape[-1])).to(device=device, dtype=dtype)
    return I_in, bdata.shape


# ====================================================================
def load_given_background(cfg, meta, device=None, dtype=torch.float32):
    """Load the pre-calculated background, aligned to the run's fit window.

    Thin wrapper over load_given_background_window that pulls the crop from
    cfg["data"] and the fit window from `meta` (fit_cube.load_data), so alignment
    with the observed data is guaranteed by construction.

    Parameters
    ----------
    cfg : dict
        Uses cfg["data"]: `border`, `lam`, `path`, and optional `bkg_path`
        (defaults to the reconstruction file next to the data cube).
    meta : dict
        The meta from fit_cube.load_data: uses `ts`, `ys`, `xs`, `Iqs`, and
        (`Nt`, `Cy`, `Cx`, `L`) for the alignment check.

    Returns
    -------
    I_in : (Nt*Cy*Cx, L) tensor -- see load_given_background_window.
    """
    d = cfg["data"]
    data_path = _resolve_data_path(d["path"])
    bkg_path = d.get("bkg_path") or default_bkg_path(data_path)

    b = int(d["border"])
    YS = slice(b, -b) if b else slice(None)
    XS = slice(b, -b) if b else slice(None)
    lam = slice(d["lam"][0], d["lam"][1])
    ys = slice(meta["ys"][0], meta["ys"][1])
    xs = slice(meta["xs"][0], meta["xs"][1])

    I_in, shape = load_given_background_window(
        bkg_path, meta["ts"], YS, XS, lam, ys, xs, meta["Iqs"], device, dtype)

    want = (meta["Nt"], meta["Cy"], meta["Cx"], meta["L"])
    if shape != want:
        raise ValueError("given background shape %s != fit window %s -- check the "
                         "crop/frame alignment" % (shape, want))
    return I_in


# ====================================================================
def _resolve_data_path(path):
    """First existing path among a str or list (mirrors fit_cube.resolve_path, kept
    local so this module imports standalone in a notebook)."""
    cands = path if isinstance(path, (list, tuple)) else [path]
    for p in cands:
        if os.path.exists(p):
            return p
    raise FileNotFoundError("no data file found among: %s" % (cands,))
