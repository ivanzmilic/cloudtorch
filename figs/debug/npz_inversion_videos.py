#!/usr/bin/env python
"""npz_inversion_videos.py -- render nice videos from a packed inversion result.

Sibling of png_to_video.py: instead of stitching existing PNGs, this renders each
time frame with matplotlib and encodes the frames with OpenCV (same VideoWriter
path). It reads a single .npz holding the arrays

    obs    (Nt, Ny, Nx, L)    observed spectra
    fits   (Nt, Ny, Nx, L)    fitted   spectra
    params (Nt, Ny, Nx, 10)   inferred cloud params [S,dtau,vlos,dv,loga] x 2 (canonical)

Legacy files that store these flattened as (t*y*x, ...) are still accepted (see --shape).

and writes TWO videos:
    * <prefix>_fits.mp4    obs / fit / residual at a few wavelengths
    * <prefix>_params.mp4  the 8 free cloud-parameter maps (loga is frozen -> skipped)

KEY POINT (why it looks nice): every panel's colour limits are computed ONCE over
ALL time and held fixed for the whole clip, so colours/colourbars never jump between
frames. vlos panels use a symmetric seismic scale; obs & fit share a scale per
wavelength; residuals get a symmetric scale per wavelength.

OPTIONAL tau-weighting (--tau-weight-vlos): a cloud's velocity is only meaningful where
that cloud has opacity, so fade each vlos map by its own dtau. The weight is the
absorption fraction w = 1 - exp(-dtau/tau_scale): 0 where dtau=0, saturating at 1 for
large tau (never blows up). Two looks: "alpha" (default) fades optically-thin pixels
toward a neutral grey; "scale" multiplies vlos by w so thin pixels go to 0 (white on
the seismic map). Only the vlos1/vlos2 panels are touched.

The grid shape is read straight from the file: the arrays are stored as full
(Nt, Ny, Nx, ...) cubes. Only a legacy file whose arrays are flat (t*y*x, ...) needs
--shape Nt,Ny,Nx (flatten order (t, y, x); N must equal Nt*Ny*Nx); --shape is ignored
when the arrays already carry their shape.

Usage:
    python npz_inversion_videos.py                                   # defaults
    python npz_inversion_videos.py --npz /path/f.npz --fps 6
    python npz_inversion_videos.py --dwl 120 --prefix inv --fits-only
    python npz_inversion_videos.py --params-only --tau-weight-vlos --prefix inversion_tauw
    python npz_inversion_videos.py --shape 32,116,130 --npz legacy_flat.npz   # legacy
"""
import os
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_NPZ = "/dat/milic/MiHI_filament/inv_results/fits_and_params.npz"

# canonical cloud columns; the 8 shown (loga1/loga2 are frozen -> constant, skipped)
PARAM_SPECS = [("S1", 0, "viridis"), ("dtau1", 1, "viridis"), ("vlos1", 2, "seismic"), ("dv1", 3, "magma"),
               ("S2", 5, "viridis"), ("dtau2", 6, "viridis"), ("vlos2", 7, "seismic"), ("dv2", 8, "magma")]

# which dtau column weights each vlos column (a cloud's velocity is only meaningful where
# that same cloud has opacity): vlos1(col 2) <- dtau1(col 1); vlos2(col 7) <- dtau2(col 6)
VLOS_TAU = {2: 1, 7: 6}


# ----------------------------------------------------------------------- helpers
def fig_to_bgr(fig):
    """Rasterise an Agg figure to a (H, W, 3) uint8 BGR array for cv2."""
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())        # (H, W, 4) uint8
    return cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)


def open_writer(path, fps, w, h):
    ext = os.path.splitext(path)[1].lower()
    fourcc = cv2.VideoWriter_fourcc(*("MJPG" if ext == ".avi" else "mp4v"))
    vw = cv2.VideoWriter(path, fourcc, fps, (w, h))
    if not vw.isOpened():
        raise SystemExit("VideoWriter failed to open %s (codec/extension issue)" % path)
    return vw


def encode(render_frame, n_frames, out_path, fps):
    """Render frame 0 to fix the size, then encode all frames to out_path."""
    first = render_frame(0)
    h, w = first.shape[:2]
    w -= w % 2; h -= h % 2                              # even dims for the encoder
    vw = open_writer(out_path, fps, w, h)
    for t in range(n_frames):
        im = first if t == 0 else render_frame(t)
        if im.shape[1] != w or im.shape[0] != h:
            im = cv2.resize(im, (w, h), interpolation=cv2.INTER_AREA)
        vw.write(im)
        print("  + frame %d/%d" % (t + 1, n_frames))
    vw.release()
    print("wrote %s  (%d frames, %.1f fps, %dx%d)" % (out_path, n_frames, fps, w, h))


def sym(lo, hi):
    m = max(abs(lo), abs(hi)); return -m, m


def opacity_weight(dtau, tau_scale):
    """Cloud-visibility weight in [0, 1): 0 where dtau=0, rising toward 1 as dtau grows.

    w = 1 - exp(-dtau / tau_scale) is the fraction of light the cloud absorbs, so it
    saturates smoothly at 1 for large tau instead of running to infinity. tau_scale is
    the soft-knee: w ~= 0.63 at dtau = tau_scale."""
    return 1.0 - np.exp(-np.maximum(dtau, 0.0) / float(tau_scale))


def to_cube(a, name, legacy_shape):
    """Return array a as (Nt, Ny, Nx, C).

    Modern files already store 4-D cubes -> passed straight through and the shape is
    taken from the file. A legacy flat (t*y*x, C) array is reshaped with legacy_shape
    (Nt, Ny, Nx), which must be given for that case."""
    if a.ndim == 4:
        return a
    if a.ndim == 2:
        if legacy_shape is None:
            raise SystemExit("%s is flat %s and no --shape Nt,Ny,Nx was given" % (name, a.shape))
        Nt, Ny, Nx = legacy_shape
        if Nt * Ny * Nx != a.shape[0]:
            raise SystemExit("--shape %dx%dx%d = %d != %s rows %d in the file"
                             % (Nt, Ny, Nx, Nt * Ny * Nx, name, a.shape[0]))
        return a.reshape(Nt, Ny, Nx, a.shape[-1])
    raise SystemExit("%s has unexpected ndim %d (want 4-D cube or 2-D flat)" % (name, a.ndim))


# ----------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--npz", default=DEFAULT_NPZ, help="packed obs/fits/params .npz")
    ap.add_argument("--shape", default="32,116,130",
                    help="Nt,Ny,Nx -- only for legacy flat (t*y*x,...) files; "
                         "ignored when the .npz already stores full cubes")
    ap.add_argument("--outdir", default=HERE, help="where to write the videos (default: this folder)")
    ap.add_argument("--prefix", default="inversion", help="output filename prefix")
    ap.add_argument("--fps", type=float, default=6.0)
    ap.add_argument("--dwl", type=int, default=120, help="wavelength offset from core for the fit panels")
    ap.add_argument("--pct", type=float, default=2.0, help="colour-limit percentile (uses pct .. 100-pct)")
    ap.add_argument("--dpi", type=int, default=110)
    ap.add_argument("--fits-only", action="store_true")
    ap.add_argument("--params-only", action="store_true")
    # --- vlos tau-weighting: down-weight velocity where the cloud is optically thin ---
    ap.add_argument("--tau-weight-vlos", action="store_true",
                    help="weight each vlos panel by its cloud's opacity w = 1 - exp(-dtau/tau_scale)")
    ap.add_argument("--tau-weight-mode", choices=("alpha", "scale"), default="alpha",
                    help="alpha: fade thin pixels toward --tau-fade-color; "
                         "scale: shrink vlos toward 0 (fades to white on seismic)")
    ap.add_argument("--tau-scale", type=float, default=1.0,
                    help="opacity soft-knee for the weight (w ~= 0.63 at dtau = tau_scale)")
    ap.add_argument("--tau-fade-color", default="0.8",
                    help="matplotlib colour shown through fully-faded pixels (alpha mode)")
    args = ap.parse_args()

    legacy_shape = tuple(int(v) for v in args.shape.split(",")) if args.shape else None
    f = np.load(args.npz, allow_pickle=True)

    # shape comes straight from the params cube; obs/fits only loaded if we render them
    # (they are the big arrays -- skip the read entirely in --params-only mode)
    params = to_cube(f["params"], "params", legacy_shape)
    Nt, Ny, Nx = params.shape[:3]

    obs = fits = None
    L = None
    if not args.params_only:
        obs = to_cube(f["obs"], "obs", legacy_shape)
        fits = to_cube(f["fits"], "fits", legacy_shape)
        if obs.shape[:3] != (Nt, Ny, Nx) or fits.shape[:3] != (Nt, Ny, Nx):
            raise SystemExit("obs/fits grid %s / %s != params grid %s"
                             % (obs.shape[:3], fits.shape[:3], (Nt, Ny, Nx)))
        L = obs.shape[-1]
    print("loaded %s | grid Nt=%d Ny=%d Nx=%d | params %s%s"
          % (args.npz, Nt, Ny, Nx, params.shape,
             "" if obs is None else " | obs/fits L=%d" % L))

    os.makedirs(args.outdir, exist_ok=True)
    plo, phi = args.pct, 100.0 - args.pct

    # ============================ FITS VIDEO ============================
    if not args.params_only:
        core = L // 2
        wl_idx = [max(core - args.dwl, 0), core, min(core + args.dwl, L - 1)]
        col_lab = ["core%+d" % (i - core) if i != core else "core" for i in wl_idx]
        # fixed limits over ALL time+space: obs&fit share per wavelength; resid symmetric per wl
        of_lim, res_lim = [], []
        for ii in wl_idx:
            stack = np.concatenate([obs[..., ii].ravel(), fits[..., ii].ravel()])
            of_lim.append((np.nanpercentile(stack, plo), np.nanpercentile(stack, phi)))
            r = (obs[..., ii] - fits[..., ii]).ravel()
            res_lim.append(sym(np.nanpercentile(r, plo), np.nanpercentile(r, phi)))

        def render_fit(t):
            fig, ax = plt.subplots(3, 3, figsize=(11, 9.5), dpi=args.dpi, constrained_layout=True)
            for j, ii in enumerate(wl_idx):
                lo, hi = of_lim[j]; rlo, rhi = res_lim[j]
                o = obs[t, :, :, ii].T; ft = fits[t, :, :, ii].T; rr = o - ft
                rlo = -0.1; rhi = 0.1  # fixed residual limits for better visibility
                h0 = ax[0, j].imshow(o,  origin="lower", cmap="magma",   vmin=lo,  vmax=hi)
                h1 = ax[1, j].imshow(ft, origin="lower", cmap="magma",   vmin=lo,  vmax=hi)
                h2 = ax[2, j].imshow(rr, origin="lower", cmap="seismic", vmin=rlo, vmax=rhi)
                ax[0, j].set_title(col_lab[j])
                for a in ax[:, j]:
                    a.set_xticks([]); a.set_yticks([])
                fig.colorbar(h1, ax=[ax[0, j], ax[1, j]], fraction=0.046, pad=0.02)
                fig.colorbar(h2, ax=ax[2, j], fraction=0.046, pad=0.02)
            for r, lab in enumerate(("obs", "fit", "obs - fit")):
                ax[r, 0].set_ylabel(lab, fontsize=12)
            fig.suptitle("inversion fits -- frame %d / %d" % (t, Nt - 1), fontsize=13)
            img = fig_to_bgr(fig); plt.close(fig); return img

        out = os.path.join(args.outdir, "%s_fits.mp4" % args.prefix)
        print("FITS video (wavelengths %s):" % col_lab)
        encode(render_fit, Nt, out, args.fps)

    # ============================ PARAMS VIDEO ============================
    if not args.fits_only:
        # fixed per-panel limits over ALL time+space (seismic panels symmetric)
        plim = {}
        for nm, j, cmap in PARAM_SPECS:
            lo = float(np.nanpercentile(params[..., j], plo))
            hi = float(np.nanpercentile(params[..., j], phi))
            plim[nm] = sym(lo, hi) if (cmap == "seismic" or cmap == 'bwr') else (lo, hi)

        def render_param(t):
            fig, ax = plt.subplots(2, 4, figsize=(18, 8), dpi=args.dpi, constrained_layout=True)
            ax = ax.ravel()
            for k, (nm, j, cmap) in enumerate(PARAM_SPECS):
                lo, hi = plim[nm]
                field = params[t, :, :, j].T
                kw = dict(origin="lower", cmap=cmap, vmin=lo, vmax=hi)
                weighted = args.tau_weight_vlos and j in VLOS_TAU
                if weighted:
                    w = opacity_weight(params[t, :, :, VLOS_TAU[j]].T, args.tau_scale)
                    if args.tau_weight_mode == "scale":
                        field = field * w                     # velocity shrinks toward 0 (white)
                    else:                                     # alpha: fade thin pixels to grey
                        ax[k].set_facecolor(args.tau_fade_color)
                        kw["alpha"] = w
                h = ax[k].imshow(field, **kw)
                tag = r" ($\tau$-w)" if weighted else ""
                ax[k].set_title("%s%s   [%.2f, %.2f]" % (nm, tag, lo, hi))
                ax[k].set_xticks([]); ax[k].set_yticks([])
                fig.colorbar(h, ax=ax[k], fraction=0.046, pad=0.02)
            extra = (r" -- vlos $\tau$-weighted (w=1-e$^{-\tau/%.1f}$, %s)"
                     % (args.tau_scale, args.tau_weight_mode)) if args.tau_weight_vlos else ""
            fig.suptitle("inferred cloud parameters -- frame %d / %d (fixed scales; log a frozen)%s"
                         % (t, Nt - 1, extra), fontsize=14)
            img = fig_to_bgr(fig); plt.close(fig); return img

        out = os.path.join(args.outdir, "%s_params.mp4" % args.prefix)
        print("PARAMS video (8 free cloud fields):")
        encode(render_param, Nt, out, args.fps)


if __name__ == "__main__":
    main()
