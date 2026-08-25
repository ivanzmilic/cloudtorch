#!/usr/bin/env python
"""npz_inversion_videos.py -- render nice videos from a packed inversion result.

Sibling of png_to_video.py: instead of stitching existing PNGs, this renders each
time frame with matplotlib and encodes the frames with OpenCV (same VideoWriter
path). It reads a single .npz holding the flattened (t*y*x, ...) arrays

    obs    (N, L)   observed spectra
    fits   (N, L)   fitted   spectra
    params (N, 10)  inferred cloud params [S,dtau,vlos,dv,loga] x 2 (canonical order)

and writes TWO videos:
    * <prefix>_fits.mp4    obs / fit / residual at a few wavelengths
    * <prefix>_params.mp4  the 8 free cloud-parameter maps (loga is frozen -> skipped)

KEY POINT (why it looks nice): every panel's colour limits are computed ONCE over
ALL time and held fixed for the whole clip, so colours/colourbars never jump between
frames. vlos panels use a symmetric seismic scale; obs & fit share a scale per
wavelength; residuals get a symmetric scale per wavelength.

The file stores no grid shape, so pass it with --shape Nt,Ny,Nx (default 32,116,130,
the full bordered MiHI FOV; N must equal Nt*Ny*Nx). Flatten order is (t, y, x) to
match the cloudtorch pipeline (data.reshape(Nt,Ny,Nx,L)).

Usage:
    python npz_inversion_videos.py                                   # defaults
    python npz_inversion_videos.py --npz /path/f.npz --shape 32,116,130 --fps 6
    python npz_inversion_videos.py --dwl 120 --prefix inv --fits-only
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


# ----------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--npz", default=DEFAULT_NPZ, help="packed obs/fits/params .npz")
    ap.add_argument("--shape", default="32,116,130", help="Nt,Ny,Nx (default: full MiHI FOV)")
    ap.add_argument("--outdir", default=HERE, help="where to write the videos (default: this folder)")
    ap.add_argument("--prefix", default="inversion", help="output filename prefix")
    ap.add_argument("--fps", type=float, default=6.0)
    ap.add_argument("--dwl", type=int, default=120, help="wavelength offset from core for the fit panels")
    ap.add_argument("--pct", type=float, default=2.0, help="colour-limit percentile (uses pct .. 100-pct)")
    ap.add_argument("--dpi", type=int, default=110)
    ap.add_argument("--fits-only", action="store_true")
    ap.add_argument("--params-only", action="store_true")
    args = ap.parse_args()

    Nt, Ny, Nx = (int(v) for v in args.shape.split(","))
    f = np.load(args.npz, allow_pickle=True)
    obs, fits, params = f["obs"], f["fits"], f["params"]
    N, L = obs.shape
    if Nt * Ny * Nx != N:
        raise SystemExit("shape %dx%dx%d = %d != N=%d in %s"
                         % (Nt, Ny, Nx, Nt * Ny * Nx, N, args.npz))
    obs = obs.reshape(Nt, Ny, Nx, L)
    fits = fits.reshape(Nt, Ny, Nx, L)
    params = params.reshape(Nt, Ny, Nx, 10)
    print("loaded %s | obs/fits %s | params %s" % (args.npz, obs.shape, params.shape))

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
            plim[nm] = sym(lo, hi) if cmap == "seismic" else (lo, hi)

        def render_param(t):
            fig, ax = plt.subplots(2, 4, figsize=(18, 8), dpi=args.dpi, constrained_layout=True)
            ax = ax.ravel()
            for k, (nm, j, cmap) in enumerate(PARAM_SPECS):
                lo, hi = plim[nm]
                h = ax[k].imshow(params[t, :, :, j].T, origin="lower", cmap=cmap, vmin=lo, vmax=hi)
                ax[k].set_title("%s   [%.2f, %.2f]" % (nm, lo, hi))
                ax[k].set_xticks([]); ax[k].set_yticks([])
                fig.colorbar(h, ax=ax[k], fraction=0.046, pad=0.02)
            fig.suptitle("inferred cloud parameters -- frame %d / %d (fixed scales; log a frozen)"
                         % (t, Nt - 1), fontsize=14)
            img = fig_to_bgr(fig); plt.close(fig); return img

        out = os.path.join(args.outdir, "%s_params.mp4" % args.prefix)
        print("PARAMS video (8 free cloud fields):")
        encode(render_param, Nt, out, args.fps)


if __name__ == "__main__":
    main()
