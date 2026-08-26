#!/usr/bin/env python
"""params_3d_movie.py -- a PR / teaching 3D movie of the inverted filament.

Reads the SAME packed inversion result that feeds npz_inversion_videos.py (the
obs/fits/params .npz behind the inversion_tauw_params movie -- not rasterised PNGs,
so the 3D render stays crisp) and turns the two most telling parameters into a
position-position-velocity (PPV) point cloud that both evolves AND drifts in time:

    * horizontal plane   = the sky (x, y pixels of the FOV)
    * vertical axis      = line-of-sight velocity. SIGN CONVENTION: negative v_los is
                           UP (blueshift, material rising / approaching), positive is
                           DOWN (redshift, sinking / receding). Height = -v_los.
    * brightness (alpha) = optical depth dtau, a proxy for the amount of material:
                           w = 1 - exp(-dtau/tau_scale), so thin gas fades out and the
                           weight saturates at 1 (never blows up) for thick gas
    * colour             = v_los (blue = up/approaching, red = down/receding), so the
                           colourbar and the height axis tell the same story
    * horizontal drift   = the plane-of-sky motion of the threads, recovered as dense
                           optical flow of the dtau maps and used to ADVECT the points
                           between frames -- so filaments slide across the sky instead
                           of merely cross-fading (--no-advect to disable).

Both clouds are merged into one population, each pixel placed at its own velocity, so
the clip shows the filament's material spread through velocity space, deforming and
drifting frame to frame while the camera slowly orbits and bobs. Optically-thin points
vanish, so only real filaments are drawn.

This favours a nice-looking PR/teaching clip over strict accuracy: parameters are
interpolated in time for smooth motion (--sub), velocity limits are clipped to where
material actually lives, and the horizontal flow can be exaggerated (--flow-gain).

There are two looks: the default scientific render (labelled axes, colourbar, captions)
and a `--flashy` presentation mode -- a glowing neon point cloud with no axes, a full
camera orbit, and a bloom + vignette finish. Flashy trades physical fidelity for punch
and writes to a separate <prefix>_3d_flashy.mp4 so it never clobbers the science clip.

Usage:
    python params_3d_movie.py                                   # scientific, dark bg
    python params_3d_movie.py --sub 4 --spin-deg 60 --fps 20
    python params_3d_movie.py --flow-gain 1.8 --point-size 9    # punchier drift + gas
    python params_3d_movie.py --no-advect --light               # static grid, light bg
    python params_3d_movie.py --flashy                          # presentation: neon glow
    python params_3d_movie.py --flashy --bloom 2.2 --fps 24     # extra glow, smoother spin
"""
import os
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize, LinearSegmentedColormap
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers the 3d projection)
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_NPZ = "/dat/milic/MiHI_filament/inv_results/fits_and_params.npz"

# (dtau column, vlos column) for the two clouds in the canonical 10-wide params row
CLOUDS = [(1, 2), (6, 7)]

# a punchy neon velocity palette for the flashy/presentation mode: electric cyan for
# material rising (up / blueshift), a bright lavender core near rest, hot magenta-red
# for material sinking (down / redshift). Saturated so points glow over deep black.
NEON = LinearSegmentedColormap.from_list("neon_vel", [
    (0.00, "#16e6ff"),   # up  : electric cyan
    (0.30, "#2f8bff"),   # blue
    (0.50, "#eae6ff"),   # near-rest: bright lavender-white
    (0.70, "#ff7a1c"),   # orange
    (1.00, "#ff2d6b"),   # down: hot magenta-red
])


# ----------------------------------------------------------------------- helpers
def fig_to_bgr(fig):
    """Rasterise an Agg figure to a (H, W, 3) uint8 BGR array for cv2."""
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    return cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)


def open_writer(path, fps, w, h):
    ext = os.path.splitext(path)[1].lower()
    fourcc = cv2.VideoWriter_fourcc(*("MJPG" if ext == ".avi" else "mp4v"))
    vw = cv2.VideoWriter(path, fourcc, fps, (w, h))
    if not vw.isOpened():
        raise SystemExit("VideoWriter failed to open %s (codec/extension issue)" % path)
    return vw


def opacity_weight(dtau, tau_scale):
    """Cloud-visibility weight in [0, 1): 0 where dtau=0, saturating at 1 for thick gas."""
    return 1.0 - np.exp(-np.maximum(dtau, 0.0) / float(tau_scale))


def thread_flow(a, b, winsize):
    """Dense optical flow (dx, dy) carrying opacity map a onto the next frame b.

    a, b are weight maps in [0,1]; scaled to 8-bit for Farneback, then the flow is
    lightly blurred so whole threads translate coherently rather than pixel-by-pixel."""
    pa = np.clip(a * 255.0, 0, 255).astype(np.uint8)
    pb = np.clip(b * 255.0, 0, 255).astype(np.uint8)
    flow = cv2.calcOpticalFlowFarneback(pa, pb, None,
                                        0.5, 3, winsize, 3, 5, 1.2, 0)  # (Ny,Nx,2)
    return cv2.GaussianBlur(flow, (0, 0), 2.0)


_VIGNETTE_CACHE = {}


def post_process(img, bloom, vignette):
    """Cheap 'flashy' finish: screen-blend a blurred bright pass (bloom) and darken the
    corners (vignette). Operates on the BGR uint8 frame; channel order is irrelevant."""
    f = img.astype(np.float32) / 255.0
    if bloom > 0:
        lum = f.max(axis=2, keepdims=True)
        bright = f * np.clip((lum - 0.12) / 0.88, 0.0, 1.0)   # only luminous points bloom
        g = 0.6 * cv2.GaussianBlur(bright, (0, 0), 7) + 0.5 * cv2.GaussianBlur(bright, (0, 0), 18)
        glow = np.clip(g * bloom, 0.0, 1.0)
        f = 1.0 - (1.0 - f) * (1.0 - glow)                    # screen blend
    if vignette > 0:
        h, w = f.shape[:2]
        mask = _VIGNETTE_CACHE.get((h, w, vignette))
        if mask is None:
            yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
            r = np.hypot((xx - w / 2.0) / (w / 2.0), (yy - h / 2.0) / (h / 2.0))
            mask = np.clip(1.0 - vignette * r * r, 0.0, 1.0)[..., None]
            _VIGNETTE_CACHE[(h, w, vignette)] = mask
        f = f * mask
    return (np.clip(f, 0.0, 1.0) * 255.0).astype(np.uint8)


def style_axes(ax, fg, dark):
    """Light-on-dark (or dark-on-light) 3D axes: translucent panes, faint grid."""
    pane = (1, 1, 1, 0.03) if dark else (0, 0, 0, 0.02)
    edge = (1, 1, 1, 0.10) if dark else (0, 0, 0, 0.12)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        try:
            axis.set_pane_color(pane)
            axis.pane.set_edgecolor(edge)
        except Exception:
            pass
        try:
            axis.line.set_color(fg)
            axis._axinfo["grid"].update(color=edge, linewidth=0.5)
        except Exception:
            pass
    ax.tick_params(colors=fg)
    for lab in (ax.xaxis.label, ax.yaxis.label, ax.zaxis.label):
        lab.set_color(fg)


# ----------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--npz", default=DEFAULT_NPZ, help="packed obs/fits/params .npz")
    ap.add_argument("--shape", default="32,116,130", help="Nt,Ny,Nx (default: full MiHI FOV)")
    ap.add_argument("--outdir", default=HERE, help="where to write the movie (default: this folder)")
    ap.add_argument("--prefix", default="inversion", help="output filename prefix -> <prefix>_3d.mp4")
    ap.add_argument("--fps", type=float, default=15.0)
    ap.add_argument("--sub", type=int, default=4, help="interpolated sub-frames per time step (smoothness)")
    ap.add_argument("--tau-scale", type=float, default=1.0, help="opacity soft-knee (w~=0.63 at dtau=tau_scale)")
    ap.add_argument("--min-w", type=float, default=0.06, help="drop points thinner than this weight")
    ap.add_argument("--alpha-gain", type=float, default=1.0, help="overall opacity of the material points")
    ap.add_argument("--point-size", type=float, default=8.0)
    ap.add_argument("--cmap", default="bwr", help="diverging colour map for v_los (height & colour)")
    ap.add_argument("--pct", type=float, default=1.0, help="velocity-limit percentile over material pixels")
    ap.add_argument("--zpad", type=float, default=0.05, help="padding above/below the velocity range")
    ap.add_argument("--elev", type=float, default=24.0, help="camera elevation (deg)")
    ap.add_argument("--azim0", type=float, default=-60.0, help="starting camera azimuth (deg)")
    ap.add_argument("--spin-deg", type=float, default=45.0, help="total azimuth swept across the clip")
    ap.add_argument("--bob-deg", type=float, default=3.0, help="elevation bob amplitude (deg)")
    ap.add_argument("--floor-alpha", type=float, default=0.15, help="opacity of the floor shadow (0 = off)")
    # --- horizontal thread motion (plane-of-sky drift from the dtau maps) ---
    ap.add_argument("--no-advect", dest="advect", action="store_false",
                    help="disable optical-flow advection (points stay on the fixed grid)")
    ap.add_argument("--flow-gain", type=float, default=1.4, help="exaggerate the horizontal drift (1 = as measured)")
    ap.add_argument("--flow-winsize", type=int, default=21, help="Farneback window (larger = smoother flow)")
    ap.set_defaults(advect=True)
    ap.add_argument("--dpi", type=int, default=130)
    ap.add_argument("--light", action="store_true", help="light background instead of the default dark")
    # --- flashy / presentation look (glow, neon, no axes, bloom + vignette) ---
    ap.add_argument("--flashy", action="store_true",
                    help="presentation mode: glowing neon point cloud, no axes, bloom + vignette")
    ap.add_argument("--bloom", type=float, default=1.6, help="flashy: glow/bloom strength (0 = off)")
    ap.add_argument("--vignette", type=float, default=0.4, help="flashy: corner darkening (0 = off)")
    ap.add_argument("--title", default="MiHI filament · threads in motion",
                    help="flashy: small informal title (empty string = none)")
    args = ap.parse_args()

    # flashy overrides only the knobs the user left at their scientific defaults
    if args.flashy:
        if args.cmap == "bwr":
            args.cmap = None                 # -> NEON, resolved below
        if args.spin_deg == 45.0:
            args.spin_deg = 360.0            # a full, loopable orbit
        if args.flow_gain == 1.4:
            args.flow_gain = 2.4             # exaggerate the horizontal slide
        if args.bob_deg == 3.0:
            args.bob_deg = 6.0
        args.light = False                   # flashy is always dark

    Nt, Ny, Nx = (int(v) for v in args.shape.split(","))
    f = np.load(args.npz, allow_pickle=True)
    params = f["params"]
    N = params.shape[0]
    if Nt * Ny * Nx != N:
        raise SystemExit("shape %dx%dx%d = %d != N=%d in %s"
                         % (Nt, Ny, Nx, Nt * Ny * Nx, N, args.npz))
    params = params.reshape(Nt, Ny, Nx, params.shape[-1])
    print("loaded %s | params %s" % (args.npz, params.shape))

    # per-cloud fields over time: (2, Nt, Ny, Nx)
    W = np.stack([opacity_weight(params[..., jt], args.tau_scale) for jt, _ in CLOUDS])
    V = np.stack([params[..., jv] for _, jv in CLOUDS])

    # velocity limits: clip to where material actually lives (empty sky is unconstrained)
    mat = V[W > args.min_w]
    vmax = float(np.nanpercentile(np.abs(mat), 100.0 - args.pct)) if mat.size else float(np.nanmax(np.abs(V)))
    vmax = max(vmax, 1e-3)
    zpad = args.zpad * vmax
    norm = Normalize(-vmax, vmax)
    cmap = NEON if args.cmap is None else plt.get_cmap(args.cmap)
    print("velocity axis +/- %.1f km/s over %d material pixels" % (vmax, mat.size))

    # sky coords, shared by both clouds (cloud-major flatten: [c0 pixels..., c1 pixels...])
    X, Y = np.meshgrid(np.arange(Nx), np.arange(Ny))            # (Ny, Nx)

    # dense optical flow of the dtau maps -> the plane-of-sky drift of the threads
    flows = None
    if args.advect and Nt > 1:
        flows = np.zeros((2, Nt - 1, Ny, Nx, 2), np.float32)
        for c in range(2):
            for t in range(Nt - 1):
                flows[c, t] = thread_flow(W[c, t], W[c, t + 1], args.flow_winsize)
        print("optical flow: |drift| median %.2f px, 95th %.2f px/frame"
              % (np.median(np.hypot(flows[..., 0], flows[..., 1])),
                 np.percentile(np.hypot(flows[..., 0], flows[..., 1]), 95)))

    dark = not args.light
    bg = ("#04060a" if args.flashy else "#0a0d12") if dark else "white"
    fg = "0.86" if dark else "0.15"

    total = (Nt - 1) * args.sub + 1
    os.makedirs(args.outdir, exist_ok=True)

    def fields_at(of):
        """Time-interpolated + optical-flow-advected material for output frame `of`."""
        t0 = min(of // args.sub, Nt - 1)
        t1 = min(t0 + 1, Nt - 1)
        fr = (of - t0 * args.sub) / args.sub if args.sub else 0.0
        w = (1 - fr) * W[:, t0] + fr * W[:, t1]                 # (2, Ny, Nx)
        v = (1 - fr) * V[:, t0] + fr * V[:, t1]
        xs, ys = [], []
        for c in range(2):
            if flows is not None and t0 < Nt - 1:
                fl = flows[c, t0]
                xs.append((X + args.flow_gain * fr * fl[..., 0]).ravel())
                ys.append((Y + args.flow_gain * fr * fl[..., 1]).ravel())
            else:
                xs.append(X.ravel()); ys.append(Y.ravel())
        return (np.concatenate(xs), np.concatenate(ys),
                v.reshape(2, -1).ravel(), w.reshape(2, -1).ravel(), t0 + fr)

    def render(of):
        xs, ys, vel, wk, tcont = fields_at(of)
        keep = wk > args.min_w
        xs, ys, vel, wk = xs[keep], ys[keep], vel[keep], wk[keep]
        zs = -vel                                              # sign: negative v_los = up

        rgba = cmap(norm(vel))
        rgba[:, 3] = np.clip(wk * args.alpha_gain, 0.0, 1.0)
        size = args.point_size * (0.45 + 0.55 * wk)

        fig = plt.figure(figsize=(9, 8), dpi=args.dpi)
        fig.patch.set_facecolor(bg)
        ax = fig.add_axes([0.0, 0.05, 0.86, 0.88], projection="3d")  # fill the frame
        ax.set_facecolor(bg)

        zfloor = -vmax - zpad
        if args.floor_alpha > 0:
            fl = np.zeros((xs.size, 4)); fl[:, :3] = 0.55 if dark else 0.35
            fl[:, 3] = np.clip(wk * args.floor_alpha, 0.0, 0.5)
            ax.scatter(xs, ys, np.full_like(zs, zfloor), c=fl,
                       s=size * 0.7, edgecolors="none", depthshade=False)

        ax.scatter(xs, ys, zs, c=rgba, s=size, edgecolors="none", depthshade=False)

        ax.set_xlim(0, Nx); ax.set_ylim(0, Ny); ax.set_zlim(zfloor, vmax + zpad)
        ax.set_box_aspect((1.0, Ny / Nx, 0.62))
        az = args.azim0 + args.spin_deg * (of / max(total - 1, 1))
        el = args.elev + args.bob_deg * np.sin(2 * np.pi * of / max(total - 1, 1))
        ax.view_init(elev=el, azim=az)

        ax.set_xlabel("x  (sky)", fontsize=10, labelpad=-8)
        ax.set_ylabel("y  (sky)", fontsize=10, labelpad=-8)
        # z is the velocity ruler, but the colourbar carries the numbers -> keep it clean
        ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_zticklabels([])
        style_axes(ax, fg, dark)

        # colourbar: invert so blue (up / approaching) sits at the TOP, matching the scene
        sm = ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
        cax = fig.add_axes([0.885, 0.30, 0.02, 0.40])
        cb = fig.colorbar(sm, cax=cax); cax.invert_yaxis()
        cb.set_label(r"$v_\mathrm{los}$  (km/s)", color=fg, fontsize=10)
        cax.yaxis.set_tick_params(color=fg)
        plt.setp(plt.getp(cax, "yticklabels"), color=fg)
        cb.outline.set_edgecolor(fg)
        cax.text(0.5, 1.05, "up · rising", transform=cax.transAxes,
                 ha="center", va="bottom", color=fg, fontsize=8.5)
        cax.text(0.5, -0.05, "down · sinking", transform=cax.transAxes,
                 ha="center", va="top", color=fg, fontsize=8.5)

        fig.text(0.5, 0.965, "MiHI filament  --  3D structure and motion",
                 ha="center", color=fg, fontsize=15, weight="bold")
        fig.text(0.5, 0.925,
                 "height & colour = line-of-sight (up/down) velocity     brightness = amount of material",
                 ha="center", color=fg, fontsize=9.5)
        fig.text(0.5, 0.028,
                 "threads drift horizontally with their plane-of-sky motion (optical flow of the optical-depth maps)",
                 ha="center", color=("0.6" if dark else "0.4"), fontsize=8.5, style="italic")
        fig.text(0.90, 0.10, "t = %4.1f / %d" % (tcont, Nt - 1),
                 ha="right", color=fg, fontsize=11, family="monospace")

        img = fig_to_bgr(fig); plt.close(fig); return img

    # glow layers for flashy mode: (size multiple, alpha multiple), faint wide halo -> bright core
    GLOW = [(7.0, 0.05), (3.4, 0.11), (1.6, 0.26), (0.7, 0.85)]

    def render_flashy(of):
        """Immersive presentation render: a glowing neon nebula, no axes, bloom + vignette."""
        xs, ys, vel, wk, tcont = fields_at(of)
        keep = wk > args.min_w
        xs, ys, vel, wk = xs[keep], ys[keep], vel[keep], wk[keep]
        zs = -vel                                              # sign: negative v_los = up
        rgb = cmap(norm(vel))[:, :3]

        fig = plt.figure(figsize=(9, 9), dpi=args.dpi)
        fig.patch.set_facecolor(bg)
        ax = fig.add_axes([0.0, 0.0, 1.0, 1.0], projection="3d")  # fill the whole frame
        ax.set_facecolor(bg)

        base = args.point_size * (0.4 + 0.6 * wk)
        for smul, amul in GLOW:                                # stacked halos fake additive glow
            rgba = np.empty((xs.size, 4)); rgba[:, :3] = rgb
            rgba[:, 3] = np.clip(wk * amul * args.alpha_gain, 0.0, 1.0)
            ax.scatter(xs, ys, zs, c=rgba, s=base * smul, edgecolors="none", depthshade=False)

        ax.set_xlim(0, Nx); ax.set_ylim(0, Ny); ax.set_zlim(-vmax - zpad, vmax + zpad)
        ax.set_box_aspect((1.0, Ny / Nx, 0.62))
        ax.set_axis_off()                                      # no axes / grid / panes at all
        az = args.azim0 + args.spin_deg * (of / max(total - 1, 1))
        el = args.elev + args.bob_deg * np.sin(2 * np.pi * of / max(total - 1, 1))
        ax.view_init(elev=el, azim=az)

        if args.title:
            fig.text(0.5, 0.945, args.title, ha="center", va="center",
                     color="0.92", fontsize=17, weight="bold")
        # tiny informal legend instead of a colourbar
        fig.text(0.045, 0.085, "●", color=cmap(0.04), fontsize=13, va="center")
        fig.text(0.075, 0.085, "rising", color="0.8", fontsize=10.5, va="center")
        fig.text(0.045, 0.045, "●", color=cmap(0.96), fontsize=13, va="center")
        fig.text(0.075, 0.045, "sinking", color="0.8", fontsize=10.5, va="center")

        img = fig_to_bgr(fig); plt.close(fig)
        return post_process(img, args.bloom, args.vignette)

    draw = render_flashy if args.flashy else render

    # --- encode ---
    first = draw(0)
    h, w = first.shape[:2]; w -= w % 2; h -= h % 2
    suffix = "_3d_flashy" if args.flashy else "_3d"
    out = os.path.join(args.outdir, "%s%s.mp4" % (args.prefix, suffix))
    vw = open_writer(out, args.fps, w, h)
    for of in range(total):
        im = first if of == 0 else draw(of)
        if im.shape[1] != w or im.shape[0] != h:
            im = cv2.resize(im, (w, h), interpolation=cv2.INTER_AREA)
        vw.write(im)
        print("  + frame %d/%d" % (of + 1, total))
    vw.release()
    print("wrote %s  (%d frames, %.1f fps, %dx%d)" % (out, total, args.fps, w, h))


if __name__ == "__main__":
    main()
