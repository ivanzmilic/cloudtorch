#!/usr/bin/env python
"""
fit_cube.py -- Step 6 of the develop/ plan: deployment.

Turns the validated step-1..5 stack (a fitted PCA background (+) two absorbing
clouds, fit by a Fourier neural field with explicit regularisation) into ONE
configurable, resumable command-line script that fits the full (t, y, x, lambda)
cube on GPU.

No new physics: `ParameterField`, `composite_synth`, `chi2_per_pixel` and
`regularization_loss` are reused verbatim from steps 1-5. This file only adds the
driver: config, data loading, the training loop, checkpoint/resume, saving the
fitted parameter maps, and an optional diagnostic report.

The Fourier bandwidths are defined at a reference window and SCALED with the
domain (see `sigmas_for`), so the step-4/5 tuning transfers as the crop / frame
count grow (step-6 outline sec.4). With the baked-in defaults, `python fit_cube.py`
reproduces the tuned step-5 notebook.

Usage
-----
    python fit_cube.py                                   # defaults (== step-5 notebook)
    python fit_cube.py --config my.json --out run1       # load a config, run
    python fit_cube.py --set data.crop=null train.n_iters=20000 --out full
    python fit_cube.py --resume run1/checkpoint.pt        # continue a run
    python fit_cube.py --resume run1/checkpoint.pt --no-train --report
    python fit_cube.py --dump-config default.json         # write the default config

See STEP6_HOWTO.md for the full guide.
"""
import os
import sys
import json
import time
import copy
import argparse

import numpy as np
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR

# make the develop/ modules (and the parent repo, for cloud_model_torch) importable
_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, os.path.dirname(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from pca_background import fit_background_basis, PCABackground          # noqa: E402
from pinn_model import (ParameterField, grid_coords,                    # noqa: E402
                        composite_synth, chi2_per_pixel)
from regularizers import regularization_loss, expand_group_config       # noqa: E402

CLOUD_NAMES = ['S1', 'dtau1', 'vlos1', 'dv1', 'loga1',
               'S2', 'dtau2', 'vlos2', 'dv2', 'loga2']


# ==========================================================================
# Default configuration == the tuned step-5 values. EVERY knob lives here; a
# --config file and --set overrides are deep-merged on top.
# ==========================================================================
DEFAULT_CFG = {
    "data": {
        "path": ["/dat/milic/MiHI_filament/mihi_all_data.npz",
                 "/home/milic/data/MiHI_halpha_filament/mihi_all_data.npz"],
        "n_basis": 10, "k": 6, "lam": [100, 500], "border": 15,
        "t0": 13, "n_t": 16, "t_start": None, "crop": 64,   # n_t/crop=null -> all frames/full field; t_start=null -> centre on t0
    },
    "field": {
        "n_freq": 128, "sigma_xy_ref": 4.0, "sigma_t_ref": 2.0,
        "crop_ref": 40, "n_t_ref": 4,         # sigmas scale with domain relative to these
        "width": 256, "depth": 4, "s_max": 0.7, "dv_floor": 8.0, "seed": 0,
    },
    "train": {
        "n_iters": 5000, "lr": 5e-3, "grad_clip": 1.0, "batch": 4096,
        "device": "auto",                     # "auto" | "cuda" | "cpu"
    },
    "reg": {
        "bkg":  {"w_xy": 0.2, "w_t": 0.2},
        "S":    {"w_xy": 1e-3, "w_t": 1e-3, "lo": 0.0, "w_lo": 1e-2},
        "dtau": {"w_xy": 1e-3, "w_t": 1e-3, "lo": 0.0, "w_lo": 1e-2},
        "vlos": {"w_xy": 1e-3, "w_t": 1e-3, "lo": -60.0, "hi": 60.0, "w_lo": 1e-2, "w_hi": 1e-2},
        "dv":   {"w_xy": 1e-3, "w_t": 1e-3, "lo": 8.0, "w_lo": 1e-2},
        "no_emission": {"weight": 50.0, "cap": 1.0},
        "anchor": {"weight": 0.1},
    },
    "io": {"out": "fit_out", "ckpt_every": 500},
}


# ==========================================================================
# Config helpers
# ==========================================================================
def deep_merge(base, over):
    """Recursively merge `over` onto a deepcopy of `base`."""
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        out[k] = deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def apply_overrides(cfg, pairs):
    """Apply `--set a.b.c=VAL` overrides in place. VAL is parsed as JSON when it
    can be (so 32, 0.2, null, true, [0,0] work), else kept as a string."""
    for pair in pairs or []:
        key, _, raw = pair.partition("=")
        try:
            val = json.loads(raw)
        except json.JSONDecodeError:
            val = raw
        d, parts = cfg, key.split(".")
        for p in parts[:-1]:
            d = d.setdefault(p, {})
        d[parts[-1]] = val
    return cfg


def resolve_device(cfg):
    d = cfg["train"]["device"]
    if d == "auto":
        d = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(d)


def resolve_path(path):
    cands = path if isinstance(path, (list, tuple)) else [path]
    for p in cands:
        if os.path.exists(p):
            return p
    raise FileNotFoundError("no data file found among: %s" % (cands,))


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ==========================================================================
# Data, basis, coordinates, field
# ==========================================================================
def load_data(cfg, device):
    """Load the cube, build the step-1 basis, carve the fit window, and build the
    filament-free anchor reference c_ref. Returns (obs_pt, c_ref, pca, xl, meta)."""
    d = cfg["data"]
    f = np.load(resolve_path(d["path"]), allow_pickle=True)
    wav = np.asarray(f["wav"])
    cube = f["data"]                                        # (t, y, x, lambda)
    lam = slice(d["lam"][0], d["lam"][1])
    b = d["border"]
    YS = slice(b, -b) if b else slice(None)
    XS = slice(b, -b) if b else slice(None)

    blk = np.copy(cube[-d["n_basis"]:, YS, XS, lam]).astype(np.float32)
    Iqs = float(blk[..., -20:].mean())
    fullsp = cube[:, YS, XS, lam]
    T, Ny, Nx = fullsp.shape[0], fullsp.shape[1], fullsp.shape[2]

    crop = d["crop"]
    if crop is None:                                        # full field
        ys, xs, Cy, Cx = slice(0, Ny), slice(0, Nx), Ny, Nx
    else:
        cy, cx = Ny // 2, Nx // 2
        ys = slice(cy - crop // 2, cy - crop // 2 + crop)
        xs = slice(cx - crop // 2, cx - crop // 2 + crop)
        Cy = Cx = crop

    if d["n_t"] is None:
        ts = list(range(T))
    elif d.get("t_start") is not None:                     # explicit window: first n_t from t_start
        ts = list(range(d["t_start"], d["t_start"] + d["n_t"]))
    else:                                                  # centred on t0
        ts = list(range(d["t0"] - d["n_t"] // 2, d["t0"] - d["n_t"] // 2 + d["n_t"]))
    if max(ts) >= T or min(ts) < 0:
        raise ValueError(f"frame window {ts[0]}..{ts[-1]} out of range for {T} frames")

    obs = np.copy(fullsp[ts][:, ys, xs, :]).astype(np.float32) / Iqs
    ref_spec = blk[:, ys, xs, :].mean(0) / Iqs             # filament-free per-pixel bkg
    del f, cube, fullsp

    Nt, Cy, Cx, L = obs.shape
    wl = np.copy(wav[lam]).astype(np.float32)
    xl = [wl, np.asarray([np.mean(wl)], dtype=np.float32)]
    pca = fit_background_basis(blk, n_snapshots=d["n_basis"], n_components=d["k"], normalization=Iqs)

    obs_pt = torch.from_numpy(obs.reshape(-1, L)).to(device)
    ref_pt = torch.from_numpy(ref_spec.reshape(-1, L)).to(device)
    c_ref = pca.project(ref_pt, k=d["k"]).repeat(Nt, 1).detach()   # (Nt*Cy*Cx, K), tiled over t

    meta = dict(Nt=Nt, Ny=Ny, Nx=Nx, Cy=Cy, Cx=Cx, L=L, Iqs=Iqs,
                ts=ts, wl=wl, ys=[ys.start, ys.stop], xs=[xs.start, xs.stop])
    return obs_pt, c_ref, pca, xl, meta


def make_coords(meta, device, grad=False):
    c = grid_coords(meta["Nt"], meta["Cy"], meta["Cx"], device=device)
    return c.requires_grad_(True) if grad else c


def sigmas_for(cfg, meta):
    """Per-axis Fourier bandwidths, SCALED with the domain so the reference-window
    resolution transfers (space columns by Cx / Cy, time by Nt)."""
    fc = cfg["field"]
    sx = fc["sigma_xy_ref"] * meta["Cx"] / fc["crop_ref"]
    sy = fc["sigma_xy_ref"] * meta["Cy"] / fc["crop_ref"]
    st = fc["sigma_t_ref"] * meta["Nt"] / fc["n_t_ref"]
    return sx, sy, st


def build_field(cfg, pca, obs_pt, meta, device):
    fc = cfg["field"]
    sx, sy, st = sigmas_for(cfg, meta)
    bkg_bias = pca.project(obs_pt, k=cfg["data"]["k"]).mean(0).detach().cpu()
    field = ParameterField(d_in=3, n_freq=fc["n_freq"], sigmas=(sx, sy, st),
                           width=fc["width"], depth=fc["depth"],
                           s_max=fc["s_max"], dv_floor=fc["dv_floor"],
                           bkg_bias=bkg_bias, seed=fc["seed"]).to(device)
    return field, (sx, sy, st)


# ==========================================================================
# Checkpointing
# ==========================================================================
def save_checkpoint(out, field, opt, sched, it, cfg, pca):
    os.makedirs(out, exist_ok=True)
    torch.save(dict(
        field=field.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(),
        iter=it, cfg=cfg,
        torch_rng=torch.get_rng_state(),
        cuda_rng=(torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None),
        pca=dict(mean=pca.mean_, components_full=pca.components_full,
                 eigenvalues_full=pca.eigenvalues_full,
                 n_components=pca.n_components, meta=pca.meta),
    ), os.path.join(out, "checkpoint.pt"))


def pca_from_ckpt(d):
    return PCABackground(mean=d["mean"], components_full=d["components_full"],
                         eigenvalues_full=d["eigenvalues_full"],
                         n_components=d["n_components"], meta=d.get("meta", {}))


# ==========================================================================
# Training
# ==========================================================================
def train(field, cfg, obs_pt, c_ref, pca, xl, meta, regw, device, out,
          opt=None, sched=None, start_iter=0):
    tc = cfg["train"]
    n_iters, batch = tc["n_iters"], tc["batch"]
    coords = make_coords(meta, device, grad=(batch is None))
    if opt is None:
        opt = torch.optim.Adam(field.parameters(), lr=tc["lr"])
    if sched is None:
        sched = CosineAnnealingLR(opt, T_max=n_iters, eta_min=tc["lr"] * 0.05)
    n = coords.shape[0]
    hist = []
    t0 = time.time()
    for it in range(start_iter, n_iters):
        opt.zero_grad()
        if batch is None:
            cb, ob, cr = coords, obs_pt, c_ref
        else:
            idx = torch.randint(0, n, (batch,), device=device)
            cb = coords[idx].detach().requires_grad_(True)   # clean leaf for autograd smoothness
            ob, cr = obs_pt[idx], c_ref[idx]                 # SAME pixels
        c, pc = field(cb)
        _, data = chi2_per_pixel(ob, composite_synth(xl, pca, c, pc, loga_clamp=None), reduce_mean=True)
        reg = regularization_loss(c, pc, cb, pca, weights=regw, c_ref=cr)
        loss = data + reg["total"]
        if not torch.isfinite(loss):
            print("[stop] non-finite loss at iter", it)
            break
        loss.backward()
        torch.nn.utils.clip_grad_norm_(field.parameters(), tc["grad_clip"])
        opt.step()
        sched.step()
        hist.append((loss.detach().item(), data.detach().item(), reg["total"].detach().item()))
        if (it + 1) % cfg["io"]["ckpt_every"] == 0 or (it + 1) == n_iters:
            save_checkpoint(out, field, opt, sched, it + 1, cfg, pca)
            el = time.time() - t0
            print("iter %6d/%d | loss %.4e | data %.4e | reg %.4e | %.0fs (%.1f ms/it)"
                  % (it + 1, n_iters, hist[-1][0], hist[-1][1], hist[-1][2],
                     el, 1e3 * el / max(it + 1 - start_iter, 1)))
    return hist, opt, sched


# ==========================================================================
# Evaluation, saving, metrics, report
# ==========================================================================
def eval_params(field, meta, device, chunk=50000):
    """Field parameter maps over the full grid (MLP only, cheap). Returns
    c_map (Nt,Cy,Cx,K) and p_map (Nt,Cy,Cx,10)."""
    coords = make_coords(meta, device, grad=False)
    cs, ps = [], []
    field.eval()
    with torch.no_grad():
        for i in range(0, coords.shape[0], chunk):
            c, pc = field(coords[i:i + chunk])
            cs.append(c.cpu()); ps.append(pc.cpu())
    c = torch.cat(cs).numpy(); pc = torch.cat(ps).numpy()
    sh = (meta["Nt"], meta["Cy"], meta["Cx"])
    return c.reshape(*sh, c.shape[1]), pc.reshape(*sh, 10)


def save_outputs(field, cfg, pca, meta, out):
    c_map, p_map = eval_params(field, meta, device=next(field.parameters()).device)
    path = os.path.join(out, "params.npz")
    np.savez_compressed(
        path, c=c_map, p_cloud=p_map, cloud_names=np.asarray(CLOUD_NAMES),
        wl=meta["wl"], lam=np.asarray(cfg["data"]["lam"]), Iqs=meta["Iqs"],
        ts=np.asarray(meta["ts"]), ys=np.asarray(meta["ys"]), xs=np.asarray(meta["xs"]),
        cfg=json.dumps(cfg))
    print("[save] %s | c %s | p_cloud %s" % (path, c_map.shape, p_map.shape))
    return c_map, p_map


def metrics(field, pca, xl, obs_pt, meta, device, cap=1.0, chunk=20000):
    """Stream the synthetic over the grid to report RMS and the background
    emission diagnostic without materialising the whole cube."""
    coords = make_coords(meta, device, grad=False)
    N = coords.shape[0]
    se = 0.0; cnt = 0; peaks = []
    field.eval()
    with torch.no_grad():
        for i in range(0, N, chunk):
            c, pc = field(coords[i:i + chunk])
            r = obs_pt[i:i + chunk] - composite_synth(xl, pca, c, pc, loga_clamp=None)
            se += float((r ** 2).sum()); cnt += r.numel()
            peaks.append(pca.reconstruct(c).max(dim=1).values.cpu())
    rms = (se / cnt) ** 0.5
    peak = torch.cat(peaks)
    mean_i = float(obs_pt.mean())
    print("[metrics] RMS %.4f (%.2f%% of mean I) | bkg peak/cont: med %.3f 95th %.3f max %.3f | >5%% emission %.1f%%"
          % (rms, 100 * rms / mean_i, peak.median(), peak.quantile(0.95),
             peak.max(), 100 * float((peak > cap + 0.05).float().mean())))
    return dict(rms=rms, rms_pct=100 * rms / mean_i,
                peak_median=float(peak.median()), peak_max=float(peak.max()))


def _forward_full(field, pca, xl, meta, device, chunk=20000):
    """Chunked full-grid synth / background / params, as CPU numpy cubes."""
    coords = make_coords(meta, device, grad=False)
    N, L, K = coords.shape[0], meta["L"], pca.n_components
    syn = np.empty((N, L), np.float32); bkg = np.empty((N, L), np.float32)
    cc = np.empty((N, K), np.float32); pp = np.empty((N, 10), np.float32)
    field.eval()
    with torch.no_grad():
        for i in range(0, N, chunk):
            c, pc = field(coords[i:i + chunk])
            syn[i:i + chunk] = composite_synth(xl, pca, c, pc, loga_clamp=None).cpu().numpy()
            bkg[i:i + chunk] = pca.reconstruct(c).cpu().numpy()
            cc[i:i + chunk] = c.cpu().numpy(); pp[i:i + chunk] = pc.cpu().numpy()
    sh = (meta["Nt"], meta["Cy"], meta["Cx"])
    return (syn.reshape(*sh, L), bkg.reshape(*sh, L), cc.reshape(*sh, K), pp.reshape(*sh, 10))


def report(field, cfg, pca, xl, obs_pt, meta, out, hist=None):
    """Write diagnostic PNGs (loss, obs-vs-fit maps, emission hist, spectra)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    device = next(field.parameters()).device
    Nt, Cy, Cx, L = meta["Nt"], meta["Cy"], meta["Cx"], meta["L"]
    syn, bkg, _, _ = _forward_full(field, pca, xl, meta, device)
    oc = obs_pt.cpu().numpy().reshape(Nt, Cy, Cx, L)
    kf, core, wl = Nt // 2, L // 2, meta["wl"]

    if hist:
        h = np.asarray(hist)
        plt.figure(figsize=(5, 3))
        for j, lab in enumerate(("total", "data", "reg")):
            plt.semilogy(h[:, j], label=lab)
        plt.legend(); plt.xlabel("iter"); plt.ylabel("loss"); plt.tight_layout()
        plt.savefig(os.path.join(out, "loss.png"), dpi=110); plt.close()

    wl_idx = [max(core - 120, 0), core, min(core + 120, L - 1)]
    fig, ax = plt.subplots(2, 3, figsize=(9, 5), constrained_layout=True)
    for j, ii in enumerate(wl_idx):
        vmn = min(oc[kf, :, :, ii].min(), syn[kf, :, :, ii].min())
        vmx = max(oc[kf, :, :, ii].max(), syn[kf, :, :, ii].max())
        ax[0, j].imshow(oc[kf, :, :, ii].T, origin="lower", cmap="magma", vmin=vmn, vmax=vmx)
        ax[1, j].imshow(syn[kf, :, :, ii].T, origin="lower", cmap="magma", vmin=vmn, vmax=vmx)
        ax[0, j].set_title("obs %.3f" % wl[ii]); ax[1, j].set_title("fit %.3f" % wl[ii])
    fig.savefig(os.path.join(out, "obs_vs_fit.png"), dpi=110); plt.close(fig)

    peak = bkg.reshape(-1, L).max(1)
    plt.figure(figsize=(5, 3)); plt.hist(peak, bins=60)
    plt.axvline(cfg["reg"]["no_emission"]["cap"], color="r", ls="--")
    plt.xlabel("bkg peak / continuum"); plt.ylabel("count"); plt.tight_layout()
    plt.savefig(os.path.join(out, "emission_hist.png"), dpi=110); plt.close()

    o2, f2, b2 = (a[kf].reshape(-1, L) for a in (oc, syn, bkg))
    rng = np.random.RandomState(2); ii = rng.randint(0, Cy * Cx, size=3)
    fig, ax = plt.subplots(1, 3, figsize=(12, 3))
    for j, p in enumerate(ii):
        ax[j].plot(wl, o2[p], "k", label="obs")
        ax[j].plot(wl, b2[p], "b:", label="bkg")
        ax[j].plot(wl, f2[p], "r--", label="fit")
        ax[j].set_title("px %d" % p)
    ax[0].legend(); plt.tight_layout()
    fig.savefig(os.path.join(out, "spectra.png"), dpi=110); plt.close(fig)
    print("[report] figures ->", out)


# ==========================================================================
# CLI
# ==========================================================================
def main():
    ap = argparse.ArgumentParser(
        description="Step 6: fit the (t,y,x,lambda) cube with the PCA-background + two-cloud neural field.")
    ap.add_argument("--config", help="JSON config file, deep-merged over the defaults")
    ap.add_argument("--set", nargs="*", default=[], metavar="k.v=VAL",
                    help="dotted-key overrides, e.g. data.crop=32 train.n_iters=2000")
    ap.add_argument("--out", help="output directory (overrides io.out)")
    ap.add_argument("--device", help="cuda | cpu | auto (overrides train.device)")
    ap.add_argument("--resume", help="checkpoint.pt to resume from")
    ap.add_argument("--no-train", action="store_true",
                    help="skip training; evaluate/save/report only (use with --resume)")
    ap.add_argument("--report", action="store_true", help="also write diagnostic figures")
    ap.add_argument("--dump-config", metavar="PATH", help="write the default config to PATH and exit")
    args = ap.parse_args()

    if args.dump_config:
        with open(args.dump_config, "w") as fh:
            json.dump(DEFAULT_CFG, fh, indent=2)
        print("wrote default config to", args.dump_config)
        return

    # ---- assemble config: base (resume ckpt or defaults) <- file <- --set <- flags
    if args.resume:
        ck = torch.load(args.resume, map_location="cpu", weights_only=False)
        cfg = copy.deepcopy(ck["cfg"])
    else:
        ck = None
        cfg = copy.deepcopy(DEFAULT_CFG)
    if args.config:
        with open(args.config) as fh:
            cfg = deep_merge(cfg, json.load(fh))
    cfg = apply_overrides(cfg, args.set)
    if args.out:
        cfg["io"]["out"] = args.out
    if args.device:
        cfg["train"]["device"] = args.device

    out = cfg["io"]["out"]
    os.makedirs(out, exist_ok=True)
    device = resolve_device(cfg)
    set_seed(cfg["field"]["seed"])
    print("[cfg] device %s | out %s | crop %s | n_t %s | iters %d"
          % (device, out, cfg["data"]["crop"], cfg["data"]["n_t"], cfg["train"]["n_iters"]))

    # ---- data / basis / field ----
    obs_pt, c_ref, pca, xl, meta = load_data(cfg, device)
    if ck is not None and ck.get("pca") is not None:
        pca = pca_from_ckpt(ck["pca"]).to(device=device)    # exact basis from the checkpoint
    regw = expand_group_config(cfg["reg"])
    field, sig = build_field(cfg, pca, obs_pt, meta, device)
    print("[field] weights %d | sigmas (x,y,t)=(%.2f,%.2f,%.2f) | samples %d | L %d"
          % (sum(p.numel() for p in field.parameters()), sig[0], sig[1], sig[2],
             meta["Nt"] * meta["Cy"] * meta["Cx"], meta["L"]))

    opt = sched = None
    start_iter = 0
    if ck is not None:
        field.load_state_dict(ck["field"])
        opt = torch.optim.Adam(field.parameters(), lr=cfg["train"]["lr"])
        opt.load_state_dict(ck["opt"])
        sched = CosineAnnealingLR(opt, T_max=cfg["train"]["n_iters"], eta_min=cfg["train"]["lr"] * 0.05)
        sched.load_state_dict(ck["sched"])
        torch.set_rng_state(ck["torch_rng"])
        if ck.get("cuda_rng") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(ck["cuda_rng"])
        start_iter = ck["iter"]
        print("[resume] %s at iter %d" % (args.resume, start_iter))

    # persist the resolved config actually used (incl. the scaled sigmas)
    with open(os.path.join(out, "config_used.json"), "w") as fh:
        json.dump(dict(cfg=cfg, sigmas=list(sig)), fh, indent=2, default=str)

    hist = None
    if not args.no_train:
        hist, opt, sched = train(field, cfg, obs_pt, c_ref, pca, xl, meta, regw, device, out,
                                 opt=opt, sched=sched, start_iter=start_iter)

    save_outputs(field, cfg, pca, meta, out)
    metrics(field, pca, xl, obs_pt, meta, device, cap=cfg["reg"]["no_emission"]["cap"])
    if args.report:
        report(field, cfg, pca, xl, obs_pt, meta, out, hist=hist)


if __name__ == "__main__":
    main()
