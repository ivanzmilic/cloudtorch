#!/usr/bin/env python
"""step_6a_reg_fit.py -- script form of step_5a_reg_test.ipynb (given-background PINN + reg).

This is the *scripted* twin of notebook 5a: the given-background PINN (n_bkg=0, fixed
background, clouds carry the core absorption) trained with explicit regularisation on the
8 free cloud fields (autograd space+time smoothness + soft one-sided hinges). It reproduces
the notebook run exactly -- same window, field, init, reg mechanism and iteration count --
but as a config-driven command line tool, following fit_cube.py's conventions.

Every knob is hard-set in DEFAULT_CFG below and can be:
  * overridden from a JSON config     :  --config my.json        (deep-merged over defaults)
  * overridden per-key on the CLI     :  --set train.n_iters=2000 reg.w_reg=3e-3
  * dumped to disk to start from       :  --dump-config my.json
Every run also writes the *effective* config to <out>/config_used.json -- feed that back
with --config to re-run an identical fit (the "read it from a log file" path).

The single most useful knob is reg.w_reg: a global multiplier on ALL reg weights
(w_xy/w_t/w_lo/w_hi; the lo/hi thresholds are NOT scaled). w_reg=1e-3 reproduces 4a
(reg a gentle backstop ~3% of loss, structured velocities); raising it collapses the
weakly-constrained velocity field. See docs/step_5a_factorial.pdf and the W_REG scan
(step_5a_reg_scan.py) for the trade-off curve.

Usage:
    python step_6a_reg_fit.py                              # defaults == notebook 5a, 10k iters
    python step_6a_reg_fit.py --out run_6a --set train.n_iters=6000
    python step_6a_reg_fit.py --config run_6a/config_used.json   # exact re-run
    python step_6a_reg_fit.py --dump-config my6a.json           # write defaults and exit

Outputs (in --out, default step_6a_out/): config_used.json, summary.txt, loss.png,
<prefix>_fit_images.pdf, <prefix>_param_maps.pdf, and (io.save_npz) fit_6a.npz.
Figures use a step_6a_ prefix so they never clobber the notebook's step_5a_*.pdf.
"""
import os, sys, time, json, copy, argparse

DEV = os.path.dirname(os.path.abspath(__file__))
os.chdir(DEV); sys.path.insert(0, os.path.join(DEV, '..')); sys.path.insert(0, DEV)

import numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR

from pinn_model import ParameterField, grid_coords
from composite_model import composite_synth_given, chi2_per_pixel
from given_background import load_given_background_window, default_bkg_path
from regularizers import regularization_loss, expand_group_config, field_layout

NAMES = ['S1', 'dtau1', 'vlos1', 'dv1', 'loga1', 'S2', 'dtau2', 'vlos2', 'dv2', 'loga2']

# ==========================================================================
# Default configuration == the step-5a notebook values. EVERY knob lives here;
# a --config file and --set overrides are deep-merged on top.
# ==========================================================================
DEFAULT_CFG = {
    "data": {
        "path": ["/dat/milic/MiHI_filament/mihi_all_data.npz",
                 "/home/milic/data/MiHI_halpha_filament/mihi_all_data.npz"],
        "bkg_path": None,          # None -> given_background.default_bkg_path(data path)
        "n_basis": 10,             # trailing snapshots used only to define Iqs
        "lam": [100, 500],         # wavelength slice
        "border": 15,              # trim YS/XS = slice(border, -border)
        "t0": 4, "n_t": 8,         # temporal window (frames t0-n_t/2 .. )
        "crop": 64                 # CROP x CROP spatial crop (null = whole bordered FOV)
    },
    "field": {
        "n_freq": 256,
        "sigma_xy": 12.0, "sigma_t": 8.0,   # Fourier bandwidths in normalised [-1,1] coords
        "width": 256, "depth": 4,
        "s_max": 0.5, "dv_floor": 8.0,
        "seed": 0,
        "freeze": ["loga1", "loga2"],       # cloud params held at init
        "s2_init": 0.3                      # S2 default 0.6 > s_max -> start below the cap
    },
    "train": {
        "n_iters": 10000, "lr": 0.005, "grad_clip": 1.0,
        "batch": None,                       # None = full batch; int = minibatch over coords
        "device": "auto"
    },
    # reg.w_reg is a GLOBAL scale on every w_* below (thresholds lo/hi untouched).
    # 1e-3 reproduces 4a; the per-group base weights match the notebook's REG dict.
    "reg": {
        "w_reg": 1e-3,
        "S":    {"w_xy": 1e-3, "w_t": 1e-2, "lo": 0.0,    "w_lo": 1e-2, "hi": 0.6, "w_hi": 1e-2},
        "dtau": {"w_xy": 1e-3, "w_t": 1e-3, "lo": 0.0,    "w_lo": 1e-2},
        "vlos": {"w_xy": 1e-3, "w_t": 2e-3, "lo": -120.0, "hi": 120.0, "w_lo": 1e-2, "w_hi": 1e-2},
        "dv":   {"w_xy": 1e-3, "w_t": 1e-3, "hi": 45.0,   "w_hi": 1e-2}
    },
    "io": {
        "out": "step_6a_out",
        "fig_prefix": "step_6a",
        "log_every": 2000,
        "save_npz": True
    }
}


# ==========================================================================
# Config helpers (same semantics as fit_cube.py)
# ==========================================================================
def deep_merge(base, over):
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        out[k] = deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def apply_overrides(cfg, pairs):
    """Apply `--set a.b.c=VAL`; VAL parsed as JSON when possible (32, 1e-3, null, [0,0])."""
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


def resolve_path(path):
    cands = path if isinstance(path, (list, tuple)) else [path]
    for p in cands:
        if os.path.exists(p):
            return p
    raise FileNotFoundError("no data file found among: %s" % (cands,))


def resolve_device(cfg):
    d = cfg["train"]["device"]
    if d == "auto":
        d = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(d)


def scaled_regw(reg_cfg):
    """Scale every w_* by reg.w_reg; leave lo/hi thresholds alone. -> per-field (8,) vectors."""
    w_reg = reg_cfg.get("w_reg", 1.0)
    groups = {g: {k: (v * w_reg if k.startswith("w_") else v) for k, v in d.items()}
              for g, d in reg_cfg.items() if g != "w_reg"}
    return expand_group_config(groups, n_bkg=0)


# ==========================================================================
# Data / field
# ==========================================================================
def load_window(cfg, device, emit):
    d = cfg["data"]
    data_npz = resolve_path(d["path"])
    bkg_npz = d["bkg_path"] or default_bkg_path(data_npz)
    lam = slice(d["lam"][0], d["lam"][1])
    b = d["border"]; YS, XS = slice(b, -b), slice(b, -b)

    f = np.load(data_npz, allow_pickle=True); wav = np.asarray(f['wav']); cube = f['data']
    Iqs = float(np.copy(cube[-d["n_basis"]:, YS, XS, lam]).astype(np.float32)[..., -20:].mean())
    full = cube[:, YS, XS, lam]; Ny, Nx = full.shape[1], full.shape[2]
    crop = d["crop"]
    if crop is None:
        ys = xs = slice(None)
    else:
        cy, cx = Ny // 2, Nx // 2
        ys = slice(cy - crop // 2, cy - crop // 2 + crop); xs = slice(cx - crop // 2, cx - crop // 2 + crop)
    ts = list(range(d["t0"] - d["n_t"] // 2, d["t0"] - d["n_t"] // 2 + d["n_t"]))
    obs = np.copy(full[ts][:, ys, xs, :]).astype(np.float32) / Iqs
    del f, cube, full
    Nt, Cy, Cx, L = obs.shape
    xl = [np.copy(wav[lam]).astype(np.float32), np.asarray([np.mean(wav[lam])], dtype=np.float32)]
    obs_pt = torch.from_numpy(obs.reshape(-1, L)).to(device)
    coords = grid_coords(Nt, Cy, Cx, device=device)
    I_in, _ = load_given_background_window(bkg_npz, ts, YS, XS, lam, ys, xs, Iqs, device=device)
    assert I_in.shape == obs_pt.shape and not I_in.requires_grad
    emit("data: %s" % data_npz)
    emit("bkg : %s" % bkg_npz)
    emit("frames %s | crop (%d,%d) | N=%d | L=%d | Iqs=%.4f | I_in mean=%.4f"
         % (ts, Cy, Cx, coords.shape[0], L, Iqs, float(I_in.mean())))
    return dict(wav=wav[lam], xl=xl, obs_pt=obs_pt, coords=coords, I_in=I_in,
                Nt=Nt, Cy=Cy, Cx=Cx, L=L, ts=ts)


def build_field(cfg, device):
    fc = cfg["field"]
    field = ParameterField(d_in=3, n_freq=fc["n_freq"],
                           sigmas=(fc["sigma_xy"], fc["sigma_xy"], fc["sigma_t"]),
                           width=fc["width"], depth=fc["depth"],
                           s_max=fc["s_max"], dv_floor=fc["dv_floor"],
                           n_bkg=0, freeze=tuple(fc["freeze"]),
                           init={"S2": fc["s2_init"]}).to(device)
    return field


# ==========================================================================
# Train
# ==========================================================================
def train(cfg, field, W, regw, device, emit):
    tc = cfg["train"]; n_iters = tc["n_iters"]; batch = tc["batch"]
    obs_pt, I_in, xl = W["obs_pt"], W["I_in"], W["xl"]
    coords_full = W["coords"].requires_grad_(True)
    opt = torch.optim.Adam(field.parameters(), lr=tc["lr"])
    sched = CosineAnnealingLR(opt, T_max=n_iters, eta_min=tc["lr"] * 0.05)
    hist, hist_data, hist_reg = [], [], []
    n = coords_full.shape[0]
    log_every = cfg["io"]["log_every"]
    t0 = time.time()
    for it in range(n_iters):
        opt.zero_grad()
        if batch is None:
            cb, ob, ib = coords_full, obs_pt, I_in
        else:
            idx = torch.randint(0, n, (int(batch),), device=device)
            cb = coords_full[idx].detach().requires_grad_(True)   # fresh grad leaf for smoothness
            ob, ib = obs_pt[idx], I_in[idx]
        c, pc = field(cb)
        _, data = chi2_per_pixel(ob, composite_synth_given(xl, ib, pc, loga_clamp=None), reduce_mean=True)
        reg = regularization_loss(c, pc, cb, weights=regw)
        loss = data + reg['total']
        if not torch.isfinite(loss):
            emit("non-finite loss at iter %d -- stopping" % it); break
        loss.backward(); torch.nn.utils.clip_grad_norm_(field.parameters(), tc["grad_clip"])
        opt.step(); sched.step()
        hist.append(float(loss.detach())); hist_data.append(float(data.detach())); hist_reg.append(float(reg['total'].detach()))
        if (it + 1) % log_every == 0 or (it + 1) == n_iters:
            emit("  iter %6d | total %.4e | data %.4e | reg %.4e (%.2f%%)"
                 % (it + 1, hist[-1], hist_data[-1], hist_reg[-1], 100 * hist_reg[-1] / hist[-1]))
    dt = time.time() - t0
    emit("DONE %d iters | data %.4e | reg %.4e (%.2f%% of loss) | %.0fs (%.1f ms/iter)"
         % (len(hist), hist_data[-1], hist_reg[-1], 100 * hist_reg[-1] / hist[-1], dt, 1e3 * dt / max(len(hist), 1)))
    return hist, hist_data, hist_reg


# ==========================================================================
# Summary + figures (mirror notebook cells 8/9/10)
# ==========================================================================
def summarize_and_save(cfg, field, W, hist, hist_data, hist_reg, out, emit):
    device = W["obs_pt"].device
    coords = W["coords"]
    with torch.no_grad():
        _, p_all = field(coords)
        syn = composite_synth_given(W["xl"], W["I_in"], p_all, loga_clamp=None)
    obs_pt = W["obs_pt"]
    resid = obs_pt - syn; chi2v = (resid ** 2).sum(-1)
    rms = 100.0 * float(torch.sqrt((resid ** 2).mean())) / float(obs_pt.mean())
    pa = p_all.detach().cpu().numpy()
    Nt, Cy, Cx, L = W["Nt"], W["Cy"], W["Cx"], W["L"]
    kf = Nt // 2
    pm = pa.reshape(Nt, Cy, Cx, 10)[kf]

    emit("\nfinal mean per-pixel chi2: %.4f | median: %.4f | p95: %.4f"
         % (float(chi2v.mean()), float(chi2v.median()), float(torch.quantile(chi2v, 0.95))))
    emit("RMS residual: %.2f %% of mean intensity" % rms)
    emit("vlos1 std=%.2f  vlos2 std=%.2f   (4a/noreg ref: 0.170 chi2, v1 std ~19, v2 std ~10)"
         % (pm[:, :, 2].std(), pm[:, :, 7].std()))
    emit("\n  param      min    median      max")
    for j, nm in enumerate(NAMES):
        emit("  %-7s %8.3f %8.3f %8.3f" % (nm, pa[:, j].min(), np.median(pa[:, j]), pa[:, j].max()))

    fig_prefix = cfg["io"]["fig_prefix"]
    wl = W["wav"]; core = L // 2

    # loss curve
    plt.figure(figsize=(5.2, 3))
    plt.semilogy(hist, label='total'); plt.semilogy(hist_data, label='data (chi2)'); plt.semilogy(hist_reg, label='reg')
    plt.legend(); plt.xlabel('iteration'); plt.ylabel('loss'); plt.title('given-background PINN loss (step 6a)')
    plt.tight_layout(); plt.savefig(os.path.join(out, 'loss.png'), dpi=120); plt.close()

    # obs vs fit at 3 wavelengths (centre frame)
    oc = obs_pt.cpu().numpy().reshape(Nt, Cy, Cx, L)[kf]
    fc = syn.detach().cpu().numpy().reshape(Nt, Cy, Cx, L)[kf]
    wl_idx = [max(core - 120, 0), core, min(core + 120, L - 1)]
    fig, ax = plt.subplots(2, 3, figsize=(12, 7), constrained_layout=True)
    for j, ii in enumerate(wl_idx):
        vmn = min(oc[:, :, ii].min(), fc[:, :, ii].min()); vmx = max(oc[:, :, ii].max(), fc[:, :, ii].max())
        ax[0, j].imshow(oc[:, :, ii].T, origin='lower', cmap='magma', vmin=vmn, vmax=vmx)
        im = ax[1, j].imshow(fc[:, :, ii].T, origin='lower', cmap='magma', vmin=vmn, vmax=vmx)
        ax[0, j].set_title('obs @ %.3f A' % wl[ii]); ax[1, j].set_title('fit @ %.3f A' % wl[ii])
        for a in (ax[0, j], ax[1, j]):
            a.set_xticks([]); a.set_yticks([])
        fig.colorbar(im, ax=[ax[0, j], ax[1, j]], fraction=0.046, pad=0.02)
    fig.suptitle('step 6a obs vs fit (frame %d)' % W["ts"][kf])
    plt.savefig(os.path.join(out, '%s_fit_images.pdf' % fig_prefix), dpi=150); plt.close()

    # cloud parameter maps (centre frame)
    specs = [('S1', 0, 'viridis'), ('dtau1', 1, 'viridis'), ('vlos1', 2, 'seismic'), ('dv1', 3, 'magma'),
             ('S2', 5, 'viridis'), ('dtau2', 6, 'viridis'), ('vlos2', 7, 'seismic'), ('dv2', 8, 'magma')]
    fig, ax = plt.subplots(2, 4, figsize=(18, 8), constrained_layout=True); ax = ax.ravel()
    for k, (nm, j, cmap) in enumerate(specs):
        im2 = pm[:, :, j].T; lo, hi = np.percentile(im2, [2, 98])
        if cmap == 'seismic':
            m = max(abs(lo), abs(hi)); lo, hi = -m, m
        h = ax[k].imshow(im2, origin='lower', cmap=cmap, vmin=lo, vmax=hi)
        ax[k].set_title('%s   [%.2f, %.2f] (2-98%%)' % (nm, lo, hi)); ax[k].set_xticks([]); ax[k].set_yticks([])
        fig.colorbar(h, ax=ax[k], fraction=0.046, pad=0.02)
    fig.suptitle('step 6a cloud parameter maps, frame %d (regularised; log a frozen)' % W["ts"][kf])
    plt.savefig(os.path.join(out, '%s_param_maps.pdf' % fig_prefix), dpi=150); plt.close()

    if cfg["io"]["save_npz"]:
        np.savez_compressed(os.path.join(out, 'fit_6a.npz'),
                            params=pa.reshape(Nt, Cy, Cx, 10).astype(np.float32),
                            obs=oc.astype(np.float32), fit=fc.astype(np.float32),
                            chi2=chi2v.detach().cpu().numpy().reshape(Nt, Cy, Cx).astype(np.float32),
                            wav=wl.astype(np.float32), ts=np.asarray(W["ts"]))
    emit("\nsaved: %s/{config_used.json, summary.txt, loss.png, %s_fit_images.pdf, %s_param_maps.pdf%s}"
         % (out, fig_prefix, fig_prefix, ", fit_6a.npz" if cfg["io"]["save_npz"] else ""))


# ==========================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", help="JSON config file, deep-merged over the defaults")
    ap.add_argument("--set", nargs="*", default=[], metavar="k.v=VAL", help="per-key overrides")
    ap.add_argument("--out", help="output directory (overrides io.out)")
    ap.add_argument("--device", help="cuda | cpu | auto (overrides train.device)")
    ap.add_argument("--dump-config", metavar="PATH", help="write the default config to PATH and exit")
    args = ap.parse_args()

    if args.dump_config:
        with open(args.dump_config, "w") as fh:
            json.dump(DEFAULT_CFG, fh, indent=2)
        print("wrote default config to", args.dump_config); return

    cfg = copy.deepcopy(DEFAULT_CFG)
    if args.config:
        with open(args.config) as fh:
            cfg = deep_merge(cfg, json.load(fh))
    cfg = apply_overrides(cfg, args.set)
    if args.out:
        cfg["io"]["out"] = args.out
    if args.device:
        cfg["train"]["device"] = args.device

    out = os.path.join(DEV, cfg["io"]["out"]) if not os.path.isabs(cfg["io"]["out"]) else cfg["io"]["out"]
    os.makedirs(out, exist_ok=True)
    log = open(os.path.join(out, "summary.txt"), "w")
    def emit(*a):
        s = " ".join(str(x) for x in a); print(s, flush=True); log.write(s + "\n"); log.flush()

    device = resolve_device(cfg)
    np.random.seed(cfg["field"]["seed"]); torch.manual_seed(cfg["field"]["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg["field"]["seed"])

    emit("step 6a (given-background PINN + regularisation) | device %s" % device)
    emit("reg.w_reg=%g | n_iters=%d | seed=%d" % (cfg["reg"]["w_reg"], cfg["train"]["n_iters"], cfg["field"]["seed"]))

    # log the effective config (the "read it from a log file" path: --config <out>/config_used.json)
    with open(os.path.join(out, "config_used.json"), "w") as fh:
        json.dump(cfg, fh, indent=2)

    W = load_window(cfg, device, emit)
    regw = scaled_regw(cfg["reg"])
    emit("cloud fields: %s" % field_layout(0)[0])
    emit("scaled w_xy=%.2e | vlos w_t=%.2e | vlos hinge w=%.2e | vlos hinge [%.0f,%.0f] | dv cap %.0f"
         % (float(regw['w_xy'][0]), float(regw['w_t'][2]), float(regw['w_hi'][2]),
            float(regw['lo'][2]), float(regw['hi'][2]), float(regw['hi'][3])))

    field = build_field(cfg, device)
    n_par = sum(p.numel() for p in field.parameters())
    with torch.no_grad():
        _, pc0 = field(W["coords"])
        _, l0 = chi2_per_pixel(W["obs_pt"], composite_synth_given(W["xl"], W["I_in"], pc0, loga_clamp=None), reduce_mean=True)
    emit("field weights: %d | frozen %s | init chi2: %.4f" % (n_par, cfg["field"]["freeze"], float(l0)))

    hist, hist_data, hist_reg = train(cfg, field, W, regw, device, emit)
    summarize_and_save(cfg, field, W, hist, hist_data, hist_reg, out, emit)
    log.close()


if __name__ == "__main__":
    main()
