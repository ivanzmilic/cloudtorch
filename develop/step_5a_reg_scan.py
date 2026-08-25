#!/usr/bin/env python
"""step_5a_reg_scan.py -- W_REG parameter search for the step-5a given-background PINN.

Packs the step_5a_reg_test.ipynb setup into a script and sweeps the single global
regularisation knob W_REG to find where the fit spoils. W_REG scales EVERY reg weight
(w_xy / w_t / w_lo / w_hi) by the same factor; the lo/hi thresholds are unchanged.

Background: on the given-background problem vlos is a near-flat, weakly-constrained
direction (the fixed background owns the line wings), so the reg trades ~monotonically
against BOTH chi2 and velocity structure -- there is no strong-reg sweet spot. W_REG=1e-3
reproduces 4a (reg a gentle backstop, ~3% of loss); raising it collapses the velocities.
See docs/step_5a_factorial.pdf. This scan quantifies the transition.

For each W_REG it reports final chi2 (mean/median/p95), RMS, reg share of loss, and the
per-cloud velocity spread (std + min/median/max) -- collapse shows up as std -> 0 and
min ~ median ~ max. W_REG=0 is the reg-OFF reference (== step 4a on this window).

Usage:
    python step_5a_reg_scan.py                         # default grid, 4000 iters
    python step_5a_reg_scan.py --wreg 0,1e-3,1e-2,1e-1 --n-iters 6000
    python step_5a_reg_scan.py --n-iters 10000 --outdir step_5a_reg_scan_out

Runtime ~ len(grid) * N_ITERS * ~72 ms on a GPU (default grid+iters ~35 min).
"""
import os, sys, time, json, argparse

DEV = os.path.dirname(os.path.abspath(__file__))
os.chdir(DEV); sys.path.insert(0, os.path.join(DEV, '..')); sys.path.insert(0, DEV)

import numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR

from pinn_model import ParameterField, grid_coords, _SLOT_INIT, _SLOT_SCALE
from composite_model import composite_synth_given, chi2_per_pixel
from given_background import load_given_background_window, default_bkg_path
from regularizers import regularization_loss, expand_group_config

# --------------------------------------------------------------------------- config
# ---- 5a fit window / field (verbatim from step_5a_reg_test.ipynb) ----
N_BASIS  = 10;  LAM = slice(100, 500)
T0 = 4;  N_T = 8;  CROP = 64
FREEZE   = ("loga1", "loga2")
N_FREQ   = 256;  SIGMA_XY = 12.0;  SIGMA_T = 8.0
WIDTH    = 256;  DEPTH = 4;  S_MAX = 0.5;  DV_FLOOR = 8.0
LR = 5e-3;  GRAD_CLIP = 1.0
S2_INIT  = 0.3   # default init 0.6 > S_MAX; start below the cap (same as the notebook)

# ---- base reg weights (before the W_REG multiplier); thresholds lo/hi are NOT scaled ----
REG_BASE = {
    'S':    dict(w_xy=1e-3, w_t=1e-2, lo=0.0,    w_lo=1e-2, hi=0.6, w_hi=1e-2),
    'dtau': dict(w_xy=1e-3, w_t=1e-3, lo=0.0,    w_lo=1e-2),
    'vlos': dict(w_xy=1e-3, w_t=2e-3, lo=-120.0, hi=120.0, w_lo=1e-2, w_hi=1e-2),
    'dv':   dict(w_xy=1e-3, w_t=1e-3, hi=45.0,   w_hi=1e-2),
}
DEFAULT_GRID = [0.0, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 1e0]

_CANDS = ['/dat/milic/MiHI_filament/mihi_all_data.npz',
          '/home/milic/data/MiHI_halpha_filament/mihi_all_data.npz']
NAMES = ['S1','dtau1','vlos1','dv1','loga1','S2','dtau2','vlos2','dv2','loga2']


def scaled_regw(w_reg):
    """Scale every WEIGHT (w_*) by w_reg; leave thresholds (lo/hi) untouched -> per-field vectors."""
    reg = {g: {k: (v * w_reg if k.startswith('w_') else v) for k, v in d.items()}
           for g, d in REG_BASE.items()}
    return expand_group_config(reg, n_bkg=0)


def load_window(device):
    """Carve the 5a fit window + same-window given background (once)."""
    data_npz = next((p for p in _CANDS if os.path.exists(p)), _CANDS[0])
    bkg_npz  = default_bkg_path(data_npz)
    f = np.load(data_npz, allow_pickle=True); wav = np.asarray(f['wav']); cube = f['data']
    YS, XS = slice(15, -15), slice(15, -15)
    Iqs = float(np.copy(cube[-N_BASIS:, YS, XS, LAM]).astype(np.float32)[..., -20:].mean())
    full = cube[:, YS, XS, LAM]; Ny, Nx = full.shape[1], full.shape[2]
    cy, cx = Ny // 2, Nx // 2
    ys = slice(cy - CROP // 2, cy - CROP // 2 + CROP); xs = slice(cx - CROP // 2, cx - CROP // 2 + CROP)
    ts = list(range(T0 - N_T // 2, T0 - N_T // 2 + N_T))
    obs = np.copy(full[ts][:, ys, xs, :]).astype(np.float32) / Iqs
    del f, cube, full
    Nt, Cy, Cx, L = obs.shape
    xl = [np.copy(wav[LAM]).astype(np.float32), np.asarray([np.mean(wav[LAM])], dtype=np.float32)]
    obs_pt = torch.from_numpy(obs.reshape(-1, L)).to(device)
    coords = grid_coords(Nt, Cy, Cx, device=device)
    I_in, _ = load_given_background_window(bkg_npz, ts, YS, XS, LAM, ys, xs, Iqs, device=device)
    return dict(data_npz=data_npz, wav=wav, xl=xl, obs_pt=obs_pt, coords=coords, I_in=I_in,
                Nt=Nt, Cy=Cy, Cx=Cx, L=L, ts=ts)


def run_one(w_reg, W, n_iters, seed, device, emit):
    """Fresh field + training for a single W_REG. Returns a metrics dict (+ vlos maps)."""
    reg_on = w_reg > 0.0
    emit("\n==========  W_REG=%-8g  (reg %s)  ==========" % (w_reg, "ON" if reg_on else "OFF"))
    torch.manual_seed(seed)
    coords = W['coords'].clone().detach()
    if reg_on:
        coords = coords.requires_grad_(True)
    field = ParameterField(d_in=3, n_freq=N_FREQ, sigmas=(SIGMA_XY, SIGMA_XY, SIGMA_T),
                           width=WIDTH, depth=DEPTH, s_max=S_MAX, dv_floor=DV_FLOOR,
                           n_bkg=0, freeze=FREEZE, init={"S2": S2_INIT}).to(device)
    regw = scaled_regw(w_reg) if reg_on else None
    opt = torch.optim.Adam(field.parameters(), lr=LR)
    sched = CosineAnnealingLR(opt, T_max=n_iters, eta_min=LR * 0.05)
    obs_pt, I_in, xl = W['obs_pt'], W['I_in'], W['xl']
    t0 = time.time(); last = (0.0, 0.0, 0.0)
    for it in range(n_iters):
        opt.zero_grad()
        c, pc = field(coords)
        _, data = chi2_per_pixel(obs_pt, composite_synth_given(xl, I_in, pc, loga_clamp=None), reduce_mean=True)
        if reg_on:
            reg = regularization_loss(c, pc, coords, weights=regw)
            loss = data + reg['total']; rt = float(reg['total'])
        else:
            loss = data; rt = 0.0
        if not torch.isfinite(loss):
            emit("  non-finite loss at iter %d -- stopping" % it); break
        loss.backward(); torch.nn.utils.clip_grad_norm_(field.parameters(), GRAD_CLIP)
        opt.step(); sched.step(); last = (float(loss.detach()), float(data.detach()), rt)
    dt = time.time() - t0

    with torch.no_grad():
        _, p_all = field(coords)
        syn = composite_synth_given(xl, I_in, p_all, loga_clamp=None)
    resid = obs_pt - syn; chi2v = (resid ** 2).sum(-1)
    rms = 100.0 * float(torch.sqrt((resid ** 2).mean())) / float(obs_pt.mean())
    pa = p_all.detach().cpu().numpy()
    kf = W['Nt'] // 2
    pm = pa.reshape(W['Nt'], W['Cy'], W['Cx'], 10)[kf]
    v1, v2 = pm[:, :, 2], pm[:, :, 7]
    reg_pct = 100 * last[2] / last[0] if last[0] else 0.0

    emit("  chi2 %.4f | median %.4f | p95 %.4f | RMS %.2f%% | reg %.1f%% of loss | %.0fs"
         % (float(chi2v.mean()), float(chi2v.median()), float(torch.quantile(chi2v, 0.95)), rms, reg_pct, dt))
    emit("  vlos1 std=%6.2f  min/med/max %7.1f /%6.1f /%6.1f" % (v1.std(), v1.min(), np.median(v1), v1.max()))
    emit("  vlos2 std=%6.2f  min/med/max %7.1f /%6.1f /%6.1f" % (v2.std(), v2.min(), np.median(v2), v2.max()))
    emit("  dtau1 max %.1f | dv1 %.1f..%.1f | S1 max %.2f | S2 max %.2f"
         % (pa[:, 1].max(), pa[:, 3].min(), pa[:, 3].max(), pa[:, 0].max(), pa[:, 5].max()))
    return dict(w_reg=w_reg, chi2=float(chi2v.mean()), chi2_med=float(chi2v.median()),
                chi2_p95=float(torch.quantile(chi2v, 0.95)), rms=rms, reg_pct=reg_pct, sec=dt,
                v1_std=float(v1.std()), v2_std=float(v2.std()),
                v1=[float(v1.min()), float(np.median(v1)), float(v1.max())],
                v2=[float(v2.min()), float(np.median(v2)), float(v2.max())],
                dtau1_max=float(pa[:, 1].max()), dv1=[float(pa[:, 3].min()), float(pa[:, 3].max())],
                s1_max=float(pa[:, 0].max()), s2_max=float(pa[:, 5].max()),
                _v1map=v1.copy(), _v2map=v2.copy())


def make_figures(results, grid, outdir, ts_kf):
    tags = [g for g in grid]
    # (1) trade-off curve: chi2 (left) and vlos std (right) vs W_REG
    reg_grid = [g for g in grid if g > 0]
    fig, axL = plt.subplots(figsize=(7.2, 4.2))
    axR = axL.twinx()
    xr = [g for g in reg_grid]
    axL.plot(xr, [results[g]['chi2'] for g in reg_grid], 'o-', color='#1B4F72', label=r'$\langle\chi^2\rangle$')
    axR.plot(xr, [results[g]['v1_std'] for g in reg_grid], 's--', color='#C0392B', label='vlos1 std')
    axR.plot(xr, [results[g]['v2_std'] for g in reg_grid], '^:', color='#E67E22', label='vlos2 std')
    if 0.0 in results:  # reg-off reference lines
        axL.axhline(results[0.0]['chi2'], color='#1B4F72', lw=0.8, alpha=0.5, ls=':')
        axR.axhline(results[0.0]['v1_std'], color='#C0392B', lw=0.8, alpha=0.4, ls=':')
    axL.set_xscale('log'); axL.set_xlabel('W_REG'); axL.set_ylabel(r'$\langle\chi^2\rangle$', color='#1B4F72')
    axR.set_ylabel('vlos std [km/s]', color='#C0392B')
    axL.set_title('step 5a: chi2 vs velocity structure across W_REG\n(dotted = reg-off / 4a reference)')
    lines = axL.get_lines()[:1] + axR.get_lines()[:2]
    axL.legend(lines, [l.get_label() for l in lines], loc='center right', fontsize=9)
    plt.tight_layout(); plt.savefig(os.path.join(outdir, 'step_5a_reg_scan_curve.png'), dpi=120); plt.close()

    # (2) vlos maps across W_REG (top vlos1 / bottom vlos2), common scale per row
    n = len(grid)
    fig, ax = plt.subplots(2, n, figsize=(3.4 * n, 7.2), constrained_layout=True)
    for r, key in enumerate(('_v1map', '_v2map')):
        vmax = max(np.percentile(np.abs(results[g][key]), 98) for g in grid)
        for cix, g in enumerate(grid):
            h = ax[r, cix].imshow(results[g][key].T, origin='lower', cmap='seismic', vmin=-vmax, vmax=vmax)
            ax[r, cix].set_xticks([]); ax[r, cix].set_yticks([])
            if r == 0:
                ax[r, cix].set_title('W_REG=%g\nchi2=%.3f\nv1std=%.1f'
                                     % (g, results[g]['chi2'], results[g]['v1_std']), fontsize=9)
            if cix == 0:
                ax[r, cix].set_ylabel(('vlos1', 'vlos2')[r])
        fig.colorbar(h, ax=list(ax[r, :]), fraction=0.015, pad=0.01)
    fig.suptitle('step 5a W_REG scan: velocity maps (frame %d) -- reg-off left/small -> collapsed right/large' % ts_kf)
    plt.savefig(os.path.join(outdir, 'step_5a_reg_scan_vlos.png'), dpi=110); plt.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--wreg', type=str, default=None,
                    help='comma-separated W_REG values (0 = reg off). Default: %s' % DEFAULT_GRID)
    ap.add_argument('--n-iters', type=int, default=4000, help='iters per run (default 4000)')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--outdir', type=str, default='step_5a_reg_scan_out')
    args = ap.parse_args()

    grid = DEFAULT_GRID if args.wreg is None else [float(x) for x in args.wreg.split(',')]
    outdir = os.path.join(DEV, args.outdir); os.makedirs(outdir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    log = open(os.path.join(outdir, 'scan_summary.txt'), 'w')
    def emit(*a):
        s = ' '.join(str(x) for x in a); print(s, flush=True); log.write(s + '\n'); log.flush()

    emit("step 5a W_REG scan | device %s | N_ITERS %d | seed %d" % (device, args.n_iters, args.seed))
    emit("pinn init: vlos %.0f/%.0f scale %.0f | S %.1f/%.1f (S2 init override %.2f) | S_MAX %.2f"
         % (_SLOT_INIT['vlos1'], _SLOT_INIT['vlos2'], _SLOT_SCALE['vlos1'],
            _SLOT_INIT['S1'], _SLOT_INIT['S2'], S2_INIT, S_MAX))
    emit("window T0=%d N_T=%d CROP=%d | SIGMA_XY=%.0f SIGMA_T=%.0f | grid %s"
         % (T0, N_T, CROP, SIGMA_XY, SIGMA_T, grid))
    emit("NOTE: metrics are at N_ITERS=%d (reg slows convergence); the good config (W_REG=1e-3)"
         % args.n_iters)
    emit("      reaches chi2~0.18 only by ~10k iters. Relative ordering across W_REG is the point.")

    W = load_window(device)
    emit("data: %s | frames %s | crop (%d,%d) | L=%d\n" % (W['data_npz'], W['ts'], W['Cy'], W['Cx'], W['L']))

    results = {}
    for g in grid:
        results[g] = run_one(g, W, args.n_iters, args.seed, device, emit)

    # ---- summary table ----
    emit("\n================================  SUMMARY  ================================")
    emit("%-9s | %7s %7s %7s | %5s | %6s | %8s %8s | %-8s"
         % ("W_REG", "chi2", "median", "p95", "RMS%", "reg%", "v1_std", "v2_std", "verdict"))
    ref = results.get(0.0)
    for g in grid:
        r = results[g]
        # spoil heuristic: velocity collapse (std small) or chi2 well above reg-off ref
        collapsed = (r['v1_std'] < 6.0) or (r['v2_std'] < 1.5)
        if ref is not None:
            chi2_bad = r['chi2'] > 1.5 * ref['chi2']
        else:
            chi2_bad = False
        if g == 0.0:
            verdict = 'REG-OFF'
        elif collapsed or chi2_bad:
            verdict = 'SPOILED'
        else:
            verdict = 'ok'
        emit("%-9g | %7.4f %7.4f %7.4f | %5.2f | %6.1f | %8.2f %8.2f | %-8s"
             % (g, r['chi2'], r['chi2_med'], r['chi2_p95'], r['rms'], r['reg_pct'], r['v1_std'], r['v2_std'], verdict))

    make_figures(results, grid, outdir, W['ts'][W['Nt'] // 2])
    json.dump({str(g): {k: v for k, v in results[g].items() if not k.startswith('_')} for g in grid},
              open(os.path.join(outdir, 'scan_metrics.json'), 'w'), indent=2)
    emit("\nDONE. outdir: %s" % outdir)
    emit("  figures: step_5a_reg_scan_curve.png  step_5a_reg_scan_vlos.png")
    log.close()


if __name__ == '__main__':
    main()
