# 3D filament movie — hand-off notes

Tooling for turning the packed inversion result into PPV (position–position–velocity)
movies of the MiHI filament. Two scripts, one shared data source.

## Data source
`/dat/milic/MiHI_filament/inv_results/fits_and_params.npz` — the same packed
`obs/fits/params` array that feeds `npz_inversion_videos.py`. Params are read straight
from the `.npz` (not from rasterised PNGs) so the 3D render stays crisp.

Canonical 10-wide params row; the two clouds live at columns
`(dtau, vlos) = (1, 2)` and `(6, 7)`.

## Environment
`cv2` (OpenCV, used for optical flow + video writing) is **not** in `base`. Use the
`pt` conda env:

```
/home/milic/miniconda3/envs/pt/bin/python figs/debug/params_3d_movie.py ...
```

## Scripts

### `params_3d_movie.py`  → PPV point-cloud movie
- Sky plane = floor (x, y). Height = **−v_los** (SIGN CONVENTION: negative v_los = UP /
  rising / blueshift; positive = DOWN / sinking / redshift).
- Brightness/alpha = optical depth via `w = 1 − exp(−dtau/tau_scale)` (0 at dtau=0,
  saturates at 1). Optically-thin points vanish, so only real filaments are drawn.
- Colour = v_los (same story as height).
- Horizontal thread drift = dense optical flow (Farneback) of the dtau maps, used to
  **advect** the point cloud between frames (real sliding, not a crossfade).
  Measured drift is gentle (~0.8 px/frame median); exaggerate with `--flow-gain`.

Two looks:
- **scientific** (default): labelled axes, inverted colourbar (blue/up on top), captions.
  → `inversion_3d.mp4`
- **`--flashy`** (presentation): glowing neon point cloud (cyan = rising, orange/magenta
  = sinking), no axes/colourbar, full 360° camera orbit, bloom + vignette post-process.
  Trades physical fidelity for punch. → `inversion_3d_flashy.mp4` (separate file, never
  clobbers the science clip).

Run:
```
# science clip
/home/milic/miniconda3/envs/pt/bin/python figs/debug/params_3d_movie.py --prefix inversion
# presentation clip
/home/milic/miniconda3/envs/pt/bin/python figs/debug/params_3d_movie.py --flashy --prefix inversion
```

Useful knobs: `--sub` (time-interp smoothness), `--fps`, `--spin-deg`, `--flow-gain`,
`--point-size`, `--tau-scale`, `--min-w`. Flashy-only: `--bloom`, `--vignette`, `--title`.
`--flashy` auto-bumps spin→360, flow-gain→2.4, bob→6 unless you override them.

### `npz_inversion_videos.py`  → 2D parameter-map movies
Renders the 8 shown params as a movie. `--tau-weight-vlos` fades each vlos panel by its
own cloud's dtau (same `1 − exp(−dtau/tau_scale)` weight); modes `alpha` (fade to grey)
or `scale` (scale toward 0). vlos panels use the `bwr` diverging map on symmetric limits.

## Convention
Script `.py` / `.md` files are git-tracked & committed; generated `.mp4` outputs stay
uncommitted.

## Open ideas for continuation
- Optional motion trails / temporal persistence for the flashy clip.
- A seamless-loop camera (spin exactly 360 with matched first/last frame).
- Per-cloud colour split (e.g. cloud 1 vs cloud 2) instead of merging both populations.
- Tunable depth fog to strengthen the sense of 3D on the presentation clip.
