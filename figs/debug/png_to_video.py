#!/usr/bin/env python
"""png_to_video.py -- stitch a numbered set of PNGs into a video with OpenCV.

Frames are ordered by the integers in each filename (natural sort), so
0_, 1_, ..., 2_, 10_ come out in the right order instead of lexicographically.
All frames are resized to the first frame's size (VideoWriter needs one constant
size), and dimensions are trimmed to even numbers (friendlier to mp4 codecs).

Defaults act on this script's own folder. Because a plain *.png here interleaves
the two plot series, select one series per video with --pattern:

    python png_to_video.py --pattern '*_fit_images.png'  --out fit_images.mp4  --fps 6
    python png_to_video.py --pattern '*_fit_spectra.png' --out fit_spectra.mp4 --fps 6
    python png_to_video.py                                    # *.png -> movie.mp4 @ 4 fps
    python png_to_video.py --dir /some/where --pattern 'frame_*.png' --out anim.avi
"""
import os
import re
import glob
import argparse

import cv2

HERE = os.path.dirname(os.path.abspath(__file__))


def natural_key(path):
    """Sort key: the integers in the basename, in order (natural sort)."""
    nums = re.findall(r"\d+", os.path.basename(path))
    return ([int(n) for n in nums] if nums else [0], os.path.basename(path))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default=HERE, help="folder to read PNGs from (default: this script's folder)")
    ap.add_argument("--pattern", default="*.png", help="glob within --dir (default: *.png)")
    ap.add_argument("--out", default="movie.mp4", help="output file; extension picks codec (.mp4 or .avi)")
    ap.add_argument("--fps", type=float, default=4.0, help="frames per second (default: 4)")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.dir, args.pattern)), key=natural_key)
    if not files:
        raise SystemExit("no PNGs match %r in %s" % (args.pattern, args.dir))

    first = cv2.imread(files[0])
    if first is None:
        raise SystemExit("could not read first frame: %s" % files[0])
    h, w = first.shape[:2]
    w -= w % 2
    h -= h % 2                                   # even dims for the encoder

    out_path = args.out if os.path.isabs(args.out) else os.path.join(args.dir, args.out)
    ext = os.path.splitext(out_path)[1].lower()
    fourcc = cv2.VideoWriter_fourcc(*("MJPG" if ext == ".avi" else "mp4v"))
    vw = cv2.VideoWriter(out_path, fourcc, args.fps, (w, h))
    if not vw.isOpened():
        raise SystemExit("VideoWriter failed to open %s (codec/extension issue)" % out_path)

    n = 0
    for f in files:
        im = cv2.imread(f)
        if im is None:
            print("  skip (unreadable):", os.path.basename(f)); continue
        if im.shape[1] != w or im.shape[0] != h:
            im = cv2.resize(im, (w, h), interpolation=cv2.INTER_AREA)
        vw.write(im); n += 1
        print("  + %s" % os.path.basename(f))
    vw.release()
    print("wrote %s  (%d frames, %.1f fps, %dx%d)" % (out_path, n, args.fps, w, h))


if __name__ == "__main__":
    main()
