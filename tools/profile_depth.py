"""
Measure the head's front-to-back shape from the side-view photograph.

Until now the skull behind the face was a guessed dome — two hand-picked
constants. The side views make it measurable: the alpha silhouette of a profile
shot IS the head's depth profile, row by row.

Registration between the front and side shots uses two landmarks both views
share and which do not depend on head orientation:
  * the top of the head (first opaque row)
  * the shoulder line (where silhouette width jumps as the neck meets the body)
Scaling by the distance between them removes any difference in framing.

Exports, per character:
  headTopY / shoulderY   in each view, for the record
  depthProfile           normalised head depth vs normalised height
  depthRatio             head depth / head height, the single number the dome
                         was previously guessing

Run:  tools/.venv/bin/python tools/profile_depth.py
"""

import json
import pathlib

import numpy as np
from PIL import Image

ROOT = pathlib.Path(__file__).resolve().parent.parent
REFS = ROOT / "input" / "character_references"
OUT = ROOT / "assets" / "facedata"
ALPHA_CUT = 128

# Located by filename, searching every character folder: these shots have moved
# between folders during the build, and a hard-coded path silently resolves to
# the wrong person rather than failing.
def find(stem):
    hits = sorted(REFS.glob(f"*/{stem}.png"))
    if not hits:
        raise FileNotFoundError(f"no reference named {stem}.png under {REFS}")
    return hits[0]


VIEWS = {
    "male": {"side": "male_character_side_view", "back": "male_character_back_view",
             "front": "male_character"},
    "female": {"side": "female_character_side_view", "back": "female_character_back_view",
               "front": "female_character"},
}


def spans(path):
    """Per-row first/last opaque pixel, in pixels."""
    a = np.asarray(Image.open(path).split()[-1])
    H, W = a.shape
    op = a > ALPHA_CUT
    rows = np.where(op.any(axis=1))[0]
    lo = np.full(H, np.nan)
    hi = np.full(H, np.nan)
    for y in rows:
        xs = np.where(op[y])[0]
        lo[y], hi[y] = xs[0], xs[-1]
    return lo, hi, W, H, rows


def neck_row(lo, hi, rows):
    """The neck: the narrowest row in the band where a neck can physically be.

    Two earlier attempts failed for instructive reasons. "Steepest width
    increase" lands inside an afro, which keeps widening downward. "Narrowest row
    below the widest" lands at the image bottom, because these are cropped busts
    and the torso leaves the frame. Bounding the search to 30-65% of the
    subject's height is the constraint that actually holds for both heads.
    """
    w = np.nan_to_num(hi - lo).astype(float)
    y0, y1 = int(rows[0]), int(rows[-1])
    span = y1 - y0
    band = np.arange(y0 + int(0.30 * span), y0 + int(0.65 * span))
    k = 21
    ker = np.ones(k) / k
    ww = np.convolve(w[band], ker, mode="valid")
    off = (k - 1) // 2
    return int(band[off + int(np.argmin(ww))])


def head_profile(name, v):
    side = find(v["side"])
    front = find(v["front"])
    slo, shi, sW, sH, srows = spans(side)
    flo, fhi, fW, fH, frows = spans(front)

    s_top, f_top = int(srows[0]), int(frows[0])
    s_sh = neck_row(slo, shi, srows)
    f_sh = neck_row(flo, fhi, frows)
    s_head = s_sh - s_top          # head height in the side shot, px
    f_head = f_sh - f_top          # head height in the front shot, px

    # depth of the head at each normalised height, in units of head height
    ys = np.arange(s_top, s_sh)
    depth_px = (shi - slo)[ys]
    t = (ys - s_top) / max(s_head, 1)
    keep = np.isfinite(depth_px)
    prof = np.stack([t[keep], depth_px[keep] / max(s_head, 1)], axis=1)

    # width of the head at each normalised height, from the FRONT shot, so the
    # two can be compared in the same units
    fys = np.arange(f_top, f_sh)
    width_px = (fhi - flo)[fys]
    ft = (fys - f_top) / max(f_head, 1)
    fkeep = np.isfinite(width_px)
    wprof = np.stack([ft[fkeep], width_px[fkeep] / max(f_head, 1)], axis=1)

    depth_ratio = float(np.nanmax(depth_px) / max(s_head, 1))
    width_ratio = float(np.nanmax(width_px) / max(f_head, 1))
    return {
        "side": {"top": s_top, "neck": s_sh, "headPx": int(s_head), "size": [sW, sH]},
        "front": {"top": f_top, "neck": f_sh, "headPx": int(f_head), "size": [fW, fH]},
        "depthProfile": prof.round(5).tolist(),
        "widthProfile": wprof.round(5).tolist(),
        "depthOverHeight": depth_ratio,
        "widthOverHeight": width_ratio,
        "depthOverWidth": depth_ratio / max(width_ratio, 1e-6),
        "views": {k: str(find(p).relative_to(ROOT)) for k, p in v.items()},
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for name, v in VIEWS.items():
        d = head_profile(name, v)
        (OUT / f"{name}.profile.json").write_text(json.dumps(d))
        print(
            f"{name:7s} head height side {d['side']['headPx']}px front {d['front']['headPx']}px"
            f"  depth/height {d['depthOverHeight']:.3f}"
            f"  width/height {d['widthOverHeight']:.3f}"
            f"  -> depth/width {d['depthOverWidth']:.3f}"
        )


if __name__ == "__main__":
    main()
