"""
Register the side and back photographs against the model.

Each reference view has to become a projection: given a point on the model,
where does it land in that photograph? Two different problems:

  SIDE  MediaPipe detects a face in both side shots (the male's is a 3/4 view,
        the female's a true profile), so the SAME 468 landmarks exist in the
        side shot and in the neutral. A robust 3D similarity between the two
        recovers the head's pose directly from the data — no angle to guess.

  BACK  There is no face to detect. Registered instead on the silhouette: the
        head's top, the neck, and the head's horizontal centre and width, which
        are all visible from behind. Mirrored, because we are looking at the
        subject from the opposite side.

Output: assets/facedata/<name>.views.json

Run:  tools/.venv/bin/python tools/register_views.py
"""

import json
import pathlib
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from face_landmarks import REFS, detect  # noqa: E402
from export_face_data import similarity_fit_3d, to_world  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA = ROOT / "assets" / "facedata"
ALPHA_CUT = 128

VIEWS = {
    "male": {"side": "male_character_side_view", "back": "male_character_back_view"},
    "female": {"side": "female_character_side_view", "back": "female_character_back_view"},
}


def find(stem):
    hits = sorted(REFS.glob(f"*/{stem}.png"))
    if not hits:
        raise FileNotFoundError(f"no reference named {stem}.png")
    return hits[0]


def alpha_spans(path):
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
    """Narrowest row in the band where a neck can physically be (30-65% of the
    subject's height). See profile_depth.py for why the obvious alternatives
    fail on an afro and on a cropped bust."""
    w = np.nan_to_num(hi - lo).astype(float)
    y0, y1 = int(rows[0]), int(rows[-1])
    span = y1 - y0
    band = np.arange(y0 + int(0.30 * span), y0 + int(0.65 * span))
    k = 21
    ww = np.convolve(w[band], np.ones(k) / k, mode="valid")
    return int(band[(k - 1) // 2 + int(np.argmin(ww))])


def register_back(name, stem, fd):
    """Orthographic projection from behind, fitted on head top / neck / centre."""
    back = find(stem)
    blo, bhi, bW, bH, brows = alpha_spans(back)
    b_top = int(brows[0])
    b_neck = neck_row(blo, bhi, brows)

    front = ROOT / fd["neutralImage"]
    flo, fhi, fW, fH, frows = alpha_spans(front)
    f_top = int(frows[0])
    f_neck = neck_row(flo, fhi, frows)

    aspect = fd["image"]["aspect"]
    # model-space heights of the front shot's head top and neck
    y_top = 0.5 - f_top / fH
    y_neck = 0.5 - f_neck / fH

    # vertical: model y -> back image v, linear through those two landmarks
    v_top = b_top / bH
    v_neck = b_neck / bH
    sv = (v_neck - v_top) / (y_neck - y_top)
    ov = v_top - sv * y_top

    # horizontal: match the head's centre and half-width over the head rows
    def head_centre_halfwidth(lo, hi, top, neck, W):
        ys = np.arange(top, neck)
        c = np.nanmean((lo[ys] + hi[ys]) * 0.5) / W
        hw = np.nanmax((hi[ys] - lo[ys])) * 0.5 / W
        return float(c), float(hw)

    bc, bhw = head_centre_halfwidth(blo, bhi, b_top, b_neck, bW)
    fc, fhw = head_centre_halfwidth(flo, fhi, f_top, f_neck, fW)
    # model x of the front head centre and half-width, in world units
    x_c = (fc - 0.5) * aspect
    x_hw = fhw * aspect
    # MIRRORED: seen from behind, the subject's left is on the other side
    su = -(bhw / max(x_hw, 1e-9))
    ou = bc - su * x_c

    return {
        "kind": "ortho",
        "image": str(back.relative_to(ROOT)),
        "size": [bW, bH],
        # u = su * x + ou ;  v = sv * y + ov   (v measured top-down)
        "su": su, "ou": ou, "sv": sv, "ov": ov,
        "headTopRow": b_top, "neckRow": b_neck,
    }


def register_side(name, stem, fd):
    """Similarity fitted on the landmarks the two shots share."""
    side = find(stem)
    slm, sW, sH = detect(side)
    aspect_s = sW / sH
    sv = to_world(slm, aspect_s)
    model = np.array(fd["landmarks"])

    # model -> side-image space (the direction we need to sample the photo)
    scale, R, t = similarity_fit_3d(model, sv)
    res = np.linalg.norm(sv - (scale * (model @ R.T) + t), axis=1)
    yaw = float(np.degrees(np.arctan2(R[0, 2], R[2, 2])))

    return {
        "kind": "similarity",
        "image": str(side.relative_to(ROOT)),
        "size": [sW, sH],
        "aspect": aspect_s,
        "scale": scale,
        "R": R.round(8).tolist(),
        "t": t.round(8).tolist(),
        "yawDeg": yaw,
        "fitResidualMedian": float(np.median(res)),
        "fitResidualP90": float(np.percentile(res, 90)),
    }


def main():
    for name, v in VIEWS.items():
        fd = json.loads((DATA / f"{name}.json").read_text())
        out = {
            "name": name,
            "front": {
                "kind": "frontal",
                "image": fd["neutralImage"],
                "aspect": fd["image"]["aspect"],
            },
            "side": register_side(name, v["side"], fd),
            "back": register_back(name, v["back"], fd),
        }
        (DATA / f"{name}.views.json").write_text(json.dumps(out, indent=1))
        s, b = out["side"], out["back"]
        print(
            f"{name:7s} side yaw {s['yawDeg']:+6.1f}deg  fit median {s['fitResidualMedian']:.4f} "
            f"p90 {s['fitResidualP90']:.4f}   |  back su {b['su']:+.3f} ou {b['ou']:.3f} "
            f"sv {b['sv']:+.3f} ov {b['ov']:.3f}"
        )


if __name__ == "__main__":
    main()
