"""
P1 (3D rebuild) — Measure the head, in 3D, from the reference photographs.

This replaces the earlier 2.5D image-warp extraction. It emits pure DATA; the
geometry is built in Blender by build_head.py.

MediaPipe FaceLandmarker returns 468 landmarks with real depth (z is relative to
the head centre, on roughly the same scale as x), so the face shape here is
measured from the photograph rather than sculpted by hand. Each expression shot
gives a second measurement of the same face, and the difference between them —
after removing the rigid head movement between frames — is a blendshape.

Output: assets/facedata/<name>.json
  landmarks    468 x 3, neutral, in world units (height 1.0, +z toward viewer)
  shapes       per expression: aligned 468 x 3 and the fitted similarity
  triangles    face-mask topology (indices into the 468), eyes/mouth carved
  silhouette   subject outline traced from the photo's alpha channel
  groups       oval / eyes / iris / lips landmark rings
  iris         centre + radius per eye, for placing real eyeballs

Run:  tools/.venv/bin/python tools/export_face_data.py
"""

import json
import pathlib
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from face_landmarks import (  # noqa: E402
    CHARACTERS,
    MODEL,
    MODEL_URL,
    detect,
    landmark_groups,
    points_in_poly,
    poly_area,
)

ROOT = pathlib.Path(__file__).resolve().parent.parent
REFS = ROOT / "input" / "character_references"
OUT = ROOT / "assets" / "facedata"

SIL_SAMPLES = 520   # points traced around the subject silhouette
SIL_INSET = 0.0015  # pull the outline inward, as a fraction of image height
ALPHA_CUT = 128


# ----------------------------------------------------------------- 3D alignment
def similarity_fit_3d(src, dst, iters=14):
    """Robust 3D similarity (uniform scale, rotation, translation) via Umeyama,
    reweighted so that landmarks which genuinely move under the expression stop
    dragging the fit.

    Without this the rigid head shift between two photographs gets baked into
    every blendshape as a slide of the whole face.
    """
    w = np.ones(len(src))
    out = (1.0, np.eye(3), np.zeros(3))
    for _ in range(iters):
        ws = w.sum()
        mu_s = (src * w[:, None]).sum(0) / ws
        mu_d = (dst * w[:, None]).sum(0) / ws
        a, b = src - mu_s, dst - mu_d
        cov = (a * w[:, None]).T @ b / ws
        U, S, Vt = np.linalg.svd(cov)
        d = np.sign(np.linalg.det(Vt.T @ U.T))
        D = np.diag([1.0, 1.0, d])
        R = Vt.T @ D @ U.T
        var = (w[:, None] * a**2).sum() / ws
        scale = float((S * np.diag(D)).sum() / max(var, 1e-12))
        t = mu_d - scale * (R @ mu_s)
        out = (scale, R, t)
        res = np.linalg.norm(dst - (scale * (src @ R.T) + t), axis=1)
        sigma = max(np.median(res), 1e-6)
        w = 1.0 / (1.0 + (res / (2.5 * sigma)) ** 2)
    return out


def apply_sim3(pts, params):
    scale, R, t = params
    return scale * (pts @ R.T) + t


# ------------------------------------------------------------------- silhouette
def trace_silhouette(img, n=SIL_SAMPLES):
    """Outline of the subject, as a closed loop in normalized image space.

    The references are cutouts, so the alpha channel already holds the exact
    outline; marching it beats guessing where the shoulders end. Traced by
    scanning each row for the first and last opaque pixel, which is robust for a
    bust (one connected span per row) and needs no contour library.
    """
    a = np.asarray(img.split()[-1])
    H, W = a.shape
    rows = np.where((a > ALPHA_CUT).any(axis=1))[0]
    if not len(rows):
        raise RuntimeError("reference image has no opaque pixels")
    y0, y1 = rows[0], rows[-1]

    ys = np.linspace(y0, y1, n // 2).astype(int)
    left, right = [], []
    for y in ys:
        xs = np.where(a[y] > ALPHA_CUT)[0]
        if not len(xs):
            continue
        left.append((xs[0] / W, y / H))
        right.append((xs[-1] / W, y / H))
    loop = np.array(left + right[::-1], dtype=np.float64)

    # Pull the outline INWARD slightly. The trace is a polygon through sampled
    # rows, so between samples it cuts straight across a curved edge and bulges
    # a little outside the true alpha boundary. Those bulges get alpha-tested
    # away at render time, and whatever sits behind shows through them — which
    # appeared as a head-shaped grey halo around the hair.
    c = loop.mean(axis=0)
    d = loop - c
    n = np.linalg.norm(d, axis=1, keepdims=True)
    return loop - d / np.maximum(n, 1e-9) * SIL_INSET


# ----------------------------------------------------------------------- export
def to_world(lm, aspect):
    """normalized image landmarks -> world units (image height = 1, +z to viewer)"""
    return np.stack(
        [(lm[:, 0] - 0.5) * aspect, 0.5 - lm[:, 1], -lm[:, 2] * aspect], axis=1
    )


def face_topology(lm2d, groups, aspect):
    """Triangulate the face mask in frontal projection, then carve the eye and
    mouth openings so real eyeballs and a real mouth bag show through."""
    from scipy.spatial import Delaunay

    tri = Delaunay(lm2d * [aspect, 1.0])
    cen = lm2d[tri.simplices].mean(axis=1)
    keep = points_in_poly(cen, lm2d[groups["faceOval"]])
    for hole in ("leftEye", "rightEye", "lipsInner"):
        keep &= ~points_in_poly(cen, lm2d[groups[hole]])
    return tri.simplices[keep]


def circle_of(pts):
    c = pts.mean(axis=0)
    return c, float(np.linalg.norm(pts - c, axis=1).mean())


def build(name, cfg):
    base = REFS / cfg["dir"]
    src = base / cfg["neutral"]
    lm, W, H = detect(src)
    aspect = W / H

    groups = landmark_groups()
    rings = groups.pop("_lipRings")
    a0, a1 = (poly_area(lm[r][:, :2]) for r in rings)
    groups["lipsInner"], groups["lipsOuter"] = (
        (rings[0], rings[1]) if a0 < a1 else (rings[1], rings[0])
    )

    lm2d = lm[:, :2]
    verts = to_world(lm, aspect)
    tris = face_topology(lm2d, groups, aspect)

    shapes = {}
    for ename, fn in cfg["expressions"].items():
        f = base / fn
        if not f.exists():
            print(f"  ! {name}: missing {fn}")
            continue
        elm, eW, eH = detect(f)
        if (eW, eH) != (W, H):
            raise RuntimeError(f"{ename}: {eW}x{eH} != neutral {W}x{H}")
        ev = to_world(elm, aspect)
        scale, R, t = similarity_fit_3d(ev, verts)
        aligned = apply_sim3(ev, (scale, R, t))
        delta = aligned - verts
        shapes[ename] = {
            "positions": aligned.round(6).tolist(),
            # kept so the runtime can map this shot's own pixels if ever needed
            "similarity": {"scale": scale, "R": R.round(8).tolist(), "t": t.round(8).tolist()},
            "maxDelta": float(np.abs(delta).max()),
            "meanDelta": float(np.linalg.norm(delta, axis=1).mean()),
        }

    img = Image.open(src)
    sil = trace_silhouette(img)
    lc, lr = circle_of(verts[groups["leftIris"]])
    rc, rr = circle_of(verts[groups["rightIris"]])

    OUT.mkdir(parents=True, exist_ok=True)
    data = {
        "name": name,
        "image": {"width": W, "height": H, "aspect": aspect},
        "landmarks": verts.round(6).tolist(),
        "triangles": tris.astype(int).tolist(),
        "groups": {k: list(map(int, v)) for k, v in groups.items()},
        "silhouette": sil.round(6).tolist(),
        "iris": {
            "left": {"c": lc.round(6).tolist(), "r": lr},
            "right": {"c": rc.round(6).tolist(), "r": rr},
        },
        "shapes": shapes,
        "neutralImage": str(src.relative_to(ROOT)),
        "expressionImages": {
            e: str((base / fn).relative_to(ROOT))
            for e, fn in cfg["expressions"].items()
            if (base / fn).exists()
        },
    }
    (OUT / f"{name}.json").write_text(json.dumps(data))

    zmin, zmax = verts[:, 2].min(), verts[:, 2].max()
    print(
        f"{name:7s} {len(verts)} landmarks  {len(tris)} face tris  "
        f"depth range {zmax - zmin:.4f}  silhouette {len(sil)} pts  "
        f"shapes {len(shapes)}"
    )
    for e, sh in shapes.items():
        print(f"   {e:11s} max {sh['maxDelta']:.4f}  mean {sh['meanDelta']:.4f}")
    return data


def main():
    if not MODEL.exists():
        import urllib.request

        print("downloading face_landmarker.task ...")
        urllib.request.urlretrieve(MODEL_URL, MODEL)

    for name, cfg in CHARACTERS.items():
        build(name, cfg)


if __name__ == "__main__":
    main()
