"""
P2 (3D rebuild) — Turn the measured landmarks into a solid bust.

Computation lives here rather than in build_head.py because Blender ships numpy
but not scipy, and the interpolation this needs is worth doing properly.

The result is ONE closed mesh. That matters: the previous 2.5D build composited
eye patches, a mouth underlay and teeth plates over a face plane, and those
overlays are exactly what read as "skin within a skin". A single surface cannot
overlay itself.

Shape:
  front   real depth — measured landmark z inside the face, a fitted skull/hair
          dome around it, a shallow torso below the neck
  edge    the subject's own alpha silhouette, so there is no background plane
  back    the silhouette swept backwards and capped, making it a closed solid

Output: assets/facedata/<name>.bust.npz

Run:  tools/.venv/bin/python tools/build_bust_geometry.py
"""

import json
import pathlib

import numpy as np
from PIL import Image
from scipy.interpolate import RBFInterpolator
from scipy.spatial import Delaunay

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA = ROOT / "assets" / "facedata"

EYE_APERTURE = 0.80   # eye opening as a fraction of the landmark ring
# Shapes kept in the shipped model. The others are still measured from the
# photographs (the measurement is cheap and the data is worth keeping) but are
# dropped before packing, because each one costs a full morph target.
KEEP_SHAPES = ("wink", "blink")
# With no mouth interior to reveal, the mouth must NOT be carved: a hole with
# nothing behind it looks straight through the head.
CARVE_MOUTH = False
TARGET_VERTS = 17000  # front-surface budget, held equal across characters so a
                      # taller portrait does not silently cost twice the mesh
SKULL_K = 2.05      # profile reaches zero this many head-radii out
# Head depth / head width. Measured from the side-view silhouette by
# profile_depth.py, then clamped: the silhouette registration is only good to
# about 10%, and a head outside this range is anatomically wrong whatever the
# pixels say.
DEPTH_OVER_WIDTH = (1.00, 1.40)
TORSO_A = 0.34      # torso is much shallower than the head
# (scale toward centre, depth in head radii). Tapers fast: a slow taper leaves a
# wide bulb behind the head whose side swings into view as soon as the head
# turns, reading as a grey paddle stuck to the ear. A skull is much narrower than
# the silhouette almost immediately behind the face.
BACK_RINGS = ((0.96, 0.05), (0.86, 0.26), (0.62, 0.54), (0.34, 0.74), (0.12, 0.88))
# The FIRST step must be small. A big first step makes the band from the
# silhouette rim to the first ring wide and almost perpendicular to the screen:
# invisible head-on, but it swings into view the moment the head moves and reads
# as a smeared grey lobe stuck to the ear.
SHAPE_FALLOFF = 2.4 # how far expression motion carries past the face, in head radii


def load(name):
    return json.loads((DATA / f"{name}.json").read_text())


def poly_path(pts):
    """even-odd point-in-polygon for a batch of points"""
    def inside(q):
        x, y = q[:, 0], q[:, 1]
        res = np.zeros(len(q), dtype=bool)
        j = len(pts) - 1
        for i in range(len(pts)):
            xi, yi = pts[i]
            xj, yj = pts[j]
            dy = yj - yi
            res ^= ((yi > y) != (yj > y)) & (
                x < (xj - xi) * (y - yi) / (dy if dy else 1e-12) + xi
            )
            j = i
        return res
    return inside


def smooth01(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0 + 1e-12), 0, 1)
    return t * t * (3 - 2 * t)


def radial_ratio(pts, poly, center):
    """How far out each point lies along its own direction, in units of the
    polygon's radius in that direction: <1 inside, 1 on the outline.

    A plain distance/mean-radius is wrong for a face oval, which is far taller
    than it is wide — the chin sits at ~1.3x the mean radius while still being
    well inside the face. Blending the measured depth into the fitted dome on
    that measure put the seam in the wrong place and left a hard rim standing
    proud all the way round the face.
    """
    rel = poly - center
    ang = np.arctan2(rel[:, 1], rel[:, 0])
    rad = np.hypot(rel[:, 0], rel[:, 1])
    o = np.argsort(ang)
    ang, rad = ang[o], rad[o]
    ang = np.concatenate([ang - 2 * np.pi, ang, ang + 2 * np.pi])
    rad = np.concatenate([rad, rad, rad])
    q = pts - center
    return np.hypot(q[:, 0], q[:, 1]) / np.maximum(
        np.interp(np.arctan2(q[:, 1], q[:, 0]), ang, rad), 1e-9
    )


def relax_depth(z, faces, pin_target, pin_weight, iters=40, lam=0.55):
    """Laplacian relaxation of the depth field, with the measured landmarks
    pinned.

    The raw field is two surfaces stitched together — measured inside the face,
    fitted outside — and the join shows as a step, while MediaPipe's per-landmark
    z is noisy enough to facet the cheeks. Relaxing everything except the
    measurements smooths both without flattening the relief that was actually
    recorded.
    """
    n = len(z)
    nbr_sum = np.zeros(n)
    nbr_cnt = np.zeros(n)
    a, b, c = faces[:, 0], faces[:, 1], faces[:, 2]
    for i, j in ((a, b), (b, c), (c, a)):
        np.add.at(nbr_sum, i, z[j])
        np.add.at(nbr_cnt, i, 1)
        np.add.at(nbr_sum, j, z[i])
        np.add.at(nbr_cnt, j, 1)
    idx_i = np.concatenate([a, b, c])
    idx_j = np.concatenate([b, c, a])
    z = z.copy()
    for _ in range(iters):
        s = np.zeros(n)
        cnt = np.zeros(n)
        np.add.at(s, idx_i, z[idx_j])
        np.add.at(cnt, idx_i, 1)
        np.add.at(s, idx_j, z[idx_i])
        np.add.at(cnt, idx_j, 1)
        avg = np.where(cnt > 0, s / np.maximum(cnt, 1), z)
        z = z + lam * (avg - z)
        # SOFT constraint. Pinning every landmark hard, including the ones on the
        # face oval, spikes the boundary: those sit exactly where the measured
        # face meets the fitted dome, so holding them fixed leaves a ring of
        # spurs. The weight fades out across the oval, and staying just under 1
        # lets relaxation take the noise out of MediaPipe's per-landmark z.
        z = z * (1 - pin_weight) + pin_target * pin_weight
    return z


def poly_area_xy(p):
    return 0.5 * abs(np.dot(p[:, 0], np.roll(p[:, 1], 1)) - np.dot(p[:, 1], np.roll(p[:, 0], 1)))


def head_frame(lm, groups):
    oval = lm[groups["faceOval"]]
    c = oval.mean(axis=0)
    r = float(np.linalg.norm(oval[:, :2] - c[:2], axis=1).mean())
    return c, r


def load_profile(name):
    f = DATA / f"{name}.profile.json"
    return json.loads(f.read_text()) if f.exists() else None


def depth_field(lm, groups, xy, hc, hr, prof=None):
    """Depth for every front-surface sample.

    Inside the face the value is measured, so the nose, brows and lips keep the
    relief the photograph actually recorded. Outside it, a dome stands in for the
    skull and hair and a shallow cylinder for the torso — neither is visible from
    the front, they exist so the head has volume to rotate with.
    """
    # Thin-plate spline, not linear interpolation. 468 landmarks over ~16k
    # vertices interpolated linearly gives a piecewise-flat surface — visibly
    # facetted across the cheeks — and relaxing that afterwards turns the facets
    # into lumps. A TPS fit is smooth everywhere by construction, and a little
    # smoothing absorbs the noise in MediaPipe's per-landmark z instead of
    # honouring it as geometry.
    rbf = RBFInterpolator(
        lm[:, :2], lm[:, 2], kernel="thin_plate_spline", smoothing=2e-6, neighbors=96
    )
    zf = rbf(xy)
    # CLAMP the extrapolation. A thin-plate spline is unbounded outside the hull
    # of its data: a short way past the face it runs off to values the face never
    # had, and blending that in dragged the outer surface back to z=-0.33 while
    # it stayed full width — a wide skirt hiding behind the head that swings out
    # as a lobe beside the ear the moment the head turns. Inside the face the
    # clamp never binds; outside it is the difference between interpolation and
    # invention.
    pad = 0.25 * (lm[:, 2].max() - lm[:, 2].min())
    zf = np.clip(zf, lm[:, 2].min() - pad, lm[:, 2].max() + pad)

    d = radial_ratio(xy, lm[groups["faceOval"], :2], hc[:2])

    # Skull depth from the side-view silhouette rather than a guessed dome: the
    # profile photograph literally is this head's depth at every height.
    ratio = 1.22
    if prof:
        ratio = float(np.clip(prof["depthOverWidth"], *DEPTH_OVER_WIDTH))
    half_depth = 0.5 * ratio * (2 * hr)
    if prof and prof.get("depthProfile"):
        pr = np.array(prof["depthProfile"])
        top = xy[:, 1].max()
        t = np.clip((top - xy[:, 1]) / max(top - (hc[1] - hr * 1.15), 1e-6), 0, 1)
        shape = np.interp(t, pr[:, 0], pr[:, 1])
        shape /= max(shape.max(), 1e-6)
    else:
        shape = np.ones(len(xy))
    dome = half_depth * shape * np.sqrt(np.clip(1.0 - (d / SKULL_K) ** 2, 0, 1))

    # torso: a shallow cylinder about the vertical axis, fading in below the jaw
    jaw_y = lm[groups["faceOval"], 1].min()
    dx = np.abs(xy[:, 0] - hc[0]) / (hr * 2.6)
    torso = TORSO_A * hr * np.sqrt(np.clip(1.0 - dx**2, 0, 1))
    below = np.clip((jaw_y - xy[:, 1]) / (hr * 0.55), 0, 1)
    env = dome * (1 - below) + torso * below

    # Make the skull MEET the face at the oval instead of merely fading into it.
    # Two independently fitted surfaces do not agree at their shared boundary, so
    # cross-fading them leaves a step — a ridge standing proud all the way round
    # the face, which is what kept reappearing. Measuring the disagreement on the
    # seam ring and decaying that correction outwards makes them continuous.
    seam = (d > 0.92) & (d < 1.08)
    if seam.sum() > 20:
        offset = float(np.median(zf[seam] - env[seam]))
        # CLAMPED, and decayed over a short range. Unclamped this quietly
        # dragged the entire surround backwards — the outermost vertices at ear
        # height ended up at z=-0.35, a flared skirt that swings out as a lobe
        # beside the ear the moment the head turns. The offset only has a job
        # right at the seam; past the head it is not a correction, it is a dent.
        lim = 0.10 * hr
        offset = float(np.clip(offset, -lim, lim))
        env = env + offset * np.clip(1.0 - (d - 1.0) / 0.45, 0, 1)

    # then a wide cross-fade, so any residual difference is spread over the
    # whole side of the head rather than concentrated at one contour
    t = np.clip((d - 0.95) / (1.35 - 0.95), 0, 1)
    w = 1.0 - t * t * (3 - 2 * t)
    return zf * w + env * (1 - w), w, d


def build(name):
    fd = load(name)
    prof = load_profile(name)
    lm = np.array(fd["landmarks"])
    groups = fd["groups"]
    sil = np.array(fd["silhouette"])
    aspect = fd["image"]["aspect"]

    # silhouette to world units, matching to_world()
    sil_w = np.stack([(sil[:, 0] - 0.5) * aspect, 0.5 - sil[:, 1]], axis=1)
    hc, hr = head_frame(lm, groups)

    # --- front surface sample points -----------------------------------------
    x0, y0 = sil_w.min(axis=0)
    x1, y1 = sil_w.max(axis=0)
    # square cells, count chosen from the silhouette's area so both characters
    # land near the same budget
    area = float(poly_area_xy(sil_w))
    step = np.sqrt(max(area, 1e-9) / TARGET_VERTS)
    gx = np.arange(x0, x1 + step, step)
    gy = np.arange(y0, y1 + step, step)
    G = np.stack(np.meshgrid(gx, gy), -1).reshape(-1, 2)

    inside = poly_path(sil_w)
    G = G[inside(G)]
    # the landmarks themselves must be vertices, or the blendshapes would only
    # ever be an approximation of the measurement
    pts = np.vstack([lm[:, :2], sil_w, G])
    n_lm, n_sil = len(lm), len(sil_w)

    # drop grid points that crowd a landmark or the outline
    from scipy.spatial import cKDTree

    keep = np.ones(len(pts), dtype=bool)
    tree = cKDTree(pts[: n_lm + n_sil])
    close = tree.query_ball_point(pts[n_lm + n_sil:], r=hr * 0.035)
    keep[n_lm + n_sil:] = [len(c) == 0 for c in close]
    pts = pts[keep]

    tri = Delaunay(pts)
    cen = pts[tri.simplices].mean(axis=1)
    faces = tri.simplices[inside(cen)]

    # Carve the eye and mouth openings. This is a solid head now, so what shows
    # through them is real geometry — eyeballs, a mouth bag, teeth, a tongue —
    # not a sprite composited on top. That distinction is the whole point of the
    # rebuild: an overlay is what produced the skin-within-skin artefact.
    lm2 = lm[:, :2]
    cen_f = pts[faces].mean(axis=1)
    holes = ("leftEye", "rightEye") + (("lipsInner",) if CARVE_MOUTH else ())
    for hole in holes:
        ring = lm2[groups[hole]]
        if hole.endswith("Eye"):
            # Shrink the eye aperture. The palpebral opening is WIDER than the
            # eyeball (~30mm across a ~24mm globe), so carving it out to the
            # corners leaves a gap no sphere can fill and you see straight
            # through the canthus. In a real face the corners are lid, not
            # eyeball — so they should stay as surface.
            c = ring.mean(axis=0)
            ring = (ring - c) * EYE_APERTURE + c
        faces = faces[~poly_path(ring)(cen_f)]
        cen_f = pts[faces].mean(axis=1)

    z, wface, dhead = depth_field(lm, groups, pts, hc, hr, prof)
    # landmarks keep their measured depth exactly, and stay pinned through the
    # relaxation so the nose, brows and lips survive it
    z[:n_lm] = lm[:, 2]
    # Hold the FITTED surface inside the face, not the raw landmark values.
    #   pinning raw landmark z  -> a dimple at each of the 468, a bumpy face
    #   pinning nothing         -> relaxation erodes the nose and brows flat
    # The thin-plate fit is what should survive; relaxation only has a job at the
    # join between the fitted face and the fitted skull, which is where wface
    # falls away.
    z = relax_depth(z, faces, z.copy(), 0.92 * wface)
    front = np.column_stack([pts, z])

    # --- close the back ------------------------------------------------------
    # Concentric rings sweeping backwards and inwards, then a small cap: the head
    # gets an actual skull behind it. A single flat cap reads as a cut-out card
    # the moment the head turns even slightly.
    ring_idx = np.arange(n_lm, n_lm + n_sil)[keep[n_lm:n_lm + n_sil]]
    m = len(ring_idx)
    z0 = float(lm[:, 2].mean())
    verts = front
    prev = ring_idx
    quads = []

    # A SMOOTHED outline for the skull. Sweeping the raw silhouette straight
    # back carries its local bulges with it, and the male's ears stick out — so
    # the ear's width extends backwards as a slab and reads as a paddle stuck to
    # the side of the head the moment it turns. The skull behind the ears is a
    # smooth oval, so the rings follow a low-passed outline; the first ring still
    # starts from the true silhouette so the surface stays closed, and the ears
    # fade out over the first two steps instead of being extruded.
    base = sil_w[keep[n_lm:n_lm + n_sil]]
    rel = base - hc[:2]
    theta = np.arctan2(rel[:, 1], rel[:, 0])
    rad = np.hypot(rel[:, 0], rel[:, 1])
    order = np.argsort(theta)
    kern = np.ones(21) / 21
    sm = np.convolve(np.r_[rad[order][-30:], rad[order], rad[order][:30]], kern, "same")[30:-30]
    rad_s = np.empty_like(rad)
    rad_s[order] = sm
    smooth = hc[:2] + np.column_stack([np.cos(theta), np.sin(theta)]) * rad_s[:, None]
    blend = (0.30, 0.75, 1.0, 1.0, 1.0)

    for ri, (k, dep) in enumerate(BACK_RINGS):
        src = base * (1 - blend[min(ri, len(blend) - 1)]) + smooth * blend[min(ri, len(blend) - 1)]
        ring = (src - hc[:2]) * k + hc[:2]
        rz = np.full(len(ring), z0 - dep * hr)
        start = len(verts)
        verts = np.vstack([verts, np.column_stack([ring, rz])])
        cur = np.arange(start, start + m)
        for i in range(m):
            a, b = prev[i], prev[(i + 1) % m]
            c, d = cur[i], cur[(i + 1) % m]
            quads += [[a, b, d], [a, d, c]]
        prev = cur
    centre = np.array([[hc[0], hc[1], z0 - (BACK_RINGS[-1][1] + 0.06) * hr]])
    verts = np.vstack([verts, centre])
    cap = len(verts) - 1
    for i in range(m):
        quads.append([prev[(i + 1) % m], prev[i], cap])

    back_tris = np.array(quads, dtype=int)
    tris = np.vstack([faces, back_tris])
    # material 0 = the photograph, material 1 = the unseen shell. The sweep can
    # only inherit stretched edge pixels, and smeared skin on a turning head is
    # far more noticeable than a plain dark interior.
    mat = np.concatenate([np.zeros(len(faces), np.int32), np.ones(len(back_tris), np.int32)])

    # --- UV: frontal projection ----------------------------------------------
    # The photo was taken from the front, so projecting it from the front is the
    # only mapping that reproduces it exactly. Everything the camera cannot see
    # from there inherits stretched edge pixels, which is what a photogrammetric
    # capture does too.
    uv = np.stack([verts[:, 0] / aspect + 0.5, verts[:, 1] + 0.5], axis=1)

    # --- blendshapes ---------------------------------------------------------
    # Landmark deltas are exact; everything else is carried by inverse-distance
    # weighting from the nearest landmarks, faded out by distance from the head
    # so the torso never moves when the face does.
    tree_lm = cKDTree(lm[:, :2])
    K = 6
    dist, idx = tree_lm.query(verts[:, :2], k=K)
    dist = np.maximum(dist, 1e-6)
    wts = 1.0 / dist**2
    wts /= wts.sum(axis=1, keepdims=True)

    dh = np.linalg.norm(verts[:, :2] - hc[:2], axis=1) / hr
    carry = np.clip(1.0 - (dh / SHAPE_FALLOFF) ** 2, 0, 1)
    carry[:n_lm] = 1.0

    shapes = {}
    for ename, sh in fd["shapes"].items():
        target = np.array(sh["positions"])
        d_lm = target - lm
        d_all = (d_lm[idx] * wts[..., None]).sum(axis=1) * carry[:, None]
        d_all[:n_lm] = d_lm            # measured vertices move exactly as measured
        shapes[ename] = d_all

    # Blink is not in the reference set — there is a wink, which closes one eye.
    # Mirroring it about the face's own midline gives the other eye, and the sum
    # is a symmetric blink measured from this person's real eyelid rather than
    # invented. Matching is nearest-neighbour on mirrored position because the
    # mesh is a sampled grid and has no symmetric topology to exploit.
    if "wink" in shapes:
        mid = float(lm[:, 0].mean())
        mirrored_xy = np.column_stack([2 * mid - verts[:, 0], verts[:, 1]])
        tree_v = cKDTree(verts[:, :2])
        dist_m, idx_m = tree_v.query(mirrored_xy, k=1)
        w = shapes["wink"]
        mirror = w[idx_m].copy()
        mirror[:, 0] *= -1                     # x flips under reflection
        # drop matches that are too far to be a real mirror partner
        bad = dist_m > hr * 0.06
        mirror[bad] = 0.0
        blink = w + mirror
        # the two eyelids must not double up where they overlap at the midline
        shapes["blink"] = np.clip(blink, -np.abs(w).max() * 2, np.abs(w).max() * 2)
        print(f"   blink derived from wink: {int((~bad).sum())}/{len(verts)} mirror matches")

    shapes = {k: v for k, v in shapes.items() if k in KEEP_SHAPES}

    out = DATA / f"{name}.bust.npz"
    np.savez_compressed(
        out,
        verts=verts.astype(np.float32),
        tris=tris.astype(np.int32),
        mat=mat,
        uv=uv.astype(np.float32),
        n_landmarks=n_lm,
        head_center=hc,
        head_radius=hr,
        shape_names=np.array(list(shapes.keys())),
        lips_inner=lm[groups["lipsInner"]].astype(np.float32),
        lips_outer=lm[groups["lipsOuter"]].astype(np.float32),
        face_oval=lm[groups["faceOval"]].astype(np.float32),
        iris_l=np.array(fd["iris"]["left"]["c"] + [fd["iris"]["left"]["r"]], np.float32),
        iris_r=np.array(fd["iris"]["right"]["c"] + [fd["iris"]["right"]["r"]], np.float32),
        # inner-lip aperture per expression, so a jaw bone can follow the
        # measured opening instead of being animated by eye
        apertures=np.array([
            float(np.array(sh["positions"])[groups["lipsInner"], 1].max()
                  - np.array(sh["positions"])[groups["lipsInner"], 1].min())
            for sh in fd["shapes"].values()], np.float32),
        aperture_neutral=float(lm[groups["lipsInner"], 1].max() - lm[groups["lipsInner"], 1].min()),
        **{f"shape_{k}": v.astype(np.float32) for k, v in shapes.items()},
    )
    print(
        f"{name:7s} {len(verts):6d} verts  {len(tris):6d} tris  "
        f"front {len(front)}  back {len(back_tris)}tris  shapes {len(shapes)}  "
        f"headR {hr:.4f}  depth {verts[:, 2].min():.3f}..{verts[:, 2].max():.3f}  "
        f"-> {out.name} {out.stat().st_size / 1024:.0f} KB"
    )


if __name__ == "__main__":
    for n in ("male", "female"):
        build(n)
