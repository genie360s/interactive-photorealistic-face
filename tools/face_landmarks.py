"""
Landmark layer: detection, region groups, and the character/reference map.

Everything geometric lives in the 3D pipeline
(export_face_data.py -> build_bust_geometry.py -> build_head.py); this module is
just the measurement front end they share.
"""

import pathlib

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
REFS = ROOT / "input" / "character_references"
MODEL = ROOT / "tools" / "face_landmarker.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)

# Characters and their expression shots. The neutral defines the topology; every
# expression is the SAME person in the SAME frame, so its landmarks become a
# measured blendshape rather than an authored one.
CHARACTERS = {
    "male": {
        "dir": "male_characters",
        "neutral": "male_character.png",
        "expressions": {
            "smile": "male_character_smile.png",
            "smileTeeth": "male_character_smile_with_teeth.png",
            "laugh": "male_character_laugh.png",
            "cry": "male_character_cry.png",
            "fear": "male_character_fear.png",
            "kiss": "male_character_kiss.png",
            "wink": "male_character_wink.png",
            "tongue": "male_character_boo_tongue_out.png",
        },
    },
    "female": {
        "dir": "female_characters",
        "neutral": "female_character.png",
        # same eight expressions, named slightly differently on disk
        "expressions": {
            "smile": "female_character_smile.png",
            "smileTeeth": "female_character_smile_with_teeth.png",
            "laugh": "female_character_laugh.png",
            "cry": "female_character_crying.png",
            "fear": "female_character_fear.png",
            "kiss": "female_character_kiss.png",
            "wink": "female_character_wink.png",
            "tongue": "female_character_boo_with_tongue.png",
        },
    },
}


# ---------------------------------------------------------------- landmark loops
def ordered_loop(edges):
    """MediaPipe ships region outlines as unordered sets of (a,b) edges. Walk the
    adjacency to recover the closed ring in order."""
    adj = {}
    for a, b in edges:
        adj.setdefault(a, []).append(b)
        adj.setdefault(b, []).append(a)
    start = min(adj)
    loop, prev, cur = [start], None, start
    while True:
        nxts = [n for n in adj[cur] if n != prev]
        if not nxts:
            break
        nxt = nxts[0]
        if nxt == start:
            break
        loop.append(nxt)
        prev, cur = cur, nxt
        if len(loop) > len(adj) + 2:
            raise RuntimeError("loop walk did not terminate")
    return loop


def split_rings(edges):
    """A lip connection set is two disjoint cycles (outer vermillion and the
    inner opening). Return each, ordered."""
    adj = {}
    for a, b in edges:
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)
    seen, comps = set(), []
    for node in adj:
        if node in seen:
            continue
        stack, comp = [node], []
        while stack:
            n = stack.pop()
            if n in seen:
                continue
            seen.add(n)
            comp.append(n)
            stack.extend(adj[n] - seen)
        comps.append(set(comp))
    return [
        ordered_loop(frozenset({(a, b) for a, b in edges if a in c and b in c}))
        for c in comps
    ]


def landmark_groups():
    """Read the region definitions out of MediaPipe itself rather than pasting
    index lists from memory — the package is the source of truth.

    Note: the legacy `solutions.face_mesh_connections` module is gone from the
    builds shipped for this platform; the connection lists now live on the tasks
    API as Connection dataclasses (verified against mediapipe 0.10.35).
    """
    from mediapipe.tasks.python.vision.face_landmarker import FaceLandmarksConnections as C

    e = lambda conns: frozenset((c.start, c.end) for c in conns)

    lip_rings = split_rings(e(C.FACE_LANDMARKS_LIPS))
    if len(lip_rings) != 2:
        raise RuntimeError(f"expected 2 lip rings, got {len(lip_rings)}")
    return {
        "faceOval": ordered_loop(e(C.FACE_LANDMARKS_FACE_OVAL)),
        "leftEye": ordered_loop(e(C.FACE_LANDMARKS_LEFT_EYE)),
        "rightEye": ordered_loop(e(C.FACE_LANDMARKS_RIGHT_EYE)),
        "leftBrow": sorted({i for p in e(C.FACE_LANDMARKS_LEFT_EYEBROW) for i in p}),
        "rightBrow": sorted({i for p in e(C.FACE_LANDMARKS_RIGHT_EYEBROW) for i in p}),
        "leftIris": sorted({i for p in e(C.FACE_LANDMARKS_LEFT_IRIS) for i in p}),
        "rightIris": sorted({i for p in e(C.FACE_LANDMARKS_RIGHT_IRIS) for i in p}),
        "_lipRings": lip_rings,
    }


# ---------------------------------------------------------------- geometry utils
def poly_area(pts):
    x, y = pts[:, 0], pts[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def points_in_poly(pts, poly):
    """Vectorised even-odd ray casting. pts (N,2), poly (M,2) -> bool (N,)"""
    x, y = pts[:, 0], pts[:, 1]
    inside = np.zeros(len(pts), dtype=bool)
    j = len(poly) - 1
    for i in range(len(poly)):
        xi, yi = poly[i]
        xj, yj = poly[j]
        dy = yj - yi
        crosses = ((yi > y) != (yj > y)) & (
            x < (xj - xi) * (y - yi) / (dy if dy != 0 else 1e-12) + xi
        )
        inside ^= crosses
        j = i
    return inside


# --------------------------------------------------------------------- detection
def detect(image_path):
    """468 face landmarks + 10 iris points, normalized, with real depth in z."""
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    opts = vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(MODEL)),
        num_faces=1,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )
    with vision.FaceLandmarker.create_from_options(opts) as lm:
        img = mp.Image.create_from_file(str(image_path))
        res = lm.detect(img)
    if not res.face_landmarks:
        raise RuntimeError(f"no face detected in {image_path}")
    pts = np.array([[p.x, p.y, p.z] for p in res.face_landmarks[0]], dtype=np.float64)
    return pts, img.width, img.height
