"""
P3 (3D rebuild) — Assemble the rigged head in Blender and export GLB.

Runs inside Blender (numpy only, no scipy — the heavy maths is done first by
build_bust_geometry.py).

What gets built:
  bust      one closed mesh, measured depth, frontal photo projection
  eyeballs  real spheres in real sockets, photo-projected, on their own bones
  mouth     a bag, two tooth rows and a tongue — geometry inside the head, so an
            open mouth reveals something that is actually there
  shapes    eight measured blendshapes, straight off the expression photographs
  armature  root -> chest -> neck -> head -> jaw, plus eye.L / eye.R

Run:
  Blender -b --factory-startup --python tools/build_head.py -- --character male
"""

import json
import pathlib
import sys

import bpy
import numpy as np
from mathutils import Matrix, Vector

ARGV = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []


def arg(flag, default=None):
    return ARGV[ARGV.index(flag) + 1] if flag in ARGV else default


ROOT = pathlib.Path(arg("--root", str(pathlib.Path(__file__).resolve().parent.parent)))
DATA = ROOT / "assets" / "facedata"
OUT = ROOT / "public" / "assets" / "model"

# Rotation the runtime is allowed to use. A frontal photograph holds no
# information about the sides of the head, so the illusion has a budget; past
# roughly this the projection visibly smears.
MAX_YAW_DEG = 13.0
EYE_APERTURE = 0.80   # must match build_bust_geometry.EYE_APERTURE


def to_blender(v):
    """(x right, y up, z toward viewer)  ->  Blender (x right, y forward, z up).

    The measurement space treats y as vertical because that is how an image is
    indexed. Blender's vertical axis is Z, so handing it the data unchanged lays
    the bust on its back — and export_yup then rotates that again, which is how
    the first export came out looking at the ceiling. One conversion, applied to
    every vertex and every bone, keeps the whole rig consistent.
    """
    v = np.asarray(v, dtype=np.float64)
    if v.ndim == 1:
        return np.array([v[0], -v[2], v[1]])
    return np.column_stack([v[:, 0], -v[:, 2], v[:, 1]])


def smoothstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0 + 1e-12), 0, 1)
    return t * t * (3 - 2 * t)


# ------------------------------------------------------------------ scene setup
def reset():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    sc = bpy.context.scene
    sc.unit_settings.system = "NONE"
    return sc


def new_mesh(name, verts, tris, uv=None, mat_idx=None):
    verts = to_blender(verts)
    me = bpy.data.meshes.new(name)
    me.from_pydata([tuple(map(float, v)) for v in verts], [], [tuple(map(int, t)) for t in tris])
    me.validate(verbose=False)
    me.update()
    if uv is not None:
        layer = me.uv_layers.new(name="UVMap")
        loops = np.empty(len(me.loops), dtype=np.int32)
        me.loops.foreach_get("vertex_index", loops)
        layer.data.foreach_set("uv", uv[loops].ravel().astype(np.float32))
    if mat_idx is not None:
        me.polygons.foreach_set("material_index", mat_idx.astype(np.int32))
    for p in me.polygons:
        p.use_smooth = True
    ob = bpy.data.objects.new(name, me)
    bpy.context.collection.objects.link(ob)
    return ob


def photo_material(name, image_path):
    """The portrait already contains its own lighting. Re-lighting it would
    double the shading and wreck the likeness, so the surface emits the photo
    directly and the runtime uses an unlit material for the same reason."""
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    tex = nt.nodes.new("ShaderNodeTexImage")
    tex.image = bpy.data.images.load(str(image_path))
    tex.image.colorspace_settings.name = "sRGB"
    tex.interpolation = "Cubic"
    nt.links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
    nt.links.new(tex.outputs["Alpha"], bsdf.inputs["Alpha"])
    nt.links.new(tex.outputs["Color"], bsdf.inputs["Emission Color"])
    bsdf.inputs["Emission Strength"].default_value = 1.0
    bsdf.inputs["Roughness"].default_value = 1.0
    bsdf.inputs["Metallic"].default_value = 0.0
    nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    return mat


def solid_material(name, value):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    c = (value, value, value, 1)
    bsdf.inputs["Base Color"].default_value = c
    bsdf.inputs["Emission Color"].default_value = c
    bsdf.inputs["Emission Strength"].default_value = 1.0
    bsdf.inputs["Roughness"].default_value = 0.65
    nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    return mat


# -------------------------------------------------------------- mouth interior
def lip_arc(ring, upper, n=18):
    """A SMOOTH arc across the mouth, fitted rather than resampled.

    Taking the ring's upper/lower half and interpolating it directly gives a
    jagged line — the ring has only ten points a side and they are not monotonic
    in x — and lofting a slab along that produced a row of fangs. A low-order
    polynomial through those points is the shape a tooth row actually follows.
    """
    c = ring.mean(axis=0)
    half = ring[ring[:, 1] >= c[1]] if upper else ring[ring[:, 1] < c[1]]
    x0, x1 = ring[:, 0].min(), ring[:, 0].max()
    inset = (x1 - x0) * 0.10          # teeth stop short of the mouth corners
    xs = np.linspace(x0 + inset, x1 - inset, n)
    fy = np.polyfit(half[:, 0], half[:, 1], 2)
    fz = np.polyfit(half[:, 0], half[:, 2], 2)
    return np.column_stack([xs, np.polyval(fy, xs), np.polyval(fz, xs)])


def loft_slab(arc, height, depth, down):
    """Sweep a rectangular cross-section along the arc. Every quad is built from
    consecutive cross-sections, so the surface cannot self-intersect the way a
    hand-indexed vertex soup can."""
    n = len(arc)
    dy = -height if down else height
    # cross-section corners: front-root, front-tip, back-tip, back-root
    rings = []
    for p in arc:
        rings.append([
            [p[0], p[1], p[2]],
            [p[0], p[1] + dy, p[2]],
            [p[0], p[1] + dy, p[2] - depth],
            [p[0], p[1], p[2] - depth],
        ])
    V = np.array(rings, dtype=float).reshape(-1, 3)
    F = []
    for i in range(n - 1):
        a, b = i * 4, (i + 1) * 4
        for k in range(4):
            k2 = (k + 1) % 4
            F += [[a + k, a + k2, b + k2], [a + k, b + k2, b + k]]
    # cap both ends so the row is a closed solid
    for a, flip in ((0, False), ((n - 1) * 4, True)):
        q = [a, a + 1, a + 2, a + 3]
        tri = [[q[0], q[1], q[2]], [q[0], q[2], q[3]]]
        F += [t[::-1] for t in tri] if flip else tri
    return V, np.array(F, dtype=int)


def ellipsoid(c, r, nu=26, nv=16):
    u = np.linspace(0, np.pi * 2, nu)
    t = np.linspace(-np.pi / 2, np.pi / 2, nv)
    U, T = np.meshgrid(u, t)
    V = np.column_stack([
        (np.cos(T) * np.cos(U) * r[0] + c[0]).ravel(),
        (np.sin(T) * r[1] + c[1]).ravel(),
        (np.cos(T) * np.sin(U) * r[2] + c[2]).ravel(),
    ])
    F = []
    for a in range(nv - 1):
        for b in range(nu - 1):
            i0 = a * nu + b
            F += [[i0, i0 + 1, i0 + nu + 1], [i0, i0 + nu + 1, i0 + nu]]
    return V, np.array(F, dtype=int)


def build_mouth(npz, hr, mats):
    """Bag, two tooth rows and a tongue — real objects inside a real head.

    Sized from the MOUTH WIDTH, not from the closed lip ring. Derived from the
    ring, every part came out as flat as the closed lips are: the "bag" was a
    slit swept backwards with no height at all, so an opening mouth revealed
    nothing to be inside of, and the tongue sat in front of the teeth and poked
    out through the face.
    """
    inner = npz["lips_inner"].astype(np.float64)
    c = inner.mean(axis=0)
    W = float(inner[:, 0].max() - inner[:, 0].min())
    parts = []

    # cavity: a closed sack big enough to hold an open mouth. Seen through the
    # aperture we are looking at its inner back wall, which is why it is dark.
    bagc = np.array([c[0], c[1] - W * 0.06, c[2] - W * 0.44])
    Vb, Fb = ellipsoid(bagc, (W * 0.60, W * 0.40, W * 0.50))
    parts.append(("mouth_bag", Vb, Fb, mats["dark"], "head"))

    # teeth: both rows meet at the occlusal plane and grow away from it
    up = lip_arc(inner, upper=True)
    lo = lip_arc(inner, upper=False)
    occl = 0.5 * (up[:, 1] + lo[:, 1])
    up[:, 1] = occl
    lo[:, 1] = occl
    setback = W * 0.10
    up[:, 2] -= setback
    lo[:, 2] -= setback
    h = W * 0.115
    d = W * 0.13
    vu, fu = loft_slab(up, h, d, down=False)
    vl, fl = loft_slab(lo, h * 0.85, d, down=True)
    parts.append(("teeth_upper", vu, fu, mats["teeth"], "head"))
    parts.append(("teeth_lower", vl, fl, mats["teeth"], "jaw"))

    # tongue: on the floor of the cavity and BEHIND the tooth rows, so a closed
    # mouth cannot show it
    # the tip must end BEHIND the tooth rows, so place the centre a full
    # z-radius back from the setback rather than guessing an offset
    rz = W * 0.32
    tc = np.array([c[0], occl.mean() - W * 0.13, c[2] - setback - rz])
    Vt, Ft = ellipsoid(tc, (W * 0.30, W * 0.085, rz))
    parts.append(("tongue", Vt, Ft, mats["tongue"], "jaw"))
    return parts


# -------------------------------------------------------------------- eyeballs
def build_eyeball(centre, iris_r, hr, opening_h):
    """A spherical CAP, set back behind the eyelid — not a full eyeball.

    A full sphere at anatomical scale (radius ~2x the iris) is far wider than the
    eye opening, so its surface runs alongside the eyelid at almost exactly the
    lid's own depth. The two then z-fight, and because the eyeball carries the
    frontal projection of the photograph — lashes and lid crease included — the
    result is the lash line drawn twice: a double eyelid.

    Capping it to a little more than the aperture, and seating it behind the lid,
    means the eyeball can only ever be seen THROUGH the opening.
    """
    R = iris_r * 2.2
    # sit the cap so its pole is just behind the socket plane
    back = np.sqrt(max(R * R - iris_r * iris_r, 1e-9)) + iris_r * 0.12
    c = np.array([centre[0], centre[1], centre[2] - back])

    # The cap must cover the aperture's WIDTH, which is its larger dimension —
    # sizing it from the height left the corners uncovered and showed straight
    # through the eye.
    reach = opening_h * 1.12
    amax = float(np.arcsin(min(reach / R, 0.985)))

    nu, nv = 30, 14
    u = np.linspace(0, np.pi * 2, nu)
    t = np.linspace(0, amax, nv)
    U, T = np.meshgrid(u, t)
    V = np.column_stack([
        (np.sin(T) * np.cos(U) * R + c[0]).ravel(),
        (np.sin(T) * np.sin(U) * R + c[1]).ravel(),
        (np.cos(T) * R + c[2]).ravel(),
    ])
    F = []
    for a in range(nv - 1):
        for b in range(nu - 1):
            i0 = a * nu + b
            F += [[i0, i0 + 1, i0 + nu + 1], [i0, i0 + nu + 1, i0 + nu]]
    return V, np.array(F, dtype=int), c, R


# -------------------------------------------------------------------- armature
def build_armature(name, verts, npz, hr):
    oval = npz["face_oval"].astype(np.float64)
    jaw_y = float(oval[:, 1].min())
    top_y = float(verts[:, 1].max())
    bot_y = float(verts[:, 1].min())
    cx = float(npz["head_center"][0])
    z_mid = float(np.median(verts[:, 2]))
    # the skull pivots about the atlas: behind and below the face, not at its centre
    pivot = np.array([cx, jaw_y + hr * 0.10, z_mid - hr * 0.45])

    arm = bpy.data.armatures.new(f"{name}_rig")
    rig = bpy.data.objects.new(f"{name}_rig", arm)
    bpy.context.collection.objects.link(rig)
    bpy.context.view_layer.objects.active = rig
    bpy.ops.object.mode_set(mode="EDIT")

    def bone(bname, head, tail, parent=None):
        b = arm.edit_bones.new(bname)
        b.head = Vector(tuple(map(float, to_blender(np.asarray(head, float)))))
        b.tail = Vector(tuple(map(float, to_blender(np.asarray(tail, float)))))
        if parent:
            b.parent = arm.edit_bones[parent]
            b.use_connect = False
        return b

    bone("root", (cx, bot_y, z_mid), (cx, bot_y + hr * 0.2, z_mid))
    bone("chest", (cx, bot_y + hr * 0.2, z_mid), (cx, jaw_y - hr * 0.95, z_mid), "root")
    bone("neck", (cx, jaw_y - hr * 0.95, z_mid), tuple(pivot), "chest")
    bone("head", tuple(pivot), (cx, top_y, z_mid - hr * 0.2), "neck")
    # no jaw bone: its only children were the lower teeth and the tongue, and
    # with the mouth interior stripped it would drive nothing
    for side, key in (("L", "iris_l"), ("R", "iris_r")):
        e = npz[key].astype(np.float64)
        R = float(e[3]) * 1.75
        back = np.sqrt(max(R * R - e[3] ** 2, 1e-9)) + e[3] * 0.22
        c = np.array([e[0], e[1], e[2] - back])
        bone(f"eye.{side}", tuple(c), (c[0], c[1], c[2] + R * 1.6), "head")

    bpy.ops.object.mode_set(mode="OBJECT")
    return rig, pivot, jaw_y


def skin_weights(ob, rig, verts, jaw_y, hr):
    """Head above the jaw, chest at the shoulders, neck blending between.

    Assigned from height rather than by heat-diffusion: this mesh is a bust with
    a deliberately simple bone chain, and a measured gradient is both predictable
    and immune to the silent failures automatic weighting has on shells.
    """
    y = verts[:, 1]
    w_head = 1.0 - smoothstep(jaw_y - hr * 0.55, jaw_y + hr * 0.08, y)
    w_head = 1.0 - w_head
    w_chest = smoothstep(jaw_y - hr * 0.35, jaw_y - hr * 1.15, y)
    w_neck = np.clip(1.0 - w_head - w_chest, 0, 1)
    total = np.maximum(w_head + w_chest + w_neck, 1e-9)

    groups = {n: ob.vertex_groups.new(name=n) for n in ("head", "neck", "chest")}
    for gname, w in (("head", w_head), ("neck", w_neck), ("chest", w_chest)):
        w = w / total
        g = groups[gname]
        for i in np.where(w > 1e-4)[0]:
            g.add([int(i)], float(w[i]), "REPLACE")

    mod = ob.modifiers.new("Armature", "ARMATURE")
    mod.object = rig
    ob.parent = rig
    return {"head": w_head, "neck": w_neck, "chest": w_chest}


def bind_rigid(ob, rig, bone_name):
    g = ob.vertex_groups.new(name=bone_name)
    g.add(list(range(len(ob.data.vertices))), 1.0, "REPLACE")
    mod = ob.modifiers.new("Armature", "ARMATURE")
    mod.object = rig
    ob.parent = rig


# ------------------------------------------------------------ multi-view bake
def _img(nt, path, label):
    n = nt.nodes.new("ShaderNodeTexImage")
    n.image = bpy.data.images.load(str(path))
    n.image.colorspace_settings.name = "sRGB"
    n.extension = "EXTEND"
    n.label = label
    return n


def _xz_to_uv(nt, src, sx, sy, ox, oy):
    """Take a 3D point, keep X and Z, and map them affinely into UV.

    Every one of these projections is orthographic in the image plane, so the
    whole thing is: drop the depth axis, scale, offset.
    """
    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    nt.links.new(src, sep.inputs["Vector"])
    com = nt.nodes.new("ShaderNodeCombineXYZ")
    nt.links.new(sep.outputs["X"], com.inputs["X"])
    nt.links.new(sep.outputs["Z"], com.inputs["Y"])
    m = nt.nodes.new("ShaderNodeMapping")
    m.vector_type = "POINT"
    nt.links.new(com.outputs["Vector"], m.inputs["Vector"])
    m.inputs["Scale"].default_value = (sx, sy, 1.0)
    m.inputs["Location"].default_value = (ox, oy, 0.0)
    return m.outputs["Vector"]


def build_bake_material(name, views, aspect):
    """Front, side and back photographs projected onto the model and blended by
    surface normal.

    The side shot is placed by the similarity that register_views.py fitted from
    the landmarks the two photographs share — which is why a 3/4 view works as
    well as a true profile: the fit recovers the angle instead of assuming one.
    """
    mat = bpy.data.materials.new(f"{name}_bakesrc")
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()

    geo = nt.nodes.new("ShaderNodeNewGeometry")
    pos = nt.nodes.new("ShaderNodeNewGeometry")  # Position output, world space

    # ---- front: u = x/aspect + 0.5, v = z + 0.5 -----------------------------
    front_uv = _xz_to_uv(nt, pos.outputs["Position"], 1.0 / aspect, 1.0, 0.5, 0.5)
    front = _img(nt, ROOT / views["front"]["image"], "front")
    nt.links.new(front_uv, front.inputs["Vector"])

    # ---- back: mirrored ortho, fitted on head top / neck / centre -----------
    b = views["back"]
    back_uv = _xz_to_uv(nt, pos.outputs["Position"], b["su"], -b["sv"], b["ou"], 1.0 - b["ov"])
    back = _img(nt, ROOT / b["image"], "back")
    nt.links.new(back_uv, back.inputs["Vector"])

    # ---- side: the fitted similarity, folded with the axis swap -------------
    sv = views["side"]
    S = float(sv["scale"])
    R = np.array(sv["R"], dtype=float)
    t = np.array(sv["t"], dtype=float)
    # Blender (X, Y, Z) -> measurement world (x, y_up, z_depth) = (X, Z, -Y)
    A = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=float)
    M = R @ A
    eul = Matrix(M.tolist()).to_euler()

    mp = nt.nodes.new("ShaderNodeMapping")
    mp.vector_type = "POINT"
    nt.links.new(pos.outputs["Position"], mp.inputs["Vector"])
    mp.inputs["Scale"].default_value = (S, S, S)
    mp.inputs["Rotation"].default_value = (eul.x, eul.y, eul.z)
    mp.inputs["Location"].default_value = tuple(t)
    # the mapped point is in the side shot's own world units: u = x/aspect+0.5,
    # v = y+0.5 — and here y IS the vertical axis, so keep X and Y
    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    nt.links.new(mp.outputs["Vector"], sep.inputs["Vector"])
    com = nt.nodes.new("ShaderNodeCombineXYZ")
    nt.links.new(sep.outputs["X"], com.inputs["X"])
    nt.links.new(sep.outputs["Y"], com.inputs["Y"])
    m2 = nt.nodes.new("ShaderNodeMapping")
    m2.vector_type = "POINT"
    nt.links.new(com.outputs["Vector"], m2.inputs["Vector"])
    m2.inputs["Scale"].default_value = (1.0 / sv["aspect"], 1.0, 1.0)
    m2.inputs["Location"].default_value = (0.5, 0.5, 0.0)
    side = _img(nt, ROOT / sv["image"], "side")
    nt.links.new(m2.outputs["Vector"], side.inputs["Vector"])

    # ---- blend by normal ----------------------------------------------------
    # Blender Y is depth pointing AWAY from the front camera, so a surface facing
    # the camera has normal.Y < 0.
    nsep = nt.nodes.new("ShaderNodeSeparateXYZ")
    nt.links.new(geo.outputs["Normal"], nsep.inputs["Vector"])

    def maprange(src, lo, hi):
        mr = nt.nodes.new("ShaderNodeMapRange")
        mr.clamp = True
        nt.links.new(src, mr.inputs["Value"])
        mr.inputs["From Min"].default_value = lo
        mr.inputs["From Max"].default_value = hi
        return mr.outputs["Result"]

    w_front = maprange(nsep.outputs["Y"], 0.25, -0.25)   # 1 when facing front
    w_back = maprange(nsep.outputs["Y"], -0.05, 0.45)    # 1 when facing back

    mix1 = nt.nodes.new("ShaderNodeMixRGB")
    mix1.blend_type = "MIX"
    nt.links.new(w_back, mix1.inputs["Fac"])
    nt.links.new(side.outputs["Color"], mix1.inputs[1])
    nt.links.new(back.outputs["Color"], mix1.inputs[2])

    mix2 = nt.nodes.new("ShaderNodeMixRGB")
    mix2.blend_type = "MIX"
    nt.links.new(w_front, mix2.inputs["Fac"])
    nt.links.new(mix1.outputs["Color"], mix2.inputs[1])
    nt.links.new(front.outputs["Color"], mix2.inputs[2])

    emit = nt.nodes.new("ShaderNodeEmission")
    nt.links.new(mix2.outputs["Color"], emit.inputs["Color"])
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    nt.links.new(emit.outputs["Emission"], out.inputs["Surface"])
    return mat


def bake_atlas(name, bust, views, aspect, res=2048):
    """Unwrap to an atlas and bake the three projections into it.

    Only the atlas is baked — the front photograph is still sampled directly
    through the original frontal UVs at runtime, so anything facing the camera
    is untouched by this resampling and the head-on match is preserved.
    """
    scene = bpy.context.scene
    bpy.context.view_layer.objects.active = bust
    bust.select_set(True)

    # CYLINDRICAL unwrap, computed directly rather than via Smart UV Project.
    # Smart UV shatters an organic 37k-tri bust into thousands of slivers: most
    # of the atlas ends up as margin and every island edge is a potential seam.
    # A head is naturally cylindrical, so one wrap around the vertical axis gives
    # continuous texel density and exactly one seam, which is put at the back.
    me = bust.data
    co = np.empty(len(me.vertices) * 3, dtype=np.float32)
    me.vertices.foreach_get("co", co)
    co = co.reshape(-1, 3)
    cx, cy = float(np.median(co[:, 0])), float(np.median(co[:, 1]))
    z0, z1 = float(co[:, 2].min()), float(co[:, 2].max())

    ang = np.arctan2(co[:, 0] - cx, -(co[:, 1] - cy))   # 0 at the front
    u_v = ang / (2 * np.pi) + 0.5
    v_v = (co[:, 2] - z0) / max(z1 - z0, 1e-9)

    uv2 = me.uv_layers.new(name="UVAtlas")
    me.uv_layers.active = uv2
    loop_vi = np.empty(len(me.loops), dtype=np.int32)
    me.loops.foreach_get("vertex_index", loop_vi)
    uvs = np.stack([u_v[loop_vi], v_v[loop_vi]], axis=1)

    # Faces crossing the seam would otherwise span the whole atlas. UVs are
    # per-loop, so the wrapped corners can simply be pushed past 1.0 and the
    # texture left to repeat.
    tri = loop_vi.reshape(-1, 3)
    uu = u_v[tri]
    wrap = (uu.max(axis=1) - uu.min(axis=1)) > 0.5
    fix = np.zeros(len(loop_vi), dtype=bool).reshape(-1, 3)
    fix[wrap] = uu[wrap] < 0.5
    uvs[fix.ravel(), 0] += 1.0
    uv2.data.foreach_set("uv", uvs.ravel().astype(np.float32))

    img = bpy.data.images.new(f"{name}_atlas", res, res, alpha=True)
    src = build_bake_material(name, views, aspect)
    tgt = src.node_tree.nodes.new("ShaderNodeTexImage")
    tgt.image = img
    src.node_tree.nodes.active = tgt

    saved = [m.material for m in bust.material_slots]
    for slot in bust.material_slots:
        slot.material = src

    scene.render.engine = "CYCLES"
    scene.cycles.samples = 1
    scene.cycles.use_denoising = False
    scene.render.bake.use_pass_direct = False
    scene.render.bake.use_pass_indirect = False
    scene.render.bake.margin = 8
    bpy.ops.object.bake(type="EMIT", use_clear=True)

    out = DATA / f"{name}_atlas.png"   # intermediate; the GLB carries the real copy
    img.filepath_raw = str(out)
    img.file_format = "PNG"
    img.save()

    for slot, m in zip(bust.material_slots, saved):
        slot.material = m
    return out, uv2.name


# ----------------------------------------------------------------------- render
def setup_render(scene, res=(900, 900)):
    scene.render.resolution_x, scene.render.resolution_y = res
    scene.render.film_transparent = True
    scene.render.image_settings.file_format = "PNG"
    for engine in ("BLENDER_EEVEE_NEXT", "BLENDER_EEVEE", "BLENDER_WORKBENCH"):
        try:
            scene.render.engine = engine
            break
        except TypeError:
            continue
    try:
        # AgX would bleach the photograph; Standard keeps the tones as shot
        scene.view_settings.view_transform = "Standard"
    except TypeError:
        pass
    cam_d = bpy.data.cameras.new("Cam")
    cam_d.type = "ORTHO"
    cam_d.ortho_scale = 1.0
    cam = bpy.data.objects.new("Cam", cam_d)
    bpy.context.collection.objects.link(cam)
    # Blender is Z-up: the front view looks along +Y from -Y
    cam.location = (0, -3, 0)
    cam.rotation_euler = (np.pi / 2, 0, 0)
    scene.camera = cam
    return cam


def render_turntable(scene, rig, out_dir, angles=(0, -26)):
    """The real test that this is geometry and not a picture: a picture looks
    identical from every angle."""
    out_dir.mkdir(parents=True, exist_ok=True)
    pb = rig.pose.bones
    for a in angles:
        pb["head"].rotation_mode = "XYZ"
        pb["head"].rotation_euler = (0, 0, np.radians(a))   # yaw about Z now
        bpy.context.view_layer.update()
        scene.render.filepath = str(out_dir / f"head_{a:+03d}.png")
        bpy.ops.render.render(write_still=True)
    pb["head"].rotation_euler = (0, 0, 0)


# ------------------------------------------------------------------------- main
def build(name, preview=False):
    scene = reset()
    npz = np.load(DATA / f"{name}.bust.npz", allow_pickle=True)
    fd = json.loads((DATA / f"{name}.json").read_text())

    verts = npz["verts"].astype(np.float64)
    tris = npz["tris"]
    uv = npz["uv"]
    mat_idx = npz["mat"]
    hr = float(npz["head_radius"])

    skin = photo_material(f"{name}_skin", ROOT / fd["neutralImage"])
    mats = {
        # The shell carries the photo's ALPHA (so it vanishes behind the hair
        # fringe, where a solid shell measured rgb 7.6 against the photo's 30)
        # but the runtime tints it dark, so the band sweeping back from the
        # silhouette reads as shadow instead of a pale paddle stuck to the ear.
        "shell": photo_material(f"{name}_shell", ROOT / fd["neutralImage"]),
        "dark": solid_material(f"{name}_bag", 0.02),
        # enamel, not paper: the photos top out around 0.65 luminance and a
        # brighter row reads as a sticker glued over the mouth
        "teeth": solid_material(f"{name}_teeth", 0.30),
        "tongue": solid_material(f"{name}_tongue", 0.16),
    }

    bust = new_mesh(f"{name}_head", verts, tris, uv, mat_idx)
    bust.data.materials.append(skin)
    bust.data.materials.append(mats["shell"])

    # --- measured blendshapes -------------------------------------------------
    bust.shape_key_add(name="Basis", from_mix=False)
    shape_names = [str(s) for s in npz["shape_names"]]
    for sname in shape_names:
        key = bust.shape_key_add(name=sname, from_mix=False)
        co = to_blender(verts + npz[f"shape_{sname}"]).astype(np.float32).ravel()
        key.data.foreach_set("co", co)

    rig, pivot, jaw_y = build_armature(name, verts, npz, hr)
    skin_weights(bust, rig, verts, jaw_y, hr)

    # the mouth interior is stripped from the shipped model; pass --parts
    # eyes,mouth to build it back for a character that needs an open mouth
    parts_on = arg("--parts", "eyes").split(",")

    # --- eyes -----------------------------------------------------------------
    for side, key, eyegrp in (
        (("L", "iris_l", "leftEye"), ("R", "iris_r", "rightEye"))
        if "eyes" in parts_on else ()
    ):
        e = npz[key].astype(np.float64)
        ring = np.array(fd["landmarks"])[fd["groups"][eyegrp]]
        rc = ring.mean(axis=0)
        # the carve shrinks the ring, so the cap is sized against the SHRUNK
        # aperture's half-width (its largest radius), not the raw landmarks
        opening_h = float(
            np.max(np.linalg.norm((ring[:, :2] - rc[:2]) * EYE_APERTURE, axis=1))
        )
        V, F, c, R = build_eyeball(e[:3], float(e[3]), hr, opening_h)
        euv = np.stack([V[:, 0] / fd["image"]["aspect"] + 0.5, V[:, 1] + 0.5], axis=1)
        ob = new_mesh(f"{name}_eye_{side}", V, F, euv)
        ob.data.materials.append(skin)
        bind_rigid(ob, rig, f"eye.{side}")

    # --- mouth ----------------------------------------------------------------
    for pname, V, F, mat, bone_name in (build_mouth(npz, hr, mats) if "mouth" in parts_on else []):
        ob = new_mesh(f"{name}_{pname}", V, F)
        ob.data.materials.append(mat)
        bind_rigid(ob, rig, bone_name)

    # ---- multi-view bake -----------------------------------------------------
    atlas_path = None
    views_file = DATA / f"{name}.views.json"
    if "--no-bake" not in ARGV and views_file.exists():
        views = json.loads(views_file.read_text())
        atlas_path, atlas_uv = bake_atlas(name, bust, views, fd["image"]["aspect"])
        # Carried as the EMISSIVE texture on the atlas UV set purely so the
        # exporter keeps both UV layers and both images in the GLB; the runtime
        # reads them back and does its own blend. glTF has no slot for "second
        # albedo with its own UVs", and inventing an extension for it would be
        # worse than reusing one that already round-trips.
        nt = skin.node_tree
        tex = nt.nodes.new("ShaderNodeTexImage")
        tex.image = bpy.data.images.load(str(atlas_path))
        tex.image.colorspace_settings.name = "sRGB"
        uvn = nt.nodes.new("ShaderNodeUVMap")
        uvn.uv_map = atlas_uv
        nt.links.new(uvn.outputs["UV"], tex.inputs["Vector"])
        bsdf = next(n for n in nt.nodes if n.type == "BSDF_PRINCIPLED")
        nt.links.new(tex.outputs["Color"], bsdf.inputs["Emission Color"])
        bsdf.inputs["Emission Strength"].default_value = 1.0

    cam = setup_render(scene)
    if preview:
        tag = arg("--tag", "")
        render_turntable(scene, rig, ROOT / "verification" / "model3d" / (name + tag))

    # --- export ---------------------------------------------------------------
    OUT.mkdir(parents=True, exist_ok=True)
    glb = OUT / f"{name}.glb"
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.export_scene.gltf(
        filepath=str(glb),
        export_format="GLB",
        export_apply=False,           # modifiers must not be applied: shape keys
        export_morph=True,
        export_skins=True,
        export_yup=True,
        export_cameras=False,
        export_lights=False,
        # the photo is the whole payload; PNG in a GLB is several MB for nothing
        export_image_format="WEBP",
        export_image_quality=92,
        # unlit at runtime, so per-morph normals are data nobody reads — and they
        # cost as much as the positions they accompany
        export_morph_normal=False,
        export_morph_tangent=False,
        export_draco_mesh_compression_enable=True,
        export_draco_mesh_compression_level=6,
        export_draco_position_quantization=14,
        export_draco_texcoord_quantization=12,
    )

    aper = npz["apertures"].astype(float)
    a0 = float(npz["aperture_neutral"])
    meta = {
        "name": name,
        "shapes": shape_names,
        "maxYawDeg": MAX_YAW_DEG,
        "headRadius": hr,
        "aspect": fd["image"]["aspect"],
        # measured inner-lip opening per expression, normalised — the runtime
        # drives the jaw bone from this so the lower teeth and tongue follow the
        # blendshape instead of being keyframed by eye
        # normalised against the widest measured opening, so the jaw bone's
        # range is 0..1 by construction rather than whatever the arithmetic
        # happened to produce (the first version peaked at 6.5)
        "hasAtlas": atlas_path is not None,
        "jawOpen": {
            s: round(float(np.clip((a - a0) / max(aper.max() - a0, 1e-6), 0, 1)), 3)
            for s, a in zip(shape_names, aper)
        },
    }
    (OUT / f"{name}.model.json").write_text(json.dumps(meta, indent=1))

    print(
        f"[build_head] {name}: {len(verts)} verts, {len(tris)} tris, "
        f"{len(shape_names)} shapes -> {glb.name} {glb.stat().st_size / 1024:.0f} KB"
    )
    print("   jawOpen " + "  ".join(f"{k}:{v:.2f}" for k, v in meta["jawOpen"].items()))


if __name__ == "__main__":
    build(arg("--character", "male"), preview="--preview" in ARGV)
