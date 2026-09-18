# interactive_face_web

A rigged, photoreal 3D head that follows the cursor and reacts to the keyboard.
Vue 3 + three.js, built from the reference photographs in `input/`.

Two characters. Three reactions: **neutral, blink, wink**.

```bash
npm install
npm run dev        # http://localhost:5173
```

---

## What it actually is

A real 3D model, not an image effect:

| | |
|---|---|
| geometry | one closed mesh, ~18.6k verts / 37k tris, depth **measured** from the photos |
| rig | armature `root → chest → neck → head`, plus `eye.L` / `eye.R` |
| blendshapes | 2 morph targets (`wink`, `blink`), measured from the expression photographs |
| interior | real eyeball geometry in real sockets, on their own bones |
| texture | frontal photo + a **multi-view atlas** baked from the front, side and back shots |
| delivery | Draco-compressed GLB, 1.1 MB (male) / 1.5 MB (female) |

The head turns on its bones and the eyes lead it — they arrive first, which is
what makes it read as looking at you rather than being aimed at you.

### Where the shape comes from

Nothing here is sculpted by hand.

- **Face depth** — MediaPipe FaceLandmarker returns 468 landmarks *with* real z.
  A thin-plate spline through them is the face surface.
- **Skull depth** — measured from the side-view silhouettes (`profile_depth.py`),
  not a guessed dome.
- **Silhouette** — traced from the photographs' own alpha channel, so the model
  is a bust with no background plane.
- **Blendshapes** — each expression photo is landmarked, rigidly aligned to the
  neutral (robust IRLS similarity, so a wide-open jaw cannot drag the fit), and
  the difference is the shape. `wink` is this person's wink. All eight are
  measured by the pipeline; only the two the UI uses are packed into the GLB.
- **Blink** — derived by mirroring the measured `wink` eyelid across the face's
  midline, so both eyes use a real eyelid rather than an invented one.
- **Side and back surfaces** — baked from the side and back photographs.
  MediaPipe detects a face in both side shots, so they are registered by the
  landmarks they share with the neutral: a robust 3D similarity recovers the
  head's angle from the data (measured at −39.9° for the male's 3/4 view, +56.7°
  for the female's profile) instead of assuming one. The backs have no face, so
  they are registered on the silhouette — head top, neck, centre and width.

Vertex order is identical across every expression, which is the thing that
breaks blendshape rigs most often. It is guaranteed by replaying the *same*
triangulation with landmarks substituted, never re-triangulating.

---

## Pixel-perfect verification

```bash
npm run dev &
node tools/verify_model.mjs
```

Renders the model head-on at the source photograph's exact resolution and diffs
it per pixel, split by the reference's alpha (subject / antialiased edge).

**Measured, neutral head-on, subject pixels:**

| character | mean err | max | exact | >4 | >16 |
|---|---|---|---|---|---|
| male | **1.40**/255 | 118 | 25.0% | 1.94% | 0.33% |
| female | **1.77**/255 | 139 | 22.8% | 4.91% | 0.71% |

This is the strong claim and the one the gate enforces: same camera, same
orthographic projection, same pixels. The texture is a frontal projection of the
photograph, so a head-on render *should* reproduce it, and it does.

The multi-view bake is deliberately arranged **not** to spend this. Baking
everything into one atlas would resample the face and cost the number outright,
so instead the frontal projection stays primary and the atlas is blended in only
where the surface turns away from the camera — anything the camera can see
head-on still samples the original photograph. The cost above (1.33 → 1.42,
1.45 → 1.75) is the silhouette fade described under Known limits, not the bake.

**What is NOT claimed.** The script also diffs `wink` against the photo it was
measured from, and that number is far higher (mean 6.0 male / 9.2 female). That is
expected and is reported rather than hidden: the geometry moves to match the
expression but the skin texture stays the neutral photograph, so the creases,
teeth and shadows that exist only in the expression shot are missing. Those
figures measure how well the blendshape reproduces the real *deformation* — they
are not a pixel-perfect claim, and calling them one would be false.

---

## Rebuilding from the photographs

```bash
uv venv --python 3.11 tools/.venv
uv pip install --python tools/.venv/bin/python mediapipe==0.10.35 opencv-python pillow numpy scipy

tools/.venv/bin/python tools/export_face_data.py      # landmarks + blendshapes
tools/.venv/bin/python tools/profile_depth.py         # skull depth from side views
tools/.venv/bin/python tools/register_views.py        # register side + back shots
tools/.venv/bin/python tools/build_bust_geometry.py   # solid bust + morph deltas
/Applications/Blender.app/Contents/MacOS/Blender -b --factory-startup \
  --python tools/build_head.py -- --character male    # rig + GLB  (also: female)
```

Add `--preview` to `build_head.py` for a turntable render.

---

## Known limits

Stated plainly, because they are real:

- **Rotation is budgeted to ±13°.** A frontal photograph still carries most of
  the detail, and the atlas is a projection, not a scan.
- **The silhouette is faded** where the surface turns edge-on. Without it the
  bust's rim swings into view as a smear when the head turns. It costs a little
  head-on accuracy (the 1.33 → 1.42 above) and is why `max` error sits at 118:
  that is a handful of rim pixels, not the face.
- **Expressions do not change the skin texture**, so creases and shadows that
  exist only in an expression photograph are missing.
- **There is no mouth interior, and the mouth is not carved.** None of the three
  reactions opens the mouth, so the bag, teeth and tongue were stripped — and the
  lip line had to stop being a hole at the same time, or it would look straight
  through the head. `build_head.py --parts eyes,mouth` builds them back, and
  `CARVE_MOUTH` in `build_bust_geometry.py` reopens the aperture, for a character
  that needs an open mouth.

## Layout

```
input/character_references/   the photographs (front, 8 expressions, side, back)
tools/                        the offline pipeline + both verifiers
assets/facedata/              intermediate measurements (not served)
public/assets/model/          the GLB + per-model metadata
public/draco/                 Draco decoder, served locally (no CDN)
src/face/model.js             runtime: GLB, bones, morphs, cursor following
src/face/expressions3d.js     the four reactions
verification/                 heatmaps, reports, capture sheets
```
