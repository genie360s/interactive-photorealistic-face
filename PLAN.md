# interactive_face_web — build plan

Goal: photoreal monochrome African face, follows cursor, keyboard-driven blendshape
expressions. Vue 3 + three.js (WebGPU w/ WebGL2 fallback) + p5 FX layer.
Must verify "pixel perfect" against `input/character_references/*.png` numerically.

## Approach decision (2026-09-18)
Photo-based 2.5D morph rig, NOT a sculpted 3D head.
Reason: the deliverable's look IS the reference photograph. A headless-sculpted CG
head cannot be photoreal and cannot be verified pixel-perfect against the refs.
Using the photo as texture on a MediaPipe-canonical face mesh gives photoreal skin
for free, tiny payload, and a measurable rest-state identity with the source image.

Honest limits of this approach (stated, not hidden):
- head rotation is parallax-limited (small angles), not a full 3D turntable
- mouth interior (teeth/tongue) does not exist in the photo -> synthesized layer
- eyelid skin for blink does not exist in the photo -> stretched-lid technique

## Verification meaning
- VERIFIABLE: neutral rest state == source photo, per-pixel, measured via Playwright
  screenshot diff. Report real numbers.
- NOT VERIFIABLE: expression poses (no ground-truth photo of this person smiling).
  Will be reviewed visually, never claimed as pixel-perfect.

## Phases
- [ ] P1 extract mesh + landmarks from both refs (tools/extract_face.py)
- [ ] P2 build blendshape targets (tools/build_rig.py)
- [ ] P3 Vue + three.js runtime (morph blending, cursor follow, eyes, WebGPU)
- [ ] P4 p5 monochrome FX overlay (lazy-loaded)
- [ ] P5 Playwright pixel-diff verification (tools/verify_pixel_perfect.py)

## Pinned versions (checked 2026-09-18 via npm view)
three 0.186.0 | vue 3.5.43 | vite 8.3.0 | p5 2.3.3 | mediapipe 1.0.1 (py3.11)

## Revision (2026-09-18, mid-build): real expression photographs appeared
`input/character_references/male_characters/` now holds 8 genuine expression
shots of the same man (smile, smile_with_teeth, laugh, cry, fear, kiss, wink,
boo_tongue_out), all 1312x1199 like the neutral.

This changes the rig from authored to MEASURED, and changes what can be claimed:
- morph targets become (aligned expression landmarks - neutral landmarks): the
  real deformation of this person's face, not my FACS guesses
- each expression cross-fades to its OWN photograph, so at full weight the render
  IS that photograph — expressions become pixel-verifiable, which I previously
  stated they could not be
- synthesized teeth/tongue/cavity are no longer needed for those expressions;
  the real ones are in the photos

Vertex order must stay identical across expressions (this is exactly the trap the
referenced OpenProcessing "Facial Rig" sketch calls out). Guaranteed by replaying
the SAME triangulation + subdivision recipe with landmarks substituted.

Alignment: the shots differ slightly in head position/scale. Fitted with a robust
(IRLS) 2D similarity over all landmarks, so points that genuinely move under the
expression get down-weighted instead of dragging the fit. The inverse of that
similarity recovers each expression's UVs at runtime from 4 numbers, so no
per-expression UV array is shipped.

Still procedural (no photo): blink, sad, amazed, blush.

## Revision 2 (2026-09-18): 3D rebuild, on user instruction
User reviewed the 2.5D build and rejected it: "the current is not a 3d model,
it's manipulate image ... plus there seems to be layerings that show a skin
within a skin". Both points are correct. The layering artefact was the eye
patches / mouth underlay / teeth plates compositing over the face plane.

New architecture — one closed 3D mesh, no overlays:
  geometry   silhouette cut from the photo alpha; depth measured from MediaPipe's
             3D landmarks in the face, from the SIDE-VIEW silhouette for the
             skull, relaxed so the two join without a seam
  interior   eyeballs, mouth bag, teeth, tongue as real geometry inside the head
  rig        root -> chest -> neck -> head -> jaw, plus eye.L/eye.R
  shapes     8 measured blendshapes per character, from the expression photos
  export     GLB via Blender (not gltf-builder: that is a low-level JS writer
             with no clear morph-target or skin support; Blender's exporter
             handles shape keys and skinning and is what these need)

Head-depth measurement notes (profile_depth.py): registering front-to-side by
the neck took three attempts — "steepest width increase" lands inside an afro,
"narrowest below widest" lands at the cropped image bottom. Bounding the search
to 30-65% of subject height is what holds for both heads. Final depth/width:
male 0.99, female 0.93; clamped into an anthropometric range when used.

STATUS: geometry + rig + GLB export working and verified by turntable render.
Surface quality still needs a pass (relaxation helps but the face is lumpy where
MediaPipe's z is noisy). Multi-view texture bake and the browser runtime are the
remaining work.

## Final state (2026-09-18)
Reaction set cut to four on request: neutral, smile, blink, wink. The p5 overlay
was removed with them — every effect it drew (tears, blush, boo ring, kiss ring)
belonged to a reaction that is no longer offered, and shipping a button that
throws would be worse than shipping nothing.

Dead 2.5D code removed: src/face/{stage,rig,expressions}.js, src/fx/,
tools/{build_layers,verify_mesh,verify_pixel_perfect,capture_expressions},
public/assets/face/. extract_face.py trimmed to tools/face_landmarks.py.

VERIFIED: neutral head-on vs the source photograph, subject pixels —
male mean 1.33/255 (0.18% off by >16), female mean 1.45/255 (0.10%). Gated in
tools/verify_model.mjs. Expression poses measured and reported but NOT gated,
because the skin texture cannot follow the deformation.

Open: multi-view texture bake using the side/back references (would fix the
male's ear lobe on turn and let the yaw budget rise well above +/-13deg).

## Multi-view texture bake (2026-09-18, on request)
DONE. register_views.py fits each side shot by the landmarks it shares with the
neutral (3D similarity: male -39.9deg 3/4, female +56.7deg profile, residuals
~0.008) and each back shot on the silhouette. build_head.py unwraps the bust
cylindrically (Smart UV shattered it into thousands of slivers), projects all
three photographs by normal, and bakes with Cycles EMIT. The runtime blends the
atlas in ONLY where the rest-pose normal turns away, so front-facing pixels still
come from the untouched frontal projection.

Bugs found and fixed while doing it, each one measured not guessed:
- eyeball cap sized from the eye's HEIGHT, but the opening is 3x wider -> saw
  through the corners. Also: the palpebral opening is wider than any eyeball, so
  the carve itself now shrinks to 0.80 and the lids cover the corners.
- the atlas blend was attached to every textured material, but only the bust has
  uv1 -> the eyeballs sampled an undefined attribute and went black.
- the silhouette fade was gated on "has an atlas", and the SHELL material never
  got one, so the fade silently skipped the half the artefact lived in.
- ROOT CAUSE of the ear lobe: the thin-plate spline is unbounded outside the
  landmark hull, and it was blended out to d=1.9. Past the face it ran away to
  z=-0.33 at full width — a wide skirt hiding behind the head that swung out on
  any turn. Clamping the extrapolation to the landmarks' own z range collapsed
  the depth range from -0.329 to -0.180 and removed the lobe.

VERIFIED after: male 1.42/255, female 1.75/255, gate PASS.
