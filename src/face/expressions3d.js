/**
 * Reactions, over the MEASURED blendshapes.
 *
 * Scoped to three on request: neutral, blink, wink. Wink is a photograph of this
 * person actually making the expression, so it is a measurement rather than an
 * approximation. Blink is derived by mirroring that same measured eyelid across
 * the face's midline, so both eyes use a real eyelid rather than an invented one.
 *
 * The other measured shapes (smile, smileTeeth, laugh, cry, fear, kiss, tongue)
 * are still in the GLB and still driveable by name; they are simply not offered
 * in the UI.
 */

export const EXPRESSIONS = {
  neutral: { label: 'Neutral', key: ' ', shapes: {} },
  blink: { label: 'Blink', key: 'e', blink: true, shapes: {} },
  wink: { label: 'Wink', key: 'w', shapes: { wink: 1 }, transient: 620 },
};

export const KEY_MAP = Object.fromEntries(
  Object.entries(EXPRESSIONS).filter(([, e]) => e.key).map(([n, e]) => [e.key, n])
);
