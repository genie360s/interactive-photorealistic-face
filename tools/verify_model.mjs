/**
 * Verification for the 3D model.
 *
 * Two questions, kept separate because they deserve different confidence:
 *
 *  1. NEUTRAL, HEAD ON — the model's rest pose against the source photograph.
 *     The texture is a frontal projection of that photograph and the camera is
 *     the same orthographic front view, so this SHOULD be near-exact. It is the
 *     strongest claim in the build and the one worth measuring hardest.
 *
 *  2. EXPRESSIONS — each pose against the photograph it was measured from.
 *     This can never be exact: the geometry moves to match the expression but
 *     the skin texture stays the neutral photograph, so wrinkles, teeth and
 *     shadows that only exist in the expression shot are missing. The number
 *     here measures how well the measured blendshape reproduces the real
 *     deformation, and it is reported as such rather than as "pixel perfect".
 *
 * Both are reported per alpha band, because these are cutout portraits and the
 * antialiased silhouette behaves differently from the subject.
 *
 * Run:  node tools/verify_model.mjs [--url http://localhost:5180]
 */

import { chromium } from 'playwright';
import { readFileSync, writeFileSync, mkdirSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = path.dirname(path.dirname(fileURLToPath(import.meta.url)));
const OUT = path.join(ROOT, 'verification');
const i = process.argv.indexOf('--url');
const BASE = i > -1 ? process.argv[i + 1] : 'http://localhost:5180';

// expression -> the reference photograph it was measured from
const SHOTS = {
  male: {
    dir: 'male_characters',
    neutral: 'male_character.png',
    shapes: {
      smile: 'male_character_smile.png',
      smileTeeth: 'male_character_smile_with_teeth.png',
      laugh: 'male_character_laugh.png',
      cry: 'male_character_cry.png',
      fear: 'male_character_fear.png',
      kiss: 'male_character_kiss.png',
      wink: 'male_character_wink.png',
      tongue: 'male_character_boo_tongue_out.png',
    },
  },
  female: {
    dir: 'female_characters',
    neutral: 'female_character.png',
    shapes: {
      smile: 'female_character_smile.png',
      smileTeeth: 'female_character_smile_with_teeth.png',
      laugh: 'female_character_laugh.png',
      cry: 'female_character_crying.png',
      fear: 'female_character_fear.png',
      kiss: 'female_character_kiss.png',
      wink: 'female_character_wink.png',
      tongue: 'female_character_boo_with_tongue.png',
    },
  },
};

async function diffInPage(refDataUrl) {
  const load = async (url) => {
    const img = new Image();
    await new Promise((res, rej) => {
      img.onload = res;
      img.onerror = () => rej(new Error('reference failed to decode'));
      img.src = url;
    });
    const c = document.createElement('canvas');
    c.width = img.naturalWidth;
    c.height = img.naturalHeight;
    const x = c.getContext('2d', { willReadFrequently: true });
    x.drawImage(img, 0, 0);
    return x.getImageData(0, 0, c.width, c.height);
  };

  const ref = await load(refDataUrl);
  const got = window.__face.snapshot();
  if (ref.width !== got.width || ref.height !== got.height) {
    return { error: `size ${got.width}x${got.height} vs ref ${ref.width}x${ref.height}` };
  }

  const N = ref.width * ref.height;
  const band = () => ({ n: 0, sum: 0, max: 0, exact: 0, over1: 0, over4: 0, over16: 0 });
  const bands = { opaque: band(), edge: band(), clear: band() };
  const heat = new Uint8ClampedArray(N * 4);

  for (let i = 0; i < N; i++) {
    const o = i * 4;
    const rA = ref.data[o + 3];
    const ra = rA / 255;
    const ga = got.data[o + 3] / 255;
    let d = 0;
    for (let k = 0; k < 3; k++) {
      d = Math.max(d, Math.abs(ref.data[o + k] * ra - got.data[o + k] * ga));
    }
    d = Math.round(d);
    const b = rA === 255 ? bands.opaque : rA === 0 ? bands.clear : bands.edge;
    b.n++;
    b.sum += d;
    if (d > b.max) b.max = d;
    if (d === 0) b.exact++;
    if (d > 1) b.over1++;
    if (d > 4) b.over4++;
    if (d > 16) b.over16++;
    const v = Math.min(255, d * 6);
    heat[o] = heat[o + 1] = heat[o + 2] = v;
    heat[o + 3] = 255;
  }
  const sum = (b) => ({
    pixels: b.n,
    mean: b.n ? b.sum / b.n : 0,
    max: b.max,
    exactPct: b.n ? (100 * b.exact) / b.n : 0,
    over1Pct: b.n ? (100 * b.over1) / b.n : 0,
    over4Pct: b.n ? (100 * b.over4) / b.n : 0,
    over16Pct: b.n ? (100 * b.over16) / b.n : 0,
  });
  const hc = document.createElement('canvas');
  hc.width = ref.width;
  hc.height = ref.height;
  hc.getContext('2d').putImageData(new ImageData(heat, ref.width, ref.height), 0, 0);
  return {
    width: ref.width,
    height: ref.height,
    opaque: sum(bands.opaque),
    edge: sum(bands.edge),
    heatmap: hc.toDataURL('image/png'),
  };
}

const f2 = (x) => x.toFixed(2).padStart(6);

async function run() {
  mkdirSync(OUT, { recursive: true });
  const browser = await chromium.launch();
  const rows = [];
  let failed = false;

  for (const [name, cfg] of Object.entries(SHOTS)) {
    const meta = JSON.parse(
      readFileSync(path.join(ROOT, 'assets/facedata', `${name}.json`), 'utf8')
    );
    const { width, height } = meta.image;
    const page = await browser.newPage({
      viewport: { width, height },
      deviceScaleFactor: 1,
      reducedMotion: 'reduce',
    });
    page.on('console', (m) => m.type() === 'error' && console.error('  page:', m.text()));
    await page.goto(`${BASE}/?verify=1&still=1`, { waitUntil: 'networkidle' });
    await page.waitForFunction(() => window.__face?.stage()?.morphMeshes?.length, null, {
      timeout: 60000,
    });
    if (name !== 'male') {
      await page.evaluate((n) => window.__face.loadCharacter(n), name);
      await page.waitForFunction(
        (n) => window.__face.stage()?.meta?.name === n, name, { timeout: 60000 }
      );
    }
    await page.waitForTimeout(400);

    const dataUrl = (rel) =>
      `data:image/png;base64,${readFileSync(path.join(ROOT, rel)).toString('base64')}`;

    const cases = [['neutral', cfg.neutral, {}]];
    for (const [shape, file] of Object.entries(cfg.shapes)) {
      cases.push([shape, file, { [shape]: 1 }]);
    }

    for (const [label, file, shapes] of cases) {
      await page.evaluate((w) => {
        const s = window.__face.stage();
        s.freeze = false;
        s.idle = false;
        s.setExpression(w);
        s.current = { ...w };       // skip the ease, hold the pose exactly
        s.applyMorphs(s.current);
        s.frame();
        s.freeze = true;
      }, shapes);
      await page.waitForTimeout(120);

      const r = await page.evaluate(
        diffInPage,
        dataUrl(`input/character_references/${cfg.dir}/${file}`)
      );
      if (r.error) {
        console.error(`FAIL ${name}/${label}: ${r.error}`);
        failed = true;
        continue;
      }
      writeFileSync(
        path.join(OUT, `model-${name}-${label}.png`),
        Buffer.from(r.heatmap.split(',')[1], 'base64')
      );
      rows.push({ character: name, pose: label, ...r });
      const o = r.opaque;
      console.log(
        `${name.padEnd(7)} ${label.padEnd(11)} subject mean ${f2(o.mean)}  max ${String(o.max).padStart(3)}` +
          `  exact ${f2(o.exactPct)}%  >4 ${f2(o.over4Pct)}%  >16 ${f2(o.over16Pct)}%`
      );
    }
    await page.close();
  }
  await browser.close();

  // Gate ONLY the neutral head-on pose. It is the claim this build controls:
  // same camera, same projection, same pixels. Expression numbers are reported
  // but not gated, because the skin texture cannot follow the deformation.
  for (const r of rows.filter((x) => x.pose === 'neutral')) {
    if (r.opaque.mean > 6 || r.opaque.over16Pct > 3) {
      console.error(
        `\nFAIL ${r.character}: neutral head-on does not reproduce the source photo ` +
          `(mean ${r.opaque.mean.toFixed(2)}, ${r.opaque.over16Pct.toFixed(2)}% off by >16)`
      );
      failed = true;
    }
  }

  writeFileSync(
    path.join(OUT, 'model-report.json'),
    JSON.stringify(rows.map(({ heatmap, ...r }) => r), null, 2)
  );
  console.log(`\nreport + heatmaps -> ${path.relative(ROOT, OUT)}/`);
  if (failed) process.exit(1);
  console.log('PASS');
}

run().catch((e) => {
  console.error(e);
  process.exit(1);
});
