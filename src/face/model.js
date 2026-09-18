import * as THREE from 'three';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';
import { DRACOLoader } from 'three/examples/jsm/loaders/DRACOLoader.js';

/**
 * Runtime for the rigged 3D head.
 *
 * The model is a real GLB: one closed mesh with measured blendshapes, real
 * eyeballs and mouth geometry inside it, and an armature. Nothing here is
 * composited on top of anything else — the previous build's eye patches and
 * mouth plates are what produced the skin-within-skin artefact, and they are
 * gone along with the flat plane they sat on.
 *
 * UNLIT: the photograph already contains its own lighting, baked in by whoever
 * shot it. Lighting it again would double every shadow and destroy the likeness,
 * so every material is swapped to MeshBasicMaterial on load and no light is ever
 * added to the scene. This is also what keeps the head-on view identical to the
 * source photograph.
 */

const DPR_CAP = 2;
const DEG = Math.PI / 180;

export class FaceModel {
  constructor(canvas, { antialias = true } = {}) {
    this.canvas = canvas;
    this.antialias = antialias;
    this.scene = new THREE.Scene();
    this.camera = new THREE.OrthographicCamera(-1, 1, 1, -1, -10, 10);
    this.pointer = { x: 0, y: 0 };
    this.gaze = { x: 0, y: 0 };
    this.weights = {};       // goal morph influences by name
    this.current = {};       // eased
    this.bones = {};
    this.running = false;
    this.idle = true;
    this.freeze = false;
    this.reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    this._clock = new THREE.Clock();
    this._t = 0;
    this._blinkT = 0;
    this._nextBlink = 2 + Math.random() * 3;
  }

  async load(name, base = import.meta.env.BASE_URL || '/') {
    const loader = new GLTFLoader();
    // the mesh is Draco-compressed; the decoder is served locally rather than
    // from a CDN so the page has no third-party dependency at runtime
    const draco = new DRACOLoader();
    draco.setDecoderPath(`${base}draco/`);
    loader.setDRACOLoader(draco);

    const [gltf, meta] = await Promise.all([
      loader.loadAsync(`${base}assets/model/${name}.glb`),
      fetch(`${base}assets/model/${name}.model.json`).then((r) => r.json()),
    ]);
    draco.dispose();

    this.meta = meta;
    this.root = gltf.scene;
    this.scene.add(this.root);

    this.morphMeshes = [];
    this.root.traverse((o) => {
      if (o.isBone) this.bones[o.name] = o;
      if (!o.isMesh) return;
      o.frustumCulled = false;
      o.material = this.unlit(o.material, o.geometry);
      if (o.morphTargetDictionary) this.morphMeshes.push(o);
      if (/tongue/i.test(o.name)) {
        this.tongue = o;
        this.tongueHome = o.position.clone();
      }
    });

    this.renderer = this.renderer || (await this.initRenderer());
    this.renderer.toneMapping = THREE.NoToneMapping;
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.renderer.setClearColor(0x000000, 0);

    // remember rest rotations so pose offsets are relative, not absolute
    this.rest = {};
    for (const [k, b] of Object.entries(this.bones)) {
      b.rotation.order = 'YXZ';
      this.rest[k] = b.rotation.clone();
    }

    this.fit();
    this.resize();
    return this;
  }

  /** Photo in, photo out: no lighting model between the texture and the screen.
   *
   * The textured skin BLENDS; everything else is opaque.
   *
   * These portraits are cutouts with a feathered alpha edge — every strand of
   * the afro is a partially transparent pixel. An alpha cutout has to answer
   * each of those with a hard yes or no, so the silhouette can never match the
   * source and a rim of error runs all the way round the subject. Blending
   * reproduces the feather exactly. The interior (shell, bag, teeth, tongue)
   * stays opaque so it still sorts by depth rather than by draw order, which is
   * what went wrong when everything was transparent at once.
   */
  unlit(mat, geometry) {
    const src = Array.isArray(mat) ? mat[0] : mat;
    const textured = !!src.map;
    // The shell deliberately renders IDENTICALLY to the skin. Tinting it dark
    // makes the turn look better but measurably worse head-on (mean 3.81 vs
    // 1.33 against the source photograph), because a dark shell shows through
    // the hair's semi-transparent fringe. The head-on match is the claim worth
    // protecting, so the band is hidden by geometry and a smaller yaw instead.
    const shell = false;
    const m = new THREE.MeshBasicMaterial({
      map: src.map || null,
      color: textured ? 0xffffff : src.color || 0xffffff,
      transparent: textured,
      alphaTest: 0,
      depthWrite: true,
      depthTest: true,
      side: THREE.DoubleSide,
      toneMapped: false,
    });
    if (m.map) m.map.colorSpace = THREE.SRGBColorSpace;

    // MULTI-VIEW BLEND + SILHOUETTE FADE.
    //
    // Two things happen in one injected shader, and BOTH must apply to every
    // textured part of the bust. Gating this on "has an atlas" was a bug: the
    // shell material never got one, so the fade silently skipped the shell —
    // which is where most of the ear lobe lives — and no amount of tuning the
    // fade did anything at all.
    //
    // 1. The atlas (baked from the front, side and back photographs) is blended
    //    IN only where the surface turns away from the camera. Anything facing
    //    front keeps sampling the original photograph through the original
    //    frontal projection, untouched by the bake's resampling, which is what
    //    holds the head-on match.
    // 2. Surface that faces away is faded out. At the silhouette the bust is
    //    tangent to the view, so rotation swings it edge-on and it smears into
    //    a lobe beside the ear; head-on those faces project to sub-pixel width,
    //    so removing them costs almost nothing where it matters.
    //
    // Both use the REST-pose normal, so the regions are fixed properties of the
    // model rather than sliding about as the head turns.
    // uv1 exists only on the bust (both of its material halves). The eyeballs
    // and mouth parts are separate single-channel meshes: fading THEM by facing
    // eats the eyeball caps, whose rims face sideways, and the eyes go black.
    const isBust = !!geometry?.attributes?.uv1;
    const atlas = isBust ? src.emissiveMap : null;
    if (isBust) {
      m.onBeforeCompile = (shader) => {
        if (atlas) shader.uniforms.atlasMap = { value: atlas };
        shader.vertexShader = shader.vertexShader
          .replace('#include <common>', `#include <common>
            ${atlas ? 'attribute vec2 uv1;\nvarying vec2 vAtlasUv;' : ''}
            varying vec3 vRestNormal;`)
          .replace('#include <begin_vertex>', `#include <begin_vertex>
            ${atlas ? 'vAtlasUv = uv1;' : ''}
            vRestNormal = normal;`);
        shader.fragmentShader = shader.fragmentShader
          .replace('#include <common>', `#include <common>
            ${atlas ? 'uniform sampler2D atlasMap;\nvarying vec2 vAtlasUv;' : ''}
            varying vec3 vRestNormal;`)
          .replace('#include <map_fragment>', `#include <map_fragment>
            {
              // +Z is toward the viewer in glTF's frame after export
              float facing = normalize(vRestNormal).z;
              ${atlas ? `
              vec4 atlasCol = texture2D(atlasMap, vAtlasUv);
              diffuseColor.rgb = mix(diffuseColor.rgb, atlasCol.rgb, smoothstep(0.40, 0.02, facing));
              ` : ''}
              diffuseColor.a *= smoothstep(-0.05, 0.22, facing);
            }`);
      };
      m.customProgramCacheKey = () => (atlas ? 'mv-atlas' : 'mv-fade');
    }

    if (!textured || shell) {
      // The shell hugs the back of the skin and the two are near-coplanar where
      // the head curves away to the silhouette. The opaque shell draws first and
      // writes depth, so the blended skin then LOSES the depth test along those
      // bands and the black shell shows through the hair. Nudging the untextured
      // geometry away from the camera settles it without moving any vertices.
      m.polygonOffset = true;
      m.polygonOffsetFactor = 2;
      m.polygonOffsetUnits = 4;
    }
    return m;
  }

  async initRenderer() {
    const r = new THREE.WebGLRenderer({
      canvas: this.canvas,
      antialias: this.antialias,
      alpha: true,
      preserveDrawingBuffer: true, // the verifier reads pixels back
    });
    this.backend = 'webgl2';
    return r;
  }

  /** Frame the bust exactly as the source photograph frames it, so the head-on
   *  view can be compared against the original pixel for pixel. */
  fit() {
    const box = new THREE.Box3().setFromObject(this.root);
    this.bounds = box;
    this.center = box.getCenter(new THREE.Vector3());
    this.size = box.getSize(new THREE.Vector3());
    this.camera.position.set(this.center.x, this.center.y, 5);
    this.camera.lookAt(this.center.x, this.center.y, 0);
  }

  resize() {
    const el = this.canvas.parentElement || this.canvas;
    const w = el.clientWidth || window.innerWidth;
    const h = el.clientHeight || window.innerHeight;
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, DPR_CAP));
    this.renderer.setSize(w, h, false);

    // the model is built in units where the source image is 1.0 tall
    const imgAspect = this.meta.aspect;
    let halfH = 0.5;
    let halfW = halfH * imgAspect;
    const viewAspect = w / h;
    if (viewAspect > imgAspect) halfW = halfH * viewAspect;
    else halfH = halfW / viewAspect;

    const c = this.camera;
    c.left = -halfW;
    c.right = halfW;
    c.top = halfH;
    c.bottom = -halfH;
    c.position.x = 0;
    c.position.y = 0;
    c.updateProjectionMatrix();
    this.viewport = { w, h };
  }

  setPointer(nx, ny) {
    this.pointer.x = Math.max(-1, Math.min(1, nx));
    this.pointer.y = Math.max(-1, Math.min(1, ny));
  }

  setExpression(weights) {
    this.weights = { ...weights };
  }

  triggerBlink(duration = 0.26) {
    this._blinkDur = duration;
    this._blinkT = duration;
  }

  blinkCurve(t) {
    const k = 1 - t / this._blinkDur;
    if (k < 0.3) return k / 0.3;
    if (k < 0.5) return 1;
    return Math.max(0, 1 - (k - 0.5) / 0.5);
  }

  applyMorphs(goal) {
    for (const mesh of this.morphMeshes) {
      const dict = mesh.morphTargetDictionary;
      for (const nameKey in dict) {
        mesh.morphTargetInfluences[dict[nameKey]] = goal[nameKey] || 0;
      }
    }
  }

  frame() {
    const dt = this.freeze ? 0 : Math.min(this._clock.getDelta(), 0.05);
    this._t += dt;

    const ease = 1 - Math.exp(-6 * dt);
    this.gaze.x += (this.pointer.x - this.gaze.x) * ease;
    this.gaze.y += (this.pointer.y - this.gaze.y) * ease;

    // ---- expression weights, eased ----
    const goal = { ...this.weights };
    if (this.idle && !this.freeze && !this.reducedMotion) {
      this._nextBlink -= dt;
      if (this._nextBlink <= 0) {
        this.triggerBlink();
        this._nextBlink = 2.5 + Math.random() * 4.5;
      }
    }
    if (this._blinkT > 0 && !this.freeze) {
      this._blinkT -= dt;
      const p = this.blinkCurve(Math.max(this._blinkT, 0));
      goal.blink = Math.max(goal.blink || 0, p);
    }

    const k = 1 - Math.exp(-11 * dt);
    const names = new Set([...Object.keys(this.current), ...Object.keys(goal)]);
    for (const n of names) {
      const g = goal[n] || 0;
      const c = this.current[n] || 0;
      const v = c + (g - c) * k;
      if (Math.abs(v) < 1e-4 && g === 0) delete this.current[n];
      else this.current[n] = v;
    }
    this.applyMorphs(this.current);

    // ---- pose ----
    const maxYaw = (this.meta.maxYawDeg || 26) * DEG;
    const idleX = this.idle && !this.reducedMotion && !this.freeze
      ? Math.sin(this._t * 0.55) * 0.015
      : 0;
    const idleY = this.idle && !this.reducedMotion && !this.freeze
      ? Math.sin(this._t * 0.8 + 1.1) * 0.010
      : 0;

    const yaw = this.gaze.x * maxYaw + idleX;
    const pitch = -this.gaze.y * maxYaw * 0.55 + idleY;

    // the head takes most of the turn, the neck a third of it, so the motion
    // travels down the body instead of the skull pivoting on a fixed stump
    this.poseBone('head', pitch * 0.7, yaw * 0.7);
    this.poseBone('neck', pitch * 0.3, yaw * 0.3);

    // eyes lead the head: they arrive first, which is what makes it read as
    // looking at you rather than being aimed at you
    // a real eye travels far less than this used to allow; 0.42 rad is 24deg,
    // which rolls the cap's rim into the aperture
    const eyeYaw = this.gaze.x * 0.16;
    const eyePitch = -this.gaze.y * 0.11;
    // glTF strips the dot from bone names, so eye.L arrives as eyeL
    this.poseBone('eyeL', eyePitch, eyeYaw);
    this.poseBone('eyeR', eyePitch, eyeYaw);

    // the tongue is modelled inside the mouth; sticking it out is a real
    // translation of a real object, not a sprite fading in over the chin
    if (this.tongue) {
      const t = this.current.tongue || 0;
      const reach = (this.meta.headRadius || 0.18) * 0.9;
      this.tongue.position.set(
        this.tongueHome.x,
        this.tongueHome.y + t * reach * 0.55,   // +y is forward in glTF's frame
        this.tongueHome.z - t * reach * 0.55
      );
    }

    // jaw follows the measured opening of whichever expression is active
    let jaw = 0;
    for (const n in this.current) jaw = Math.max(jaw, (this.meta.jawOpen?.[n] || 0) * this.current[n]);
    this.poseBone('jaw', jaw * 14 * DEG, 0);

    this.renderer.render(this.scene, this.camera);
  }

  poseBone(name, rx, ry) {
    const b = this.bones[name];
    if (!b) return;
    const r = this.rest[name];
    b.rotation.set(r.x + rx, r.y + ry, r.z);
  }

  start() {
    if (this.running) return;
    this.running = true;
    this._clock.start();
    const loop = () => {
      if (!this.running) return;
      this.frame();
      this._raf = requestAnimationFrame(loop);
    };
    this._raf = requestAnimationFrame(loop);
  }

  stop() {
    this.running = false;
    cancelAnimationFrame(this._raf);
  }

  dispose() {
    this.stop();
    this.scene.traverse((o) => {
      if (o.geometry) o.geometry.dispose();
      if (o.material) {
        const ms = Array.isArray(o.material) ? o.material : [o.material];
        ms.forEach((m) => {
          m.map?.dispose();
          m.dispose();
        });
      }
    });
    this.renderer?.dispose?.();
  }
}
