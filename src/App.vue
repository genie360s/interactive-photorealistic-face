<script setup>
import { ref, shallowRef, onMounted, onBeforeUnmount, computed } from 'vue';
import { FaceModel } from './face/model.js';
import { EXPRESSIONS, KEY_MAP } from './face/expressions3d.js';

const canvas = ref(null);
const viewport = ref(null);
const stage = shallowRef(null);

const character = ref('male');
const current = ref('neutral');
const loading = ref(true);
const error = ref('');
const backend = ref('');
const fps = ref(0);

// ?verify=1 pins the face at its rest pose and hides the chrome so the canvas
// can be compared against the source photograph 1:1. ?webgpu=1 opts into the
// WebGPU build (see stage.js for why it is not the default).
const params = new URLSearchParams(location.search);
const verify = params.has('verify');
const wantWebGPU = params.has('webgpu');
const textureMode = params.get('texture') || undefined;

// NB: spread last would clobber the identity with the keyboard key, since each
// expression already has its own `key` field. Name it separately.
const list = computed(() => Object.entries(EXPRESSIONS).map(([name, v]) => ({ ...v, name })));

let transientTimer = 0;

function apply(name) {
  const e = EXPRESSIONS[name];
  if (!e || !stage.value) return;
  current.value = name;
  stage.value.setExpression(e.shapes || {});
  if (e.blink) stage.value.triggerBlink();

  clearTimeout(transientTimer);
  if (e.transient) {
    transientTimer = setTimeout(() => apply('neutral'), e.transient);
  }
}

function onKey(ev) {
  if (ev.metaKey || ev.ctrlKey || ev.altKey) return;
  const name = KEY_MAP[ev.key.toLowerCase()];
  if (!name) return;
  ev.preventDefault();
  apply(name);
}

function pointerFrom(ev) {
  const el = viewport.value;
  if (!el || !stage.value) return;
  const r = el.getBoundingClientRect();
  const p = ev.touches?.[0] || ev;
  stage.value.setPointer(
    ((p.clientX - r.left) / r.width) * 2 - 1,
    -(((p.clientY - r.top) / r.height) * 2 - 1)
  );
}

// With no pointer at all (touch idle, or a keyboard-only visitor) the face
// should still feel alive rather than staring dead ahead.
function onLeave() {
  stage.value?.setPointer(0, 0);
}

async function loadCharacter(name) {
  loading.value = true;
  error.value = '';
  try {
    stage.value?.dispose();
    const s = new FaceModel(canvas.value, { antialias: params.get('aa') !== '0' });
    s.freeze = verify;
    s.idle = !params.has('still');
    await s.load(name);
    stage.value = s;
    backend.value = s.backend;
    character.value = name;
    apply('neutral');
    s.start();
  } catch (e) {
    error.value = e.message || String(e);
    console.error('[face]', e);
  } finally {
    loading.value = false;
  }
}

let ro, fpsTimer;
onMounted(async () => {
  await loadCharacter('male');

  ro = new ResizeObserver(() => {
    stage.value?.resize();
  });
  ro.observe(viewport.value);

  window.addEventListener('keydown', onKey);
  window.addEventListener('pointermove', pointerFrom, { passive: true });
  window.addEventListener('touchmove', pointerFrom, { passive: true });
  window.addEventListener('pointerleave', onLeave);
  // stop burning battery in a background tab
  document.addEventListener('visibilitychange', () => {
    if (document.hidden) stage.value?.stop();
    else stage.value?.start();
  });

  let frames = 0;
  const seen = () => {
    frames++;
    requestAnimationFrame(seen);
  };
  requestAnimationFrame(seen);
  fpsTimer = setInterval(() => {
    fps.value = frames;
    frames = 0;
  }, 1000);

  // expose for the verification harness
  window.__face = {
    stage: () => stage.value,
    apply,
    loadCharacter,
    /** render one frame and hand back the canvas pixels, for the verifier */
    snapshot() {
      const s = stage.value;
      s.frame();
      const c = document.createElement('canvas');
      c.width = s.canvas.width;
      c.height = s.canvas.height;
      const ctx = c.getContext('2d', { willReadFrequently: true });
      ctx.drawImage(s.canvas, 0, 0);
      return ctx.getImageData(0, 0, c.width, c.height);
    },
  };
});

onBeforeUnmount(() => {
  ro?.disconnect();
  clearInterval(fpsTimer);
  clearTimeout(transientTimer);
  window.removeEventListener('keydown', onKey);
  window.removeEventListener('pointermove', pointerFrom);
  window.removeEventListener('touchmove', pointerFrom);
  window.removeEventListener('pointerleave', onLeave);
  stage.value?.dispose();
});
</script>

<template>
  <div class="stage" :class="{ verify }">
    <header class="bar top">
      <span class="brand">Interactive Face</span>
      <div class="readout">
        <button
          v-for="c in ['male', 'female']"
          :key="c"
          :aria-pressed="character === c"
          @click="loadCharacter(c)"
        >
          {{ c === 'male' ? 'M' : 'F' }}
        </button>
        <span v-if="backend">{{ backend }}</span>
        <span>{{ fps }} fps</span>
      </div>
    </header>

    <div ref="viewport" class="viewport">
      <canvas ref="canvas" aria-label="Interactive portrait that follows your cursor"></canvas>
      <div v-if="loading" class="loading">
        <div class="ring"></div>
        <span>loading</span>
      </div>
      <div v-else-if="error" class="loading">{{ error }}</div>
    </div>

    <nav class="bar bottom" aria-label="Expressions">
      <button
        v-for="e in list"
        :key="e.name"
        :aria-pressed="current === e.name"
        @click="apply(e.name)"
      >
        {{ e.label }}<kbd>{{ e.key === ' ' ? 'space' : e.key.toUpperCase() }}</kbd>
      </button>
    </nav>
    <p class="hint">Move the cursor — the head and eyes follow. Tap a reaction or press its key.</p>
  </div>
</template>
