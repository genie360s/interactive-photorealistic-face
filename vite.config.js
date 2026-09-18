import { defineConfig } from 'vite';
import vue from '@vitejs/plugin-vue';

// Match against RESOLVED module ids, which are real file paths — `three/webgpu`
// resolves to node_modules/three/build/three.webgpu.js, so testing for the
// import specifier silently never matches and the WebGPU build gets folded into
// the eager three chunk.
const rx = {
  threeWebgpu: /node_modules[\\/]three[\\/]build[\\/]three\.(webgpu|tsl)/,
  three: /node_modules[\\/]three[\\/]/,
  p5: /node_modules[\\/]p5[\\/]/,
  vue: /node_modules[\\/]@?vue/,
};

export default defineConfig({
  base: './',
  plugins: [vue()],
  build: {
    target: 'es2020',
    // every asset here is already compressed (webp) or a binary mesh; inlining
    // would only bloat the JS
    assetsInlineLimit: 0,
    rollupOptions: {
      output: {
        manualChunks(id) {
          if (rx.threeWebgpu.test(id)) return 'three-webgpu';
          if (rx.three.test(id)) return 'three';
          // p5 must be isolated or its chunk becomes the home for shared interop
          // helpers, which drags all ~1.2MB of it into the eager preload graph
          if (rx.p5.test(id)) return 'p5';
          if (rx.vue.test(id)) return 'vue';
        },
      },
    },
  },
});
