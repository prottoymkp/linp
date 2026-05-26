import tailwindcss from '@tailwindcss/vite';
import react from '@vitejs/plugin-react';
import {defineConfig} from 'vite';

export default defineConfig({
  base: '/linp/',
  plugins: [react(), tailwindcss()],
  root: 'web',
  build: {
    outDir: '../dist',
    emptyOutDir: true,
  },
});
