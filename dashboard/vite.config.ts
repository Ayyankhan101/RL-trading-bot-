import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// Relative base so the built bundle works from any static host or a file path.
export default defineConfig({
  base: './',
  plugins: [react()],
  server: { port: 5173, open: true },
});
