import { defineConfig } from 'vite';

export default defineConfig({
  server: {
    host: true, // Allow access from network (for mobile testing)
    port: 5173
  },
  build: {
    outDir: 'dist',
    sourcemap: false
  }
});
