import { defineConfig } from 'vite';

// Proxies API and WebSocket traffic to the Flask backend during development.
export default defineConfig({
  server: {
    proxy: {
      '/api': 'http://localhost:5000',
      '/socket.io': {
        target: 'http://localhost:5000',
        ws: true,
      },
    },
  },
});
