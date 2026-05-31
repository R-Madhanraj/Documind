// vite.config.js
//
// Vite config with a proxy so /api calls from React go to FastAPI.
// Without this, you'd have to write full URLs in every api.js call.
// The proxy rewrites /api → http://localhost:8000/api automatically.

import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": {
        target: "http://localhost:8000",
        changeOrigin: true,
      },
    },
  },
});