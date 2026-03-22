import path from "node:path";
import tailwindcss from "@tailwindcss/vite";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

const outDir = path.resolve(
  __dirname,
  "../../services/mvp_ui_api/static/investigation",
);

export default defineConfig({
  plugins: [react(), tailwindcss()],
  base: "/static/investigation/",
  build: {
    outDir,
    emptyOutDir: true,
  },
  optimizeDeps: {
    include: ["react-force-graph-2d"],
  },
  server: {
    port: 5173,
    proxy: {
      "/api": { target: "http://127.0.0.1:8765", changeOrigin: true },
    },
  },
});
