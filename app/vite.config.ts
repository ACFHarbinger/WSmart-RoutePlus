import fs from "node:fs";
import path from "node:path";
import { defineConfig, type Plugin } from "vite";
import react from "@vitejs/plugin-react";

function duckdbSourcemapPlugin(): Plugin {
  const ensureMapFile = () => {
    const mapPath = path.resolve(
      __dirname,
      "node_modules/@duckdb/duckdb-wasm/dist/duckdb-browser-mvp.worker.js.map"
    );
    const distDir = path.dirname(mapPath);
    if (fs.existsSync(distDir) && !fs.existsSync(mapPath)) {
      fs.writeFileSync(
        mapPath,
        JSON.stringify({
          version: 3,
          file: "duckdb-browser-mvp.worker.js",
          sources: [],
          mappings: "",
        })
      );
    }
  };

  return {
    name: "duckdb-sourcemap-fix",
    buildStart() {
      ensureMapFile();
    },
    configureServer(server) {
      ensureMapFile();
      server.middlewares.use((req, res, next) => {
        if (req.url?.includes("duckdb-browser-mvp.worker.js.map")) {
          res.setHeader("Content-Type", "application/json");
          res.end(
            JSON.stringify({
              version: 3,
              file: "duckdb-browser-mvp.worker.js",
              sources: [],
              mappings: "",
            })
          );
          return;
        }
        next();
      });
    },
  };
}

export default defineConfig({
  plugins: [react(), duckdbSourcemapPlugin()],
  clearScreen: false,
  server: {
    port: 5173,
    strictPort: true,
    watch: {
      ignored: ["**/src-tauri/**"],
    },
  },
  envPrefix: ["VITE_", "TAURI_"],
  build: {
    target: ["es2022", "chrome110", "safari16"],
    minify: !process.env.TAURI_DEBUG ? "esbuild" : false,
    sourcemap: !!process.env.TAURI_DEBUG,
    rollupOptions: {
      output: {
        manualChunks(id) {
          if (id.includes("echarts") || id.includes("zrender")) return "echarts";
          if (id.includes("maplibre-gl")) return "maplibre";
          if (id.includes("@deck.gl") || id.includes("deck.gl")) return "deckgl";
          if (id.includes("@deck.gl/geo-layers")) return "deckgl";
          if (id.includes("@monaco-editor") || id.includes("monaco-editor")) return "monaco";
          if (id.includes("@duckdb/duckdb-wasm")) return "duckdb";
          if (id.includes("three") || id.includes("@react-three")) return "r3f";
          if (id.includes("sigma") || id.includes("graphology")) return "sigma";
        },
      },
    },
  },
});
