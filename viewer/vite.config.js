import { defineConfig } from "vite";

// Relative base so the static build works from any path (file host, GitHub Pages subfolder, ...).
export default defineConfig({ base: "./" });
