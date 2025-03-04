import { defineConfig } from "tsup";

export default defineConfig({
    entry: ["src/front.ts"],  // Fichier à compiler
    outDir: "public",         // Génère le .js dans /public
    format: ["esm"],          // Format ES Module
    sourcemap: true,          // Debug facile
    minify: false,            // Pas besoin de minifier en dev
    watch: true,              // Mode watch
});