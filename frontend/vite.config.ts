import react from "@vitejs/plugin-react";
import { defineConfig } from "vitest/config";

export default defineConfig({
  plugins: [react()],
  server: { host: "127.0.0.1", port: 5173, strictPort: true },
  test: {
    environment: "jsdom",
    setupFiles: ["./src/test/setup.ts"],
    restoreMocks: true,
    // styles.test.ts reads the palette from the stylesheet (?raw); other CSS stays unprocessed in tests
    css: { include: [/styles\.css/] },
  },
});
