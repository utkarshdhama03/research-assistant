// Runs Vitest from a working directory whose Windows drive letter is upper case. Started from "c:\..."
// (as some terminals open the folder), Vitest loads its runner twice and every suite fails with
// "Vitest failed to find the current suite". On other systems the directory is unchanged.
import { spawnSync } from "node:child_process";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url)).replace(/^[a-z]:/, (drive) => drive.toUpperCase());
const result = spawnSync(process.execPath, [join(here, "node_modules", "vitest", "vitest.mjs"), ...process.argv.slice(2)], {
  cwd: here,
  stdio: "inherit",
});
process.exit(result.status ?? 1);
