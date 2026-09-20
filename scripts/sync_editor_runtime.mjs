// The VS Code package owns the bootstrap source; Zed embeds the same built artifact.
import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const source = path.join(root, "kedi-vscode/runtime/bootstrap.cjs");
const target = path.join(root, "kedi-zed/runtime/bootstrap.cjs");
const content = await fs.readFile(source);
if (process.argv.includes("--check")) {
    if (!content.equals(await fs.readFile(target))) throw new Error("Editor bootstrap artifacts differ; run npm run build:runtime in kedi-vscode and node scripts/sync_editor_runtime.mjs");
    console.log("Editor runtime bootstrap artifacts match");
} else {
    await fs.mkdir(path.dirname(target), { recursive: true });
    await fs.copyFile(source, target);
}
