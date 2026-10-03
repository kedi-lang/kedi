#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
node scripts/sync_editor_runtime.mjs --check

vscode_version="$(node -p "require('./kedi-vscode/package.json').version")"
vscode_vsix="kedi-vscode/kedi-vscode-${vscode_version}.vsix"
zed_version="$(
  grep -E '^version[[:space:]]*=' kedi-zed/extension.toml \
    | sed -E 's/^version[[:space:]]*=[[:space:]]*"([^"]+)".*/\1/'
)"
zed_archive="dist/kedi-zed-${zed_version//./}.zip"

echo "==> Verifying submodule pins..."
if git submodule status --recursive | grep -E '^[-+]'; then
  echo "Submodule pins are uninitialized or out of sync with the superproject."
  git submodule status --recursive
  exit 1
fi

echo "==> Building tree-sitter-kedi parser package..."
(
  cd tree-sitter-kedi
  npm ci
  npm run generate
)

echo "==> Compiling, bundling, and packaging kedi-vscode..."
(
  cd kedi-vscode
  npm ci
  npm run compile
  npm run bundle
  npm run package
)
node scripts/sync_editor_runtime.mjs --check

echo "==> Building and packaging kedi-zed extension..."
(
  cd kedi-zed
  cargo build --release --target wasm32-wasip1

  wasm_path="target/wasm32-wasip1/release/kedi_zed.wasm"
  if [[ ! -f "$wasm_path" ]]; then
    echo "Missing wasm artifact: $wasm_path"
    exit 1
  fi

  staging="$(mktemp -d)"
  cp extension.toml "$staging/"
  cp "$wasm_path" "$staging/extension.wasm"
  cp -R languages snippets debug_adapter_schemas LICENSE "$staging/"
  mkdir -p "$staging/embedded-python-proxy"
  cp embedded-python-proxy/server.mjs "$staging/embedded-python-proxy/"

  mkdir -p "$ROOT/dist"
  (
    cd "$staging"
    zip -qr "$ROOT/$zed_archive" .
  )
  rm -rf "$staging"

  echo "Created $ROOT/$zed_archive"
)

test -f "$vscode_vsix"
test -f "$zed_archive"
echo "==> Packaged VSIX: $vscode_vsix"
echo "==> Packaged Zed archive: $zed_archive"
echo "==> Release surface checks passed."
