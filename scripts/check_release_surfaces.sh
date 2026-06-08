#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

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

echo "==> Building and packaging kedi-zed extension..."
(
  cd kedi-zed
  cargo build --release --target wasm32-wasip1

  wasm_path="target/wasm32-wasip1/release/kedi_zed.wasm"
  if [[ ! -f "$wasm_path" ]]; then
    echo "Missing wasm artifact: $wasm_path"
    exit 1
  fi

  version="$(
    grep -E '^version[[:space:]]*=' extension.toml \
      | sed -E 's/^version[[:space:]]*=[[:space:]]*"([^"]+)".*/\1/'
  )"
  version_tag="${version//./}"
  archive_name="kedi-zed-${version_tag}.zip"

  staging="$(mktemp -d)"
  cp extension.toml "$staging/"
  cp "$wasm_path" "$staging/extension.wasm"
  cp -R languages snippets LICENSE "$staging/"
  mkdir -p "$staging/embedded-python-proxy"
  cp embedded-python-proxy/server.mjs "$staging/embedded-python-proxy/"

  mkdir -p "$ROOT/dist"
  (
    cd "$staging"
    zip -qr "$ROOT/dist/$archive_name" .
  )
  rm -rf "$staging"

  echo "Created $ROOT/dist/$archive_name"
)

echo "==> Packaged VSIX: kedi-vscode/$(basename kedi-vscode/*.vsix)"
echo "==> Packaged Zed archive: dist/$(basename dist/kedi-zed-*.zip)"
echo "==> Release surface checks passed."
