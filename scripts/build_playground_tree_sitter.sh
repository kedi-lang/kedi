#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STATIC_ROOT="$ROOT/playground/src/playground/static"
WORK_DIR="$(mktemp -d)"
trap 'rm -rf "$WORK_DIR"' EXIT

TREE_SITTER_VERSION="0.26.6"

npx -y "tree-sitter-cli@$TREE_SITTER_VERSION" build \
  --wasm \
  --output "$WORK_DIR/tree-sitter-kedi.wasm" \
  "$ROOT/tree-sitter-kedi"

npm pack --silent "web-tree-sitter@$TREE_SITTER_VERSION" \
  --pack-destination "$WORK_DIR" \
  >/dev/null
tar -xzf "$WORK_DIR/web-tree-sitter-$TREE_SITTER_VERSION.tgz" -C "$WORK_DIR"

mkdir -p "$STATIC_ROOT/grammars/kedi" "$STATIC_ROOT/vendor"
cp "$WORK_DIR/tree-sitter-kedi.wasm" "$STATIC_ROOT/grammars/kedi/"
cp "$ROOT/tree-sitter-kedi/queries/highlights.scm" "$STATIC_ROOT/grammars/kedi/"
cp "$WORK_DIR/package/web-tree-sitter.js" "$STATIC_ROOT/vendor/"
cp "$WORK_DIR/package/web-tree-sitter.wasm" "$STATIC_ROOT/vendor/"
