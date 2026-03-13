#!/usr/bin/env bash

set -euo pipefail

MODE="${1:-all}"

PROMPT='A clean poster with the word "HELLO" centered'
TEXT='HELLO'
OUT_DIR="output/backend_smoke_test"

mkdir -p "$OUT_DIR"

run_zimage() {
  echo "==> Testing zimage backend"
  python3 generate.py \
    --backend zimage \
    --prompt "$PROMPT" \
    --text "$TEXT" \
    --output "$OUT_DIR/zimage.png" \
    --steps 12 \
    --seed 42
}

run_qwen() {
  echo "==> Testing qwen backend"
  python3 generate.py \
    --backend qwen \
    --prompt "$PROMPT" \
    --text "$TEXT" \
    --output "$OUT_DIR/qwen.png" \
    --steps 12 \
    --seed 42
}

case "$MODE" in
  zimage)
    run_zimage
    ;;
  qwen)
    run_qwen
    ;;
  all)
    run_zimage
    run_qwen
    ;;
  *)
    echo "Usage: bash scripts/test_backends.sh [zimage|qwen|all]"
    exit 1
    ;;
esac

echo "Done. Outputs are in $OUT_DIR"
