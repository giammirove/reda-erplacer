#!/usr/bin/env bash

set -euo pipefail

rm -rf ./images

BINARY="./target/release/reda-erplacer"
THREADS="${RAYON_NUM_THREADS:-$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)}"
VERBOSE="${VERBOSE:-}"

usage() {
  echo "Usage: $0 <test_name> [--verbose] [--args]"
}

[ $# -lt 1 ] && usage

TEST="$1"
shift

EXTRA_ARGS=()
for arg in "$@"; do
  EXTRA_ARGS+=("$arg")
done

TEST_DIR="${TEST}"
[ -d "$TEST_DIR" ] || {
  echo "Error: test directory '$TEST_DIR' not found"
  exit 1
}

LEF=$(find "$TEST_DIR" -maxdepth 1 -name "*.lef" ! -name ".*" | head -1)
DEF=$(find "$TEST_DIR" -maxdepth 1 -name "*.def" ! -name ".*" | head -1)

[ -z "$LEF" ] && {
  echo "Error: no .lef file in $TEST_DIR"
  exit 1
}

[ -z "$DEF" ] && {
  echo "Error: no .def file in $TEST_DIR"
  exit 1
}


[ -x "$BINARY" ] && {
  echo "Binary not found, building..."
  cargo build --release
}

mkdir -p images
export RAYON_NUM_THREADS="$THREADS"

ARGS=(--lef "$LEF" --def "$DEF")

exec "$BINARY" "${ARGS[@]}" "${EXTRA_ARGS[@]}"
