#!/usr/bin/env bash
# macOS / Linux equivalent of build_native_runtime.ps1.
#
# On Windows the runtime is a pre-built llama.dll that the native worker relinks
# against. On macOS/Linux the portable path is to compile the pinned llama.cpp
# submodule from source through CMake (Metal is enabled automatically on Apple
# silicon), producing the same native executables the Python supervisor drives.
#
# Usage:
#   scripts/build_native_runtime.sh [BUILD_DIR] [BUILD_TYPE]
#     BUILD_DIR   default: build          (CMake build directory)
#     BUILD_TYPE  default: Release
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${1:-build}"
BUILD_TYPE="${2:-Release}"
LLAMA_SRC="$ROOT/controlled_inference/vendor/llama.cpp"

if [[ ! -f "$LLAMA_SRC/include/llama.h" ]]; then
  echo "error: pinned llama.cpp submodule missing at $LLAMA_SRC" >&2
  echo "       run: git submodule update --init --recursive" >&2
  exit 1
fi

cmake -S "$ROOT" -B "$ROOT/$BUILD_DIR" -DBUILD_TESTING=ON -DCMAKE_BUILD_TYPE="$BUILD_TYPE"
cmake --build "$ROOT/$BUILD_DIR" --config "$BUILD_TYPE" \
  --target model-worker-native inference-runtime-native model-worker-native-tests

echo "Built native worker:     $ROOT/$BUILD_DIR/model-worker-native"
echo "Built inference runtime: $ROOT/$BUILD_DIR/inference-runtime-native"
