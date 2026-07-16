# Model Worker v1

Single-model llama.cpp worker with a strict `model-worker.v1` API, fail-closed structured output, fresh context per request, bounded dispatch, cancellation, deadlines, and private-by-default reasoning.

## Clean setup (macOS / Linux)

Requires Python ≥ 3.11, CMake ≥ 3.24, and a C++20 compiler (Apple Clang on
macOS; Metal is enabled automatically on Apple silicon).

```bash
git submodule update --init --recursive
python3 -m venv .venv
.venv/bin/python -m pip install -e ".[test,lint]"

# Python worker + supervisor test suite (no model or GPU required):
.venv/bin/python -m pytest -m "not gpu and not soak"

# Build the native inference binaries from the pinned llama.cpp submodule:
scripts/build_native_runtime.sh          # → build/model-worker-native
ctest --test-dir build -C Release --output-on-failure

# Point a manifest at your model + built runtime, then verify hashes:
cp config/model.example.json config/model.local.json
# Edit config/model.local.json: set gguf_path, gguf_sha256, runtime.directory,
# runtime.library (libllama.dylib on macOS), and runtime.library_sha256.
.venv/bin/python scripts/verify_runtime.py config/model.local.json
```

Start on loopback:

```bash
model-worker --model-manifest config/model.local.json --native-executable build/model-worker-native
```

Non-loopback bind fails unless `MODEL_WORKER_BEARER_TOKEN` is set and
`--tls-terminated` or `--trusted-reverse-proxy` is explicit. See
[operator runbook](docs/model-worker-runbook.md) and
[release checklist](docs/model-worker-release.md).

### Windows

The original Windows path uses a pre-built `llama.dll` runtime and MSVC:
`scripts/build_native_runtime.ps1 -ModelManifest config/model.local.json
-BuildDirectory build-runtime`, then run against
`build-runtime/model-worker-native.exe`. The manifest `runtime.library` field
defaults per-platform (`llama.dll` / `libllama.dylib` / `libllama.so`).
