# Model Worker v1

Single-model llama.cpp worker with a strict `model-worker.v1` API, fail-closed structured output, fresh context per request, bounded dispatch, cancellation, deadlines, and private-by-default reasoning.

## Clean setup

```powershell
git submodule update --init --recursive
python -m venv .venv
.venv/Scripts/python -m pip install -e ".[test,lint]"
Copy-Item config/model.example.json config/model.local.json
# Fill external model/runtime paths and audited SHA-256 values.
.venv/Scripts/python scripts/verify_runtime.py config/model.local.json
cmake -S . -B build -DBUILD_TESTING=ON
cmake --build build --config Release --target model-worker-native model-worker-native-tests
ctest --test-dir build -C Release --output-on-failure
scripts/build_native_runtime.ps1 -ModelManifest config/model.local.json -BuildDirectory build-runtime
```

Start on loopback:

```powershell
model-worker --model-manifest config/model.local.json --native-executable build-runtime/model-worker-native.exe
```

Non-loopback bind fails unless `MODEL_WORKER_BEARER_TOKEN` is set and `--tls-terminated` or `--trusted-reverse-proxy` is explicit. See [operator runbook](docs/model-worker-runbook.md) and [release checklist](docs/model-worker-release.md).
