# Model Worker v1 release gate

All evidence must be generated from one Git revision and one verified manifest digest.

1. Install/build from the clean-setup section in `README.md`.
2. Run `scripts/release_gate.ps1 -ModelManifest config/model.local.json`.
3. Record native build/test output, unit/property/fake-worker reports, GPU report, 500-request soak output, manifest/runtime hashes, latency, prompt/generation throughput, peak RAM/VRAM, and restart time under `release-evidence/<revision>/`.
4. Confirm loopback default, authenticated external-mode rejection tests, private reasoning defaults, immutable artifacts, quota/retention, graceful drain, and fresh context.
5. A skipped GPU/native/soak gate is not a release pass.

The `controlled_inference/` tree is historical experiment material only and is not imported or packaged by `model_worker`.
