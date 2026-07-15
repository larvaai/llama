# Model Worker v1 operator runbook

## Startup and shutdown

Run `model-worker-validate-manifest config/model.local.json` before starting. Readiness stays false until hashes, model load, template, markers, context, artifact root, and exposure policy pass. Stop with Ctrl+C or the service manager's normal termination signal; the HTTP layer stops accepting work, cancels queued/running records, then applies its hard shutdown deadline.

## Model load failure

Compare the typed readiness error with the manifest path, SHA-256, `runtime_build=b10012`, DLL ABI, context declaration, chat template, and tokenized marker sequences. Never bypass a mismatch. Replace the external runtime/model or update and re-release the manifest intentionally.

## Watchdog restart

`deadline_exceeded` means the dispatcher sent cancel, waited the configured grace, and killed/restarted an unresponsive native process. Check prompt-decode/generation latency, watchdog-kill and worker-restart metrics. The interrupted request is never replayed; the caller decides whether to retry.

## GPU OOM

Keep the failed request terminal, drain traffic, verify manifest GPU layers/context/batch values, and restart. Do not silently lower context or switch model. Publish changed resource settings as a new manifest digest and rerun GPU plus soak gates.

## Artifact quota

Artifacts contain hashes, limits, timestamps, and terminal results—not raw prompts or reasoning by default. Cleanup skips active attempts, refuses paths outside the resolved root, and removes expired/oldest terminal attempts to regain quota. Investigate repeated quota pressure before increasing limits.

## Privacy

Do not enable reasoning event persistence in normal operation. Authorization headers, prompts, raw schemas, and chain-of-thought must not enter application logs. `/metrics` follows the same authentication/exposure policy as inference.
