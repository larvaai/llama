# Model Worker v1 release gate

All evidence must be generated from one Git revision and one verified manifest digest.

## Latest attestation — 2026-07-16

Model Worker M0 đã pass consolidated release gate từ clean runtime revision
`b38b6df32755c55e668ac11e6c8f3e8b1c2ad46b`. Evidence có thẩm quyền nằm tại
`release-evidence/b38b6df32755c55e668ac11e6c8f3e8b1c2ad46b/`; machine-readable copy
cho handoff nằm ở
`artifacts/inference-runtime/2026-07-16-m0-release-attestation.json`.

- Manifest digest: `sha256:afb03af1a03f070be8ba51f076f286065ae3574e317eec7dd9587bb303a1ae2f`.
- Runtime build: `b10012`; model digest:
  `sha256:b68fbb8167d4e0a39c8157d87ea880a38d6c593c2d7b92c153212496f635eb46`.
- Native binary digest:
  `sha256:d2dbb0f4a192e66e6c4d3381dcc9a03d30d5aa7dbec099c08ab03eb78ab6fd09`.
- Unit/property: 380 pass, 3 Windows symlink skips; fake-worker integration:
  37 pass; native build/CTest, real fault, restart recovery và GPU gates pass.
- Soak: 500/500 pass, 0 failure, process generation `[2]`; p50/p95/max
  `1,937/3,578/4,359 s`.
- Throughput: `2.038,257` prompt token/s và `86,623` generated token/s.
- Peak process-tree RSS: `5.637.550.080 byte`; process-scoped WDDM VRAM:
  `5.757,438 MiB`.
- `summary.json` SHA-256:
  `dabead013b42a5f901bbe5ac5ea7a481a9a6eb4812331886688ff5426c9c9bf5`.

Commit tài liệu sau attestation không đổi release identity trên. Bất kỳ thay đổi
model manifest, runtime build, model digest hoặc native binary nào đều cần gate
mới và không được kế thừa attestation này.

## Procedure

1. Install/build from the clean-setup section in `README.md`.
2. Run `scripts/release_gate.ps1 -ModelManifest config/model.local.json`.
3. Gate tự build binary runtime đã verify, launch đúng service đó, kiểm
   `revision`/manifest/model/runtime/native hash và không dùng service đã chạy từ
   trước.
4. Gate phải kill đúng native child, quan sát `/ready=503` + `DEGRADED`, kích
   recovery bằng request thật, rồi chứng minh PID mới, process generation tăng
   đúng một và executable hash không đổi.
5. Record native build/test output, unit/property/fake-worker reports, real
   native fault report, restart-recovery, GPU report, 500-request soak, latency,
   prompt/generation throughput và resource series dưới
   `release-evidence/<revision>/`.
6. RAM phải là aggregate của Python service và descendant native process. VRAM
   ưu tiên NVIDIA per-process; trên Windows WDDM dùng performance counter
   `GPU Process Memory(*)\\Dedicated Usage` và lọc đúng PID tree. Chỉ khi cả hai
   nguồn process-scoped không khả dụng mới dùng `total_system_fallback`; evidence
   phải ghi backend/lý do và không được giả là số per-process.
7. `initial_startup_time_ms` và `restart_time_ms` là hai phép đo riêng;
   `restart_time_ms` tính từ lúc crash được xác nhận đến lúc generation mới ready.
8. Confirm loopback default, authenticated external-mode rejection tests,
   private reasoning defaults, immutable artifacts, quota/retention, graceful
   drain, and fresh context.
9. A skipped GPU/native/fault/restart/soak/resource gate is not a release pass.

`summary.json` chỉ được ghi `consolidated_release_gate=passed` khi mọi artifact
cùng revision, manifest digest, runtime build, model digest và native binary.

The `controlled_inference/` tree is historical experiment material only and is not imported or packaged by `model_worker`.
