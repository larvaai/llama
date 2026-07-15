# Model Worker v1 release gate

All evidence must be generated from one Git revision and one verified manifest digest.

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
