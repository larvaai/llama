# M0 implementation handoff — Model Worker v1 hardening

Ngày tạo: 2026-07-15.

> Historical closure (2026-07-16): các gap và external gate mô tả dưới đây đã
> được đóng. M0 được release-attest từ revision sạch
> `b38b6df32755c55e668ac11e6c8f3e8b1c2ad46b`; consolidated summary nằm tại
> `release-evidence/b38b6df32755c55e668ac11e6c8f3e8b1c2ad46b/summary.json`. Mọi chữ
> “pending/chờ” còn lại trong file này mô tả checkpoint lịch sử 2026-07-15,
> không phải trạng thái hiện hành. Xem `docs/09-inference-runtime-implementation-handoff.md`.

Trạng thái: **historical artifact của checkpoint bắt đầu M0; đã được thay thế
bởi `09-inference-runtime-implementation-handoff.md`**.

Không dùng danh sách “việc tiếp theo” trong file này để điều khiển session mới.
Giữ file để audit các finding M0 và quyết định đã dẫn tới implementation hiện
tại.

Mục tiêu duy nhất của session mới là đóng M0 trong
`07-inference-runtime-and-agent-roadmap.md`. Không bắt đầu batching, KV reuse,
prefix cache, tool calling hoặc agent runtime trong cùng milestone.

## 1. Đọc theo thứ tự

1. `PROJECT.md`
2. `docs/06-model-worker-v1-plan.md`
3. `docs/07-inference-runtime-and-agent-roadmap.md`
4. Tài liệu này
5. `docs/model-worker-runbook.md`
6. `docs/model-worker-release.md`

`docs/03` và `docs/05` là lịch sử. `controlled_inference/` là archive,
không được production package import.

## 2. Working tree và baseline

Tại thời điểm tạo handoff, working tree đang có thay đổi do người dùng sở hữu:

~~~text
M CMakeLists.txt
M native/tests/test_reasoning_phase_controller.cpp
M pyproject.toml
M scripts/probe_worker.py
M scripts/release_gate.ps1
M scripts/soak_worker.py
M tests/gpu/test_model_release.py
M tests/integration/test_dispatcher_faults.py
M tests/integration/test_http_api.py
M tests/property/test_contract_equivalence.py
M tests/unit/test_contracts.py
M tests/unit/test_manifest_context.py
M tests/unit/test_output_contract.py
?? scripts/evaluate_bonsai_quality.py
?? tests/unit/test_request_registry.py
?? tests/unit/test_runtime_support.py
?? tests/unit/test_worker_process.py
~~~

Session mới phải đọc diff trước khi sửa và không reset/checkout bỏ các thay đổi
này.

Baseline đã chạy trên working tree trước khi thêm docs:

- `python -m pytest tests/unit tests/property tests/integration -m "not gpu" -q`
  → 107 passed.
- `ctest --test-dir build-v1 -C Release --output-on-failure`
  → 1/1 passed.
- `python -m ruff check model_worker tests scripts`
  → pass.

Chưa có evidence GPU/soak mới cho working tree hiện tại. Evidence
`release-evidence/57ee5e8/summary.json` thuộc revision cũ.

### Checkpoint triển khai 2026-07-15

M0 đã được triển khai và kiểm tra chức năng trên working tree hiện tại, nhưng
**chưa được ký release** vì tree chưa phải revision sạch và chưa chạy soak 500
request theo `scripts/release_gate.ps1`.

Các bằng chứng tạm thời đã chạy trực tiếp trên source/binary hiện tại:

- Ruff toàn bộ `model_worker`, `tests`, `scripts`: pass.
- Full non-GPU unit/property/integration: 180 passed.
- Native build lại từ source + CTest: 1/1 passed.
- Real native GPU fault suite: 2 passed; bao gồm cancel prompt/reasoning/final,
  stale attempt, crash, supervisor nhận DEGRADED và reload generation sạch.
- GPU model release suite: 3 passed ở checkpoint trước; phải chạy lại trong
  consolidated gate trên revision ký.
- Soak tạm thời 8 request, đủ bốn case qua hai vòng: 0 failure; một process
  generation; reasoning/final/total budget là 1024/64/1088.

Những thay đổi chính đã có:

- Lifecycle CAS cho queue expiry/dequeue/cancel; queue vật lý bounded và xóa
  item cancel/expired thay vì tích lũy tombstone.
- Execution timeout terminal trong deadline + watchdog grace; restart chạy
  ngoài đường terminalization.
- Supervisor readiness/start/restart/shutdown có bound; ready frame kiểm exact
  model/sequence; process cũ được reap trước generation mới.
- IPC request-frame FSM fail closed, correlation/sequence validation, process
  heartbeat và request progress được theo dõi riêng.
- Pending native cancel có TTL và capacity hữu hạn.
- Prompt instruction có owner/version/hash, được tính vào resource envelope và
  ghi vào artifact.
- `reasoning.require_start=true` là capability bắt buộc ở Python và native.
- UTF-8/strict JSON/grammar termination path có native/Python regression tests.
- Release tooling đo initial startup và crash→recovery riêng, xác minh đúng
  native child/hash/generation; resource monitor đo service process tree và
  ghi rõ VRAM per-process hoặc system fallback.

Điều kiện còn lại để đánh dấu M0 complete:

1. Chốt revision sạch mà không làm mất các thay đổi người dùng đang sở hữu.
2. Chạy consolidated release gate trên chính revision đó với GPU, fault,
   restart-recovery, 500-request soak và resource series không skip.
3. Chỉ khi `summary.json` ghi `consolidated_release_gate=passed` mới đổi trạng
   thái M0 trong `docs/07` thành complete.

## 3. Scope M0

M0 phải đóng các nhóm invariant:

1. Deadline và lifecycle đúng thời gian thực.
2. Cancellation không mất ở bất kỳ race window nào.
3. Supervisor/readiness phản ánh đúng native process/model state.
4. IPC corruption/crash không đầu độc request kế tiếp.
5. Startup, restart và shutdown đều có hard bound.
6. Prompt/output protocol khớp contract đã công bố.
7. Native grammar/UTF-8 handling fail closed.
8. Release evidence chạy đúng binary/revision/manifest.

Ngoài scope:

- SSE delta/event bus hoàn chỉnh thuộc M1, trừ phần tối thiểu cần quan sát
  progress/heartbeat cho watchdog.
- Metrics production/histogram thuộc M1.
- Multi-sequence, batching, priority và cache thuộc M2–M6.
- Tool/agent work thuộc H0–H3.

## 4. Findings cần reproduce và đóng

### M0-F1 — Queue timeout bị trễ

Hiện `model_worker/dispatcher.py` chỉ kiểm tra `queue_deadline` sau khi item
được pop. Request có thể vẫn ở `QUEUED` sau deadline nếu request trước đang
chạy. HTTP wait có thể hết hạn khi record chưa terminal.

Yêu cầu:

- Expiry xảy ra theo absolute deadline, độc lập với head request.
- Record terminal đúng một lần.
- Expired/cancelled queued item không giữ capacity vô hạn.
- HTTP không tạo response từ record non-terminal.

Test bắt buộc:

- Một active request treo, request thứ hai timeout đúng bound khi vẫn queued.
- Queue slot được thu hồi sau timeout.
- Race cancel-vs-timeout chỉ có một terminal transition/metric.

### M0-F2 — Cancellation có race trước native active

Native chỉ áp cancel khi request/attempt khớp active ID; lúc activate job lại
reset cancel flag. Cancel đến sau Python chuyển RUNNING nhưng trước native
activate có thể bị mất.

Yêu cầu:

- Cancel được key bởi request ID + attempt ID và được nhớ đến khi job activate.
- Python kiểm tra cancel trước send, sau send và khi đọc frame.
- Cancel queued/prompt/reasoning/final đều có bound.
- Sequence/frame của attempt cũ không cancel attempt mới.

Test bắt buộc:

- Cancel trước native send.
- Cancel sau send nhưng trước started.
- Cancel trong prompt chunk, reasoning và final.
- Cancel/restart race với stale attempt ID.

### M0-F3 — Readiness và supervisor chưa là source of truth

`ModelWorkerHTTPServer.ready` hiện là bool tĩnh. Native worker chỉ start lazily
khi execute; startup/restart đọc ready frame bằng blocking readline không có
hard timeout.

Yêu cầu:

- Supervisor state tối thiểu:
  `STARTING | READY | DEGRADED | RESTARTING | DRAINING | STOPPED`.
- Startup phải load/verify model trước khi `/ready` trả 200.
- `/ready` trả 503 trong crash/restart/failure.
- Startup/restart có timeout + terminate/kill fallback.
- Restart failure được giữ degraded; không tự báo ready.

Test bắt buộc:

- Fake worker không phát ready.
- Crash trước ready.
- Invalid ready frame.
- Restart success/failure và readiness transition.
- Startup không thể block service vô hạn.

### M0-F4 — Watchdog và IPC corruption

Native phát heartbeat/progress nhưng Python bỏ qua. IPC verifier error hiện có
thể fail request mà không kill/restart process; frame dư có thể đầu độc request
kế tiếp. Watchdog restart synchronous có thể tự treo trong startup.

Yêu cầu:

- Tách process heartbeat, request progress và absolute execution deadline.
- Absolute deadline luôn thắng heartbeat.
- Invalid JSON/version/ID/attempt/sequence/duplicate terminal là process
  protocol failure.
- Protocol failure terminalize request rồi kill/restart generation trước khi
  nhận request mới.
- Không cho thread request cũ đọc stdout của process generation mới.
- Restart có hard bound và return state được xử lý.

Test bắt buộc:

- Wrong request/attempt ID.
- Sequence gap/duplicate/out-of-order.
- Malformed JSON và EOF.
- Duplicate/missing terminal frame.
- Request sau corruption chạy thành công trên generation mới.

### M0-F5 — Graceful shutdown chưa có protocol đầy đủ

Backend hiện terminate process trực tiếp.

Yêu cầu:

- Ngừng admission.
- Terminalize/cancel queued và active theo policy.
- Gửi IPC shutdown, chờ grace.
- Terminate rồi kill theo hard deadline.
- Join dispatcher/reader thread có bound.
- Không để record hoặc artifact active sau shutdown.

Test bắt buộc:

- Shutdown idle.
- Shutdown khi queue full.
- Shutdown active responsive/unresponsive.
- Repeated shutdown idempotent.

### M0-F6 — Contract instructions chưa vào prompt

`output_contract.instructions` được parse/cap và có helper, nhưng native chat
template chỉ dùng request messages. GPU tests đang tự đưa semantic vào system
message.

Yêu cầu:

- Xác định một owner duy nhất cho deterministic prompt construction.
- Instructions đã cap được đưa vào prompt đúng một lần, có delimiter rõ và
  không hardcode schema cũ.
- Prompt artifact/hash cho phép điều tra version.
- Unsupported/oversized instruction fail trước queue.

Test bắt buộc:

- Semantic instruction thay đổi model-visible prompt.
- Empty instruction không thêm noise.
- Prompt builder không duplicate instruction.
- GPU probe chứng minh field semantic không cần test tự nhét đáp án.

### M0-F7 — Reasoning capability không khớp runtime

Manifest parser cho phép `reasoning.mode=none`, native vẫn luôn đọc marker và
tạo marker FSM.

Quyết định M0:

- Hoặc implement `none` end-to-end với final grammar từ token đầu,
- Hoặc reject `none` ở manifest version hiện tại.

Không được tiếp tục quảng cáo capability mà native không hỗ trợ.

Test bắt buộc:

- Capability/manifest/native conformance cho từng mode được công bố.
- Missing/multi-token marker path vẫn fail closed.

### M0-F8 — Grammar acceptance và UTF-8 chưa được chứng minh

Native đang truyền một bool suy ra từ phase thay vì query/prove grammar accepting
state. UTF-8 checker chưa chứng minh reject mọi non-scalar/overlong/surrogate
case. GBNF/strict JSON path cũng cần unpaired-surrogate regression.

Yêu cầu:

- EOG chỉ hoàn tất khi grammar thực sự accepting.
- Tách UTF-8 accumulator và termination mapping thành module test được.
- Reject invalid UTF-8, overlong, surrogate và trailing/incomplete bytes.
- HTTP/artifact serialization không crash trên malicious escaped Unicode.

Test bắt buộc:

- Grammar non-accepting + EOG.
- Valid multi-byte token pieces qua nhiều delta.
- Invalid continuation/overlong/surrogate/out-of-range/incomplete sequence.
- `\uD800` và duplicate/trailing JSON regression.

### M0-F9 — Release gate có thể dùng stale service

Release script phải chứng minh GPU/probe/soak gọi đúng binary vừa build và đúng
manifest/runtime digest.

Yêu cầu:

- Gate tự launch service từ binary vừa build hoặc verify process identity.
- Response/evidence phải khớp revision, manifest digest, runtime build, model
  digest và process generation.
- Không skip GPU/native/soak mà vẫn ký pass.
- Evidence chứa latency, prompt/decode throughput, peak/stable RAM/VRAM,
  restart time và failure counts như release doc yêu cầu.

## 5. Thứ tự implementation

Không sửa tất cả trong một patch lớn.

1. **M0.1 Registry/deadline**
   - Absolute expiry mechanism.
   - Terminal-only HTTP response.
   - Queue capacity reclamation.
2. **M0.2 Supervisor/readiness**
   - State machine.
   - Bounded start/restart.
   - Health endpoint source of truth.
3. **M0.3 IPC/cancellation**
   - Pending cancel semantics.
   - Process-generation-safe reader.
   - Corruption recovery.
4. **M0.4 Shutdown**
   - Drain/cancel/shutdown/terminate/kill sequence.
5. **M0.5 Protocol correctness**
   - Prompt instructions.
   - Reasoning mode decision.
   - Grammar acceptance.
   - UTF-8/Unicode.
6. **M0.6 Real native fault suite**
   - Không chỉ fake worker.
7. **M0.7 Release evidence**
   - Clean revision + exact service launch + GPU + soak.

Sau mỗi slice:

- Chạy unit/property/integration liên quan.
- Chạy full non-GPU suite.
- Không đổi public contract nếu chưa version hóa.
- Cập nhật handoff nếu finding hoặc dependency thay đổi.

## 6. Test architecture M0

Tối thiểu cần:

- Unit Python: registry expiry, supervisor FSM, error mapping, prompt builder.
- Native unit: UTF-8 accumulator, phase/grammar termination, mode-none nếu hỗ
  trợ.
- Property tests: lifecycle races, frame ordering, Unicode/JSON edge cases.
- Fake-process integration: startup/restart/shutdown/corruption hard bounds.
- Real native integration: cancel at prompt/reasoning/final, malformed control,
  crash/recovery.
- GPU: semantic instructions, protocol boundary, cancellation và post-crash
  request.
- Soak: sequential stability, injected crash/restart, cancel/timeout mix và
  resource slope.

M0 không được coi fake-worker pass là bằng chứng native cancel/readiness.

## 7. Commands

~~~powershell
cd D:\zalollm\agent-harness-lab

git status --short
git diff --stat

python -m ruff check model_worker tests scripts
python -m pytest tests/unit tests/property -q
python -m pytest tests/integration -m "not gpu" -q

cmake -S . -B build -DBUILD_TESTING=ON
cmake --build build --config Release --target model-worker-native model-worker-native-tests
ctest --test-dir build -C Release --output-on-failure

scripts/build_native_runtime.ps1 -ModelManifest config/model.local.json -BuildDirectory build-runtime
scripts/release_gate.ps1 -ModelManifest config/model.local.json
~~~

Không chạy consolidated release gate cho đến khi unit/fault slices đã pass.

## 8. Evidence artifact bắt buộc

Release output:

~~~text
release-evidence/<revision>/
  summary.json
  manifest.json
  build.json
  unit-property.json
  fake-worker-integration.json
  native-integration.json
  gpu.json
  fault-injection.json
  soak.json
  resource-series.json
~~~

`summary.json` chỉ được ghi `consolidated_release_gate=passed` khi mọi artifact
cùng revision/manifest/runtime và không có required gate bị skip.

## 9. M0 Definition of Done

- Queue timeout xảy ra theo wall clock và không trả response non-terminal.
- Cancel không mất trong các race window đã liệt kê.
- Startup/restart/shutdown đều có hard bound.
- Readiness phản ánh model/process state thật.
- IPC corruption làm restart sạch; request kế không bị poisoned.
- Instructions thực sự đi vào deterministic prompt.
- Reasoning modes công bố khớp native capability.
- Grammar/EOG/UTF-8 fail closed và có native tests.
- Full non-GPU, native, real GPU, fault và soak pass cùng revision.
- Evidence chứa đủ identity, latency, throughput và resource series.
- Không có regression về private reasoning, fresh context, auth/resource caps
  hoặc semantic trust boundary.

Khi M0 pass, cập nhật `docs/07`, đánh dấu M0 complete và chuyển session kế tiếp
sang M1. Không bắt đầu M2/H0 trong cùng session nếu release evidence chưa ký.

## 10. Trạng thái implementation ngày 2026-07-15

Các slice M0 đã được triển khai theo đúng thứ tự và qua gate cục bộ:

| Slice | Trạng thái | Evidence gần nhất |
|---|---|---|
| M0.1 Registry/deadline | Đã verify | Absolute queue expiry, terminal-only HTTP, capacity reclamation và cancel/timeout race tests pass. |
| M0.2 Supervisor/readiness | Đã verify | Supervisor FSM, bounded startup/restart và `/ready` theo process state pass. |
| M0.3 IPC/cancellation | Đã verify | Pending cancel theo request+attempt, generation-scoped reader, malformed/identity/sequence/terminal corruption recovery pass. |
| M0.4 Shutdown | Đã verify | Admission stop, terminalization, IPC shutdown, terminate/kill fallback và idempotent bounded joins pass. |
| M0.5 Protocol correctness | Đã verify | Prompt instruction/hash, reject `reasoning.mode=none`, grammar EOG proof, UTF-8 scalar và unpaired-surrogate tests pass. |
| M0.6 Real native fault suite | Đã verify | Verified-runtime worker pass cancel ở prompt/reasoning/final, malformed/stale control, post-fault request và real process kill/reload. |
| M0.7 Release evidence | Chờ clean revision | Gate tự build/launch exact service, verify identity, inject/recover exact native child, bắt buộc GPU/fault/soak/resource artifacts và từ chối dirty tree. |

Evidence cục bộ mới nhất trên working tree chưa commit:

- Full non-GPU: `221 passed, 2 skipped`; hai skip là symlink privilege trên
  Windows, không phải GPU/native/fault/soak gate.
- Coverage unit/property/integration: `92%` branch-aware.
- Native CTest: `1/1 passed`.
- Real native fault suite: `1 passed` trong 24.59 giây.
- GPU model suite: `3 passed` trong 18.05 giây trên exact service.
- Soak corrected: `500/500`, zero failure, process generation duy nhất `1`,
  p50 `1.969s`, p95 `4.609s`, max `13.469s`, prompt throughput
  `1325.14 tok/s`, generation throughput `76.12 tok/s`.
- Working-tree soak/resource artifacts nằm ở
  `artifacts/m0-working-tree-evidence/`; chúng chỉ là diagnostic vì identity
  là `dirty-working-tree`, không được ký làm release evidence.

Finding mới khi chạy soak lần đầu: workload cố định reasoning budget ở 256
token nên case tiếng Việt deterministically fail `reasoning_budget_exhausted`
125/500 lần. Soak runner hiện derive reasoning/final/total budgets từ manifest
và smoke 4/4 + full 500/500 đã pass. Đây là sửa workload contract, không nới
worker invariant.

### External gate còn lại

Không được đánh dấu M0 complete trên working tree hiện tại. Cần:

1. Phân loại/tách các thay đổi ngoài M0 đang cùng xuất hiện trong working tree,
   đặc biệt `inference_runtime/` và test liên quan; session M0 không được nhận
   chúng là M2 implementation và không được tự xóa thay đổi do người dùng sở hữu.
2. Tạo một clean Git revision chứa đúng baseline người dùng + M0 đã duyệt.
3. Chạy `scripts/release_gate.ps1 -ModelManifest config/model.local.json` trên
   revision sạch đó. Gate phải sinh đủ artifact cùng revision/manifest/model/
   runtime/native hash và `summary.json` mới được ghi
   `consolidated_release_gate=passed`.
4. Chỉ sau bước 3 mới cập nhật `docs/07` thành M0 complete và handoff sang M1.
