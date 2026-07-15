# Implementation checkpoint và handoff còn lại

Ngày chốt engineering checkpoint: 2026-07-15. Cập nhật release: 2026-07-16.

Trạng thái: **M0 đã release-attest từ clean revision; M1–M7 và H0 đạt
engineering gate; M7 real-provider matrix vẫn là deployment gate; H1 atomic
read-only vertical slice đã pass engineering checkpoint nhưng full H1 chưa được
tuyên bố**.

Runtime release attestation có thẩm quyền là revision
`b38b6df32755c55e668ac11e6c8f3e8b1c2ad46b` cùng evidence tại
`release-evidence/b38b6df32755c55e668ac11e6c8f3e8b1c2ad46b/`. Commit cập nhật docs sau
gate là documentation-only; không đổi identity đã ký. Không tái sử dụng
attestation nếu model manifest, runtime build, model digest hoặc native binary
thay đổi.

Semantics chuẩn của tool calling nằm ở section 5 và roadmap ở section 7 của
`07-inference-runtime-and-agent-roadmap.md`. File này ghi checkpoint code/evidence
hiện tại, phần H1 đã hoàn thành và các gate còn lại để session mới không tái
implement hoặc overclaim.

## 1. Bắt đầu session

Đọc theo thứ tự:

1. `PROJECT.md`
2. `docs/07-inference-runtime-and-agent-roadmap.md`
3. File này
4. `artifacts/inference-runtime/2026-07-15-m0-m7-final-verification.json`

Sau đó chạy:

~~~powershell
Set-Location D:\zalollm\agent-harness-lab
git status --short
git diff --stat
python -m pytest -m "not gpu and not soak" -q
python -m ruff check .
~~~

Kỳ vọng handoff là working tree sạch sau commit tài liệu. Nếu có thay đổi mới,
không reset, checkout, clean, reformat hàng loạt hoặc commit/push nếu chưa được
yêu cầu rõ.

## 2. Checkpoint identity và evidence

~~~text
Attested runtime revision:   b38b6df32755c55e668ac11e6c8f3e8b1c2ad46b
Gate working tree:           clean
Consolidated gate:           passed
GPU:                         NVIDIA GeForce RTX 3080, 10240 MiB
Driver:                      591.86
Model:                       qwen35-9b-local, Q4_K_M
llama.cpp build:             b10012
Manifest SHA-256:             afb03af1a03f070be8ba51f076f286065ae3574e317eec7dd9587bb303a1ae2f
Model SHA-256:                b68fbb8167d4e0a39c8157d87ea880a38d6c593c2d7b92c153212496f635eb46
Native binary SHA-256:        d2dbb0f4a192e66e6c4d3381dcc9a03d30d5aa7dbec099c08ab03eb78ab6fd09
Release summary SHA-256:      dabead013b42a5f901bbe5ac5ea7a481a9a6eb4812331886688ff5426c9c9bf5
Inference runtime SHA-256:   9e4da794cf2a3cacec47a0cf0b6b4fb60e06e0ff42751ed745562f47c272750f
Model worker SHA-256:        048da4064ab263f1460804c69f3630eec08df41ba90b14f6af85bc60ad1a5fe6
~~~

Machine-readable index:

`artifacts/inference-runtime/2026-07-16-handoff-index.json`

M0 release attestation:

`artifacts/inference-runtime/2026-07-16-m0-release-attestation.json`

Requirement traceability:

`artifacts/inference-runtime/2026-07-15-m0-m7-requirement-audit.json`

Evidence chính:

- M3: `2026-07-15-m3-soak-final-stable-shape.json` — 1.228 request/901,532
  giây, 0 lỗi, RSS drift 1.024.000 byte, process VRAM drift 0 MiB và shutdown
  sạch.
- M4: `2026-07-15-m4-benchmark-final.json` — `2,274732×` serial,
  `1,122704×` llama-server request throughput, token-level ITL pass.
- M5: `2026-07-15-m5-priority-gate-aggregate.json` — 3/3 paired run pass;
  heterogeneous range `0,874348–0,936995×` vẫn là order-sensitive backlog.
- M6: `2026-07-15-m6-cache-gate.json` — scoped prefix/session/COW gate pass.
- M6 performance: `2026-07-15-m6-cache-performance.json` — hit rate `80%`,
  saved prefill `2.560` token, TTFT p50 ratio `0,111332`, semantic equality và
  cache bytes về 0 sau clear; dedicated VRAM WDDM process-scoped delta `4 MiB`.
- M7: `2026-07-15-m7-backend-conformance.json` — portable pass; real provider
  matrix unavailable trên host này.
- H0: `2026-07-15-h0-verification.json` — durable contracts/state pass; không
  có side-effect executor.
- H1: `2026-07-15-h1-atomic-readonly-verification.json` — atomic read-only
  vertical slice pass; full H1 không được claim.

Verification cuối của checkpoint:

~~~text
Ruff:                         All checks passed
Non-GPU regression:           417 passed, 3 skipped, 10 deselected (17.93s)
Native source build:          model worker + inference runtime + tests built
CTest:                        1/1 passed
Native fault GPU slice:       2 passed in 40.92s
Model semantic HTTP slice:    3 passed in 18.14s
Inference sequence GPU suite: 5 passed in 86.34s
Clean consolidated M0 gate:   passed at b38b6df32755c55e668ac11e6c8f3e8b1c2ad46b
Release soak:                 500/500 passed, 0 failure, generation [2]
Release latency p50/p95/max:  1.937/3.578/4.359 seconds
~~~

Ba skip là Windows symlink privilege (một H1 scope-escape case và hai artifact
cleanup cases). Các slice pre-release vẫn là engineering evidence; clean gate
ngày 2026-07-16 là attestation riêng và đã pass đủ required gates.

Completion audit đã đóng các vấn đề evidence sau:

- Continuous scheduler từng có lost-wakeup giữa action check và idle wait.
  Re-check dưới cùng condition lock đã sửa race; scheduler/governance focused
  suite 16/16 pass và reproduction cũ pass 100/100 process độc lập.
- Release static audit từng bắt literal semantic-reserved `accepted` trong
  event-buffer outcome. Outcome nội bộ đã đổi thành `ENQUEUED/enqueued`; 44
  event/HTTP/fault tests và full regression pass, forbidden production scan
  hiện sạch.
- Một soak đầu tiên đổi prompt shape muộn từ `soak-999` sang `soak-1000`, làm
  CUDA arena mở rộng sau warmup và khiến total-system fallback fail. Artifact
  fail được giữ để chẩn đoán. Soak có thẩm quyền dùng fixed slot shapes warm từ
  wave đầu và WDDM process counter, đạt 1.228 request với dedicated VRAM drift
  0 MiB.
- Ba lần chạy clean gate đầu đã lộ lỗi ở chính release harness: literal
  `C:\Users` chưa escape trong regex, coverage threshold bị enforce trước khi
  append integration, và Windows PowerShell 5 coi native stderr warning là
  terminating error. Các lỗi này lần lượt được sửa ở `aeac217`, `4f51232` và
  `b38b6df`; gate cuối trên `b38b6df` chạy hết 1.332,4 giây và pass.

## 3. Trạng thái milestone

| Mốc | Trạng thái hiện hành | Giới hạn còn lại |
|---|---|---|
| M0 | **Released** tại `b38b6df32755c55e668ac11e6c8f3e8b1c2ad46b` | Gate lại khi model/runtime/native identity đổi |
| M1 | Engineering pass trong cùng attested revision | Không |
| M2 | Engineering pass | Không |
| M3 | Engineering pass | Không |
| M4 | Engineering pass | Tiếp tục profile workload mới |
| M5 | Portable priority gate pass | Tối ưu heterogeneous-prompt order sensitivity |
| M6 | Engineering pass | Không |
| M7 | Portable architecture pass | Real MLX-LM/vLLM/SGLang là deployment matrix theo môi trường |
| H0 | Engineering pass | Concrete executor thuộc H1, không phải H0 |
| H1 | Atomic read-only vertical slice engineering pass | Chờ real-local-model acceptance artifact; mutation/approval execution vẫn khóa |
| H2/H3 | Chưa bắt đầu | Không ghép vào H1 |

## 4. Nền H0 đã có

Package production:

~~~text
agent_runtime/
  __init__.py
  catalog.py
  contracts.py
  errors.py
  event_store.py
  events.py
  ids.py
  ports.py
  reducer.py
~~~

H0 đã cung cấp:

- `ToolIntent` với đúng bốn action `call_tool | ask_user | submit | blocked`
  và strict five-field/cross-field validation.
- `CompiledToolCall`, `ToolResultEnvelope`, `FlowTransition`, effect/retry/flow
  enums và correlation IDs.
- Immutable `ToolDefinition`/`ToolRevision` và canonical catalog digest.
- `SqliteAgentEventStore`: WAL, durable `synchronous=FULL`, optimistic
  concurrency, unique event/action/invocation claims.
- Deterministic reducer cho AgentRun/ToolInvocation/budget lifecycle.
- Ports cho compiler, permission, executor, normalizer, flow và acceptance.

Không thay event store/transcript/KV bằng in-memory orchestration. Durable event
stream vẫn là source of truth.

Concrete H1 implementation hiện nằm trên nền H0 này; mô tả “không có concrete
executor” chỉ đúng với artifact H0 độc lập, không còn đúng với toàn package
`agent_runtime/` hiện tại.

## 5. Mục tiêu và checkpoint H1

Hoàn thành một atomic task qua đúng một tool action mỗi model turn:

~~~text
bounded context
→ InferencePort tạo ToolIntent
→ strict parse + cross-field validation
→ deterministic compile
→ permission/approval
→ durable claim
→ bounded executor
→ normalize/redact/artifact
→ durable terminal result
→ code-owned FlowPolicy
→ synthesize-no-tools hoặc redecide
→ deterministic acceptance
~~~

H1 không xây Task Splitter, DAG, reviewer workflow hoặc parallel agent. H1
không đưa tool executor vào `model_worker/` hay native runtime và không giữ GPU
sequence trong lúc tool chạy.

Read-only vertical slice hiện đã đi hết pipeline trên bằng fake inference và
hai concrete adapter allowlisted `read_file`/`search_text`. Nó chứng minh
one-action-per-turn, inference release-before-tool, durable claim,
normalization/redaction, code-owned flow, recovery và deterministic acceptance.
Không có raw shell, arbitrary model path hoặc trusted raw tool output.

## 6. H1 implementation plan và trạng thái

H1.0–H1.8 dưới đây đã được implement cho read-only slice. Các yêu cầu mutation
được giữ làm fail-closed contract/test boundary; mutation execution và approval
token dispatch chưa được bật. Full H1 vẫn chờ test matrix item 14 bằng model
local thật.

### H1.0 — Version contract/event trước side effect

H0 event schema chỉ biết lifecycle/claim cơ bản. Trước khi dispatch tool thật:

- Thêm versioned durable records cho decision, compiled args digest,
  authorization/approval, dispatch boundary, result artifact, flow decision và
  acceptance verdict.
- Reader/reducer phải replay được H0 v1 stream hoặc fail với migration error rõ;
  không âm thầm reinterpret persisted event.
- `AgentError` phải mang đủ `retryable`, `retry_scope` và
  `side_effect_state`; result/dispatch metadata phải phân biệt pre-dispatch với
  post-dispatch unknown.
- Persist secret-free canonical payload. Native args chứa secret chỉ được lưu
  dưới redacted digest/ref theo policy.

Exit: crash/reopen tại mọi event boundary cho cùng state; duplicate
`action_id`/`invocation_id`/idempotency claim có đúng một winner.

### H1.1 — Decision context và inference boundary

Tạo bounded `DecisionContextBuilder` và adapter high-level tới
`InferencePort.infer()`:

- Context chỉ gồm instruction version, task objective/acceptance, bounded state
  refs, 3–8 semantic tool cards, latest normalized observation và ToolIntent
  schema.
- Schema compile riêng mỗi turn; `tool_id.enum` đúng shortlist đã lọc cộng
  `null`.
- Ghi `tool_catalog_digest`, prompt/context digest, inference request/attempt
  ID và usage/budget event.
- Inference phải terminal và release trước compiler/executor.
- Invalid JSON/schema/cross-field/unknown tool fail closed; không repair bằng
  regex và không execute partial stream fragments.

Exit: fake InferencePort tests cho bốn action, malformed output, cancellation,
deadline, catalog race và sequence release-before-tool.

### H1.2 — Registry shortlist và deterministic compiler

Mỗi tool revision giữ hai schema: semantic agent-facing và native. Compiler:

- Bind value/ref/default/alias từ allowlisted task state.
- Canonicalize native args, validate schema và scope trước authorization.
- Tạo stable `action_id`, `internal_call_id`, `invocation_id` và
  `idempotency_key` theo documented inputs; replay sinh cùng claim.
- Không suy diễn secret, permission, arbitrary path hoặc raw command.
- Nếu không bind được, trả typed `needs_resolution`; không dispatch.

V1 vertical slice giới hạn adapter nhỏ:

- `read_file`: path phải là artifact/task ref đã allowlist.
- `search_text`: root/ref và pattern bounded; không nhận raw shell.
- `run_test`: chưa bật trong read-only slice hiện tại; khi thêm chỉ được nhận
  allowlisted test target/profile, không nhận command line tùy ý.

Exit: differential schema tests, traversal/symlink/scope escape tests và
canonical ID/idempotency property tests pass.

### H1.3 — ArgumentResolver fallback

Chỉ gọi resolver khi deterministic compiler trả thiếu semantic binding:

- Resolver chỉ thấy một tool đã chọn và semantic schema nhỏ.
- Resolver không được đổi `tool_id`, effect class, permission, scope hoặc
  execute.
- Output qua grammar/strict parser/native compiler như dữ liệu model khác.
- Có resolver-call budget; exhausted chuyển `ask_user` hoặc blocked policy.

Exit: tool substitution, schema escape, invented secret/path và resolver loop
đều bị chặn trước executor.

### H1.4 — Permission, approval và taint gate

Authorization trả structured decision, không chỉ boolean:

~~~text
ALLOW | DENY | REQUIRE_APPROVAL
~~~

Gate kiểm tra role, tenant, task scope, tool revision, effect class, taint,
deadline, retry state và approval token. Mutation bắt buộc persist audit trước
dispatch; approval token one-shot và bind vào args digest/effect/scope.

Exit: denied/expired/wrong-scope approval không gọi executor; TOCTOU catalog,
task version hoặc args digest mismatch phải redecide.

### H1.5 — Bounded executor và mutation safety

Executor chạy ngoài inference/native process với:

- Per-tool timeout, cancel token, stdout/stderr/result byte cap và concurrency
  limit.
- Explicit dispatch boundary event ngay trước side effect.
- Read-only/transient retry theo policy.
- Idempotent mutation retry cùng key; executor/dedup store trả cùng outcome.
- Non-idempotent timeout/crash sau dispatch thành `UNKNOWN_OUTCOME`; chỉ
  probe/reconcile/human decision, không replay mù.
- Shutdown terminalize hoặc reconcile mọi in-flight invocation.

Không tạo generic raw-shell tool trong H1. `run_test` dùng allowlisted command
template do code sở hữu.

Exit: timeout/cancel/crash trước và sau dispatch, duplicate delivery,
idempotent retry và unknown-outcome tests pass.

### H1.6 — Result normalization và artifact boundary

Raw tool output là untrusted:

- Validate native output schema, normalize deterministic, redact secret và cap
  byte/token.
- Lưu raw/big payload thành immutable artifact với hash/provenance; chỉ bounded
  `ToolResultEnvelope` đi vào LLM.
- Phân biệt `succeeded | partial | failed | unknown_outcome`, truncated và
  insufficient result.
- Persist terminal invocation event trước FlowController.
- Không nối raw tool output vào system prompt; renderer direct-return phải
  allowlisted và escape đúng channel.

Exit: invalid schema, non-UTF-8, oversized output, secret leakage, artifact
write failure và hash mismatch fail closed.

### H1.7 — FlowController, synthesis và acceptance

Flow policy do code chọn từ tool/task/role:

- `DIRECT_RETURN_SAFE`
- `SYNTHESIZE_NO_TOOLS`
- `REPLAN_WITH_OBSERVATION`
- `VERIFY_THEN_REPLAN`
- `PAUSE_FOR_APPROVAL`
- `PERSIST_AND_STOP`
- `HANDOFF`

Lượt synthesis không có tool catalog. `submit` chỉ gọi deterministic
AcceptanceGate; chỉ event có `authority=acceptance_gate` được chuyển run thành
`SUCCEEDED`. `blocked` phải kiểm tra retry/replan/ask-user path trước khi
terminalize.

Exit: model không thể tự succeed; synthesis không thể gọi tool; acceptance
fail tạo bounded redecide/blocked path và không loop vô hạn.

### H1.8 — AtomicAgent orchestration và recovery

Tạo service/orchestrator mỏng nối các port nhưng không nuốt invariant:

- Mỗi command load/replay snapshot và append bằng expected version.
- Mỗi model turn tối đa một ToolIntent action.
- Persist trước mutation; terminal result persist trước flow.
- Cancellation/deadline/budget được kiểm tra tại mọi boundary.
- Recovery scan phân loại safe retry, reconcile-only và terminal failure.
- Correlation đầy đủ:
  `workflow→task→run→turn→inference` và
  `run→action→invocation→tool-attempt`.

Exit: restart tại mọi boundary tạo cùng durable outcome, không double-dispatch
và không double-consume budget.

## 7. File layout hiện tại

Production package hiện giữ boundary sau:

~~~text
agent_runtime/
  decision.py             # context/schema + InferencePort adapter
  compiler.py             # deterministic semantic→native binding
  resolver.py             # optional one-tool fallback
  permissions.py          # allow/deny/approval + scope/taint
  executor.py             # bounded dispatch/retry/cancel
  normalization.py        # schema/redaction/artifact envelope
  flow.py                 # code-owned policy
  acceptance.py           # deterministic task gate
  service.py              # AtomicAgent orchestration/recovery
  tool_adapters/
    read_file.py
    search_text.py
tests/unit/agent_runtime_h1/
tests/integration/agent_runtime_h1/
tests/property/agent_runtime_h1/
~~~

Không import HTTP worker internals vào agent core. `run_test` vẫn là adapter
allowlisted tương lai, không phải capability đang advertise.

## 8. Test matrix bắt buộc

Tối thiểu phải có:

1. Bốn ToolIntent action và mọi cross-field invalid combination.
2. Unknown/non-shortlisted tool, stale catalog digest và catalog TOCTOU.
3. Deterministic compile success, missing/ambiguous binding và resolver budget.
4. Scope traversal/symlink escape, tainted arg, secret redaction.
5. Permission deny/approval required/expired/replayed/mismatched approval.
6. Executor unavailable, pre-dispatch timeout, post-dispatch unknown,
   cancellation, partial output và crash.
7. Idempotent duplicate delivery/retry; non-idempotent call không replay mù.
8. Result invalid/oversized/truncated/artifact failure/hash mismatch.
9. Flow policy mapping, synthesize-no-tools và direct renderer allowlist.
10. Submit không succeed nếu AcceptanceGate fail.
11. Budget/deadline exhausted tại từng boundary và exact-once reconciliation.
12. Crash/reopen tại mọi event boundary; random illegal event stream fail
    closed mà không đổi durable state.
13. End-to-end atomic read-only task trên fake inference/executor.
14. Một real local-model decision → allowlisted read/search/test tool →
    normalized observation → deterministic acceptance, ghi artifact riêng.

Property tests phải bao phủ replay determinism, ID stability, exactly-once
claims và bounded loop. Real GPU test không thay fake fault matrix.

Checkpoint hiện đã chạy item 1–13 trong phạm vi read-only/fail-closed mutation
boundary. Item 14 chưa chạy và là gate chính còn lại. Mutation side effects
không được xem là đã verify chỉ vì các nhánh deny/unknown-outcome đã có unit
test.

## 9. H1 exit gate và giới hạn checkpoint

H1 chỉ pass khi:

- One action/model turn được chứng minh bằng contract/property tests.
- Permission/scope/approval chặn trước side effect.
- Read-only và idempotent retry không double-dispatch; non-idempotent unknown
  outcome không replay.
- Raw result không lọt vào trusted/system context; artifact/ref/hash/redaction
  đầy đủ.
- Synthesis không có tools; model không tự xác nhận completion.
- Deterministic acceptance quyết định `SUCCEEDED`.
- Crash/replay tại mọi durable boundary giữ cùng outcome và budget.
- Fake fault suite, end-to-end atomic task, non-GPU regression và Ruff pass.
- Machine-readable H1 evidence ghi revision/dirty state/hash; dirty evidence
  vẫn chỉ là engineering checkpoint.

Read-only slice đáp ứng các invariant trên bằng 88 pass, 1 skip và 92% branch
coverage cho `agent_runtime`. Tuy nhiên full H1 vẫn **pending** cho đến khi có
real local-model decision → allowlisted tool → normalized observation →
deterministic acceptance artifact. Nếu mutation được bật, approval-token,
idempotent retry và non-idempotent unknown-outcome execution matrix cũng trở
thành gate bắt buộc.

## 10. Parallel track và ngoài scope

Track M0 release đã hoàn tất:

1. Revision sạch `b38b6df32755c55e668ac11e6c8f3e8b1c2ad46b` đã được chốt.
2. `scripts/release_gate.ps1` đã chạy từ đúng revision/manifest/runtime đó.
3. `summary.json` ghi `consolidated_release_gate=passed`; 500/500 soak, không
   failure và resource series quan sát đúng restarted native child.

Deployment M7 chạy khi có đúng môi trường:

- MLX-LM trên host/provider phù hợp.
- vLLM/SGLang server thật với model/config được pin.
- Không cài hoặc khởi chạy provider nặng chỉ để làm đẹp portable gate.

Ngoài scope H1:

- H2 Task Splitter/DAG/reviewer/replan.
- H3 parallel agent execution.
- Raw shell/browser/network tool tổng quát.
- Tự viết tokenizer, attention, sampler, GPU kernel hoặc KV allocator.
- Tuyên bố thay vLLM trên mọi workload/phần cứng.

Thứ tự cho session kế tiếp:

1. Không viết lại read-only slice; verify source hash/evidence hiện có.
2. Đóng real-local-model acceptance artifact để nâng read-only checkpoint thành
   full H1.
3. Chỉ mở mutation tools hoặc H2/H3 bằng một scope riêng và gate tương ứng.
4. Nếu đổi model/runtime/native identity, tạo release revision và attestation
   mới; không sửa hoặc tái gắn nhãn evidence của `b38b6df…`.
