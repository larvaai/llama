# Inference Runtime và Agent Harness — roadmap hiện hành

Ngày chốt kiến trúc: 2026-07-15. Cập nhật release: 2026-07-16.

Trạng thái: **kiến trúc và roadmap chính thức; M0 đã release-attest, M1–M7 đạt
engineering gate; implementation checkpoint và giới hạn còn lại được ghi riêng
trong `09-inference-runtime-implementation-handoff.md`**.

Tài liệu này thay thế phần “thứ tự triển khai sau này” trong
`03-progress-and-roadmap.md` và `05-controlled-inference-handoff.md`.
`06-model-worker-v1-plan.md` vẫn là baseline và release contract của
single-model worker. `08-m0-implementation-handoff.md` là artifact lịch sử lúc
bắt đầu M0; điểm vào cho session triển khai kế tiếp là
`09-inference-runtime-implementation-handoff.md`.

## 1. Mục tiêu

Xây một AI Operating System / multi-agent harness local-first có hai năng lực
độc lập nhưng phối hợp:

1. Agent harness chia yêu cầu thành task nhỏ, kiểm soát role, tool, state,
   acceptance, retry, reviewer và handoff.
2. Inference runtime điều phối nhiều inference sequence trên llama.cpp bằng
   admission control, prefill/decode scheduling, continuous batching,
   priority, cancellation và cache policy.

Mục tiêu không phải viết lại tokenizer, attention, sampler, GPU kernel hoặc KV
allocator. llama.cpp tiếp tục sở hữu compute. Project sở hữu scheduling policy,
resource governance, orchestration và observability.

Mục tiêu cũng không phải clone toàn bộ LM Studio, Ollama, vLLM hay SGLang.
Những backend đó có thể tồn tại dưới adapter khi workload hoặc phần cứng phù
hợp hơn.

## 2. Trạng thái nền hiện tại

Production path gồm ba package/tầng:

- `model_worker/`: HTTP/service control plane và serial worker tương thích v1.
- `inference_runtime/`: port, scheduler, governance, cache policy, registry và
  backend adapter.
- `native/`: serial worker và multi-sequence llama.cpp runtime.

`controlled_inference/` là archive của prototype A–G, không phải production
dependency.

Đã implement trong codebase hiện tại:

- M0 lifecycle/safety hardening: queue expiry/cancel race, bounded supervisor,
  readiness, IPC recovery, shutdown, prompt contract, reasoning capability và
  UTF-8/grammar handling.
- M1 bounded events/SSE backpressure, terminal metrics, registry/artifact
  maintenance.
- M2 `InferencePort`, `ManagedBackend`, `SteppableBackend`, capability contract
  và deterministic scheduler simulator.
- M3 persistent per-sequence context, opaque generation-scoped handle, native
  prefill/decode batch, explicit release và scheduler lifecycle.
- M4 continuous scheduler với queue prefill/decode, decode burst bound, chunked
  prefill, microbatch và token budget mỗi tick. Idle wait re-checks work under
  the condition lock, preventing an enqueue/notify lost-wakeup race.
- M5 hierarchical workflow→agent→request selection, service class, deadline,
  aging, emergency cap, quota, KV reservation ledger và load shedding.
- M6 scoped native exact-state checkpoints cho longest safe prefix reuse;
  immutable private session generations/COW; namespace digest, TTL/LRU, byte
  budget, clear/stats và cross-workflow/agent isolation.
- M7 lazy capability registry, load/unload/keepalive, high-level router và
  managed adapter cho vLLM/SGLang/MLX-LM. llama.cpp là steppable backend;
  managed backend không giả lập sequence API mà public provider không sở hữu.
- H0 `agent_runtime/`: versioned contracts, deterministic reducer, SQLite
  append-only event store, stable claims, catalog digest và replay/property
  tests.
- H1 atomic read-only vertical slice: bounded decision context, deterministic
  compiler, structured permission, durable claim, bounded allowlisted executor,
  normalization/redaction/artifact boundary, code-owned flow và deterministic
  acceptance cho `read_file`/`search_text`.

Trạng thái release và giới hạn còn lại:

- M0 đã pass clean consolidated release gate tại revision
  `b38b6df32755c55e668ac11e6c8f3e8b1c2ad46b`: unit/property/integration/native,
  real fault injection, crash → `DEGRADED` → request-triggered recovery, GPU và
  500/500 soak cùng verified identity; `failure_count=0`. Release summary SHA-256
  là `dabead013b42a5f901bbe5ac5ea7a481a9a6eb4812331886688ff5426c9c9bf5`.
- M3 long soak đã pass: 1.228 request trong 901,532 giây, không lỗi, RSS drift
  `1.024.000 byte`, dedicated VRAM process-scoped WDDM drift `0 MiB`, process
  generation ổn định và shutdown sạch.
  Run total-system fallback bị prompt-shape transition muộn làm fail được giữ
  làm diagnostic, không được dùng làm gate pass.
- M4 đã pass benchmark cuối: `2,274732×` serial, `1,122704×` llama-server theo
  request, `1,064309×` theo sampled token và có token-level ITL evidence.
- M5 portable scheduler-priority gate đã pass 3/3 paired run: median high-class
  TTFT cải thiện `98,721%`, median throughput ratio `1,018223`, minimum
  `1,008212`, không starvation và ledger về 0. Heterogeneous-prompt stress còn
  order-sensitive `0,874348–0,936995×` token throughput và được giữ công khai
  làm batching-efficiency backlog.
- M6 đã pass scoped exact-token cache, longest safe exact-state prefix
  checkpoint, immutable private session generations/COW và cross-scope
  rejection trên GPU. Cache on/off benchmark cùng model/binary ghi hit rate
  `80%`, saved prefill `2.560` token, TTFT p50 ratio `0,111332`, semantic output
  bằng nhau, cache bytes về 0 sau clear và dedicated VRAM process-scoped WDDM
  delta `4 MiB`.
- M7 portable architecture gate đã pass. Host Windows hiện tại không có
  MLX-LM/vLLM/SGLang runtime/server thật, nên real-provider execution matrix
  vẫn là deployment gate và không được tuyên bố pass.
- H0 đã đạt engineering exit gate. H1 atomic read-only
  slice đã đạt engineering checkpoint với fake inference/executor, durable
  recovery và concrete allowlisted file tools. Full H1 chưa được tuyên bố vì
  chưa có real-local-model acceptance artifact; mutation/approval execution
  vẫn bị khóa. H2–H3 chưa bắt đầu implementation.

Release attestation hiện hành nằm ở
`artifacts/inference-runtime/2026-07-16-m0-release-attestation.json` và
`release-evidence/b38b6df32755c55e668ac11e6c8f3e8b1c2ad46b/`. Engineering evidence
M3–M7/H0/H1 vẫn được index bởi
`artifacts/inference-runtime/2026-07-15-m0-m7-final-verification.json`; các cờ
`working_tree_dirty=true` trong artifact cũ mô tả đúng checkpoint pre-release và
không phủ định attestation sạch ngày 2026-07-16.

## 3. Hai scheduler, hai state machine

Không dùng từ “scheduler” chung cho hai tầng.

### 3.1 HarnessTaskScheduler

Sở hữu:

- Task DAG, dependency và role assignment.
- Task lifecycle, reviewer, acceptance và handoff.
- Tool permission, retry, replan và human approval.
- Workflow/agent budgets và durable event log.
- Quyết định task nào đủ điều kiện gửi inference.

Không sở hữu:

- llama context, sequence ID hoặc KV.
- Decode batch, GPU scheduling hoặc cache eviction vật lý.

### 3.2 InferenceScheduler

Sở hữu:

- Admission, queue, deadline, priority class và fairness.
- Sequence lifecycle, prefill chunks và decode microbatches.
- KV/context resource ledger, cancellation và release.
- Prefix/session cache lifecycle.
- Backend/model routing và inference telemetry.

Không sở hữu:

- Task semantic completion.
- Tool permission hoặc project acceptance.
- Planner/reviewer business semantics.

### 3.3 State machine phải được namespace

- `TaskLifecycle`: trạng thái task/handoff của harness.
- `AgentRunLifecycle`: trạng thái một agent loop.
- `ToolInvocationLifecycle`: trạng thái một lần gọi tool.
- `InferenceRequestLifecycle`: request ở HTTP/control plane.
- `SequenceLifecycle`: prefill/decode/KV ở native runtime.

Không được dùng lời tuyên bố `completed` của model để đổi bất kỳ lifecycle hệ
thống nào.

## 4. Kiến trúc đích

~~~text
User / API
    ↓
Agent Harness
    ├── Product / BA / Architect / Task Splitter
    ├── HarnessTaskScheduler
    ├── Atomic Agent / Reviewer / Acceptance
    └── Tool Runtime
            ↓
        InferencePort
            ↓
Inference Control Plane
    ├── Admission + quota + deadlines
    ├── Priority/fairness policy
    ├── Model/backend routing
    └── Event/metrics/tracing
            ↓
Native Model Runtime
    ├── ModelHandle
    ├── SequenceHandle + per-sequence state
    ├── Prefill/decode microbatch builder
    ├── KV/session/prefix lifecycle
    └── llama.cpp adapter
            ↓
llama.cpp → CUDA hiện tại; backend khác về sau
~~~

### 4.1 Hai loại backend capability

Không ép mọi backend vào lowest common denominator.

- `SteppableBackend`: llama.cpp hoặc backend cho phép project sở hữu sequence,
  prefill và decode step.
- `ManagedBackend`: vLLM/SGLang hoặc server khác chỉ cung cấp generate/stream/
  cancel và tự sở hữu scheduler.

Harness luôn gọi `InferencePort` high-level. Chỉ Inference Control Plane mới
biết backend nào cho phép step-level scheduling.

### 4.2 Priority không hardcode role trong native

Harness ánh xạ role sang service class:

~~~text
Planner  → interactive-critical
Coder    → throughput
Reviewer → background
Tester   → batch
~~~

Inference runtime chỉ xử lý weight, deadline, quota, aging và cache scope.
Không nhúng tên Planner/Coder/Reviewer vào C++.

## 5. Tool calling cho model nhỏ

Tool runtime nằm trong package `agent_runtime/`, phía trên `model_worker`.
Không thêm tool frame, filesystem, command hoặc network executor vào model
worker/native process.

### 5.1 Nguyên tắc

1. Mỗi agent turn chỉ đề xuất đúng một action.
2. Model tạo semantic `ToolIntent`, không tạo native API call và không thực thi.
3. Code validate, compile, authorize và execute.
4. `ArgumentResolver` bằng LLM chỉ là fallback khi code không thể bind semantic
   slots; nó không được đổi tool, cấp quyền hoặc execute.
5. FlowController bằng code quyết định direct result, synthesize, redecide,
   verify, pause hoặc handoff.
6. Tool result luôn là dữ liệu không tin cậy.
7. Mọi tool decision kết thúc inference turn hiện tại. Không giữ GPU/KV chỉ để
   chờ I/O. Lượt sau là inference request mới, dù vẫn thuộc cùng agent run.
8. V1 không cho model đề xuất mảng parallel tool calls. Parallelism nằm ở
   HarnessTaskScheduler giữa các task độc lập.

### 5.2 ToolIntent v1 tối giản nhưng đủ bốn action

Contract quyết định phải dùng được ngay với `structured-output.v1`: root phẳng,
mọi field đều required; field không dùng nhận `null`. Đây là agent decision
contract, không phải worker response contract chung.

~~~json
{
  "action": "call_tool",
  "tool_id": "read_file",
  "objective": "Đọc file cấu hình được giao trong task",
  "input_hint": "target_file",
  "message": null
}
~~~

Schema được compile riêng cho từng turn. `tool_id.enum` chỉ chứa shortlist đã
authorize sơ bộ cộng `null`:

~~~json
{
  "type": "object",
  "properties": {
    "action": {
      "type": "string",
      "enum": ["call_tool", "ask_user", "submit", "blocked"]
    },
    "tool_id": {
      "type": ["string", "null"],
      "enum": ["read_file", "search_text", "run_test", null]
    },
    "objective": {"type": ["string", "null"]},
    "input_hint": {"type": ["string", "null"]},
    "message": {"type": ["string", "null"]}
  },
  "required": ["action", "tool_id", "objective", "input_hint", "message"],
  "additionalProperties": false
}
~~~

`DecisionValidator` bằng code áp cross-field invariant sau grammar:

| action | tool_id | objective | input_hint | message |
|---|---|---|---|---|
| `call_tool` | non-null, thuộc shortlist | non-null | semantic slot/ref non-null | null |
| `ask_user` | null | null | null | câu hỏi non-null |
| `submit` | null | null | null | submission note non-null |
| `blocked` | null | null | null | lý do non-null |

`submit` chỉ đề nghị chuyển sang deterministic acceptance; không đổi task sang
`SUCCEEDED`. `blocked` cũng chỉ là đề nghị có lý do; code kiểm tra còn đường
retry/replan/handoff hay không.

Trước khi prompt, registry lọc tool theo skill, role, task state và permission.
Model chỉ thấy shortlist nhỏ, dự kiến 3–8 tool, gồm semantic slot, effect class
và result shape đã rút gọn; model không thấy secret, raw command hoặc native
schema không cần thiết. Prompt ghi `tool_catalog_digest` để replay/audit đúng
catalog revision đã dùng.

### 5.3 Ba lane tạo arguments

1. **Deterministic fast path**
   - Lấy ref/value từ task state, artifact, default hoặc alias đã khai báo.
   - ToolAdapter canonicalize và chuyển agent-facing slots sang native schema.
2. **ArgumentResolver fallback**
   - Chỉ nhìn thấy một tool đã chọn và schema agent-facing nhỏ.
   - Đầu ra vẫn bị grammar + validator kiểm tra.
   - Không được đổi tool hoặc tự thêm quyền.
3. **Ask user**
   - Dùng khi thiếu dữ liệu, mơ hồ hoặc cần approval.

Compiler không được tự suy diễn semantic, secret, permission hoặc đường dẫn
ngoài scope. Tool native phức tạp phải có adapter đơn giản; không expose raw
shell command hay path tùy ý cho model.

### 5.4 Internal protocol và compatibility boundary

Core dùng:

~~~text
ToolIntent
→ CompiledToolCall
→ ToolResultEnvelope
→ FlowTransition
~~~

Boundary adapter có thể chuyển thành OpenAI-compatible assistant tool-call và
tool-result messages. Mọi call có `internal_call_id`; provider ID/index chỉ là
mapping của adapter.

Mặc định:

- Một call mỗi model turn.
- Executor chạy tuần tự.
- Lượt synthesize không được cấp tools.
- Không execute argument fragments khi streaming; phải assemble và validate
  call hoàn chỉnh.

### 5.5 Tool registry có hai schema

Mỗi tool revision khai báo:

- Agent-facing semantic schema nhỏ.
- Native input/output schema đầy đủ.
- Deterministic adapter/compiler revision.
- Effect class: read-only, idempotent mutation, non-idempotent mutation.
- Permission/scope policy.
- Default flow policy.
- Timeout, result byte/token cap và retry class.

### 5.6 ToolResultEnvelope

Raw output lớn được lưu bằng ref/hash. Chỉ bounded, normalized data đi vào lượt
LLM sau.

~~~json
{
  "invocation_id": "inv_...",
  "tool_id": "read_file",
  "tool_version": "1",
  "status": "success",
  "summary": "Đã đọc 120 dòng",
  "data_ref": "artifact://...",
  "data_hash": "sha256:...",
  "truncated": false,
  "side_effect_state": "none",
  "error": null
}
~~~

Result phải có provenance, byte/token cap, redaction và typed error. Không nối
raw result vào system prompt. Tool output và mọi summary sinh từ tool output
vẫn mang nhãn `untrusted`.

### 5.7 FlowPolicy

FlowController chọn policy theo tool/task/role:

- `DIRECT_RETURN_SAFE`: chỉ renderer allowlisted được trả trực tiếp.
- `SYNTHESIZE_NO_TOOLS`: gọi lượt tổng hợp không có tools.
- `REPLAN_WITH_OBSERVATION`: gọi Decision Agent mới với result envelope.
- `VERIFY_THEN_REPLAN`: bắt buộc cho mutation khi policy yêu cầu.
- `PAUSE_FOR_APPROVAL`
- `PERSIST_AND_STOP`
- `HANDOFF`

Model không tự chọn policy này.

### 5.8 AgentRun và ToolInvocation

AgentRun public lifecycle giữ nhỏ:

~~~text
READY
→ DECIDING
→ WAITING_TOOL
→ OBSERVING
→ DECIDING | SYNTHESIZING | VERIFYING
→ SUCCEEDED | BLOCKED | FAILED

Nhánh: WAITING_USER, PAUSED, CANCELLED, TIMED_OUT, BUDGET_EXHAUSTED
~~~

ToolInvocation lifecycle riêng:

~~~text
PROPOSED
→ ARGS_READY | NEEDS_RESOLUTION
→ AUTHORIZED | WAITING_APPROVAL | DENIED
→ DISPATCHED
→ SUCCEEDED | PARTIAL | FAILED | UNKNOWN_OUTCOME
~~~

Chi tiết transition được lưu dưới dạng append-only events. Transcript và KV
không phải source of truth.

### 5.9 Idempotency và mutation safety

- Tạo stable `action_id` và `idempotency_key` trước dispatch.
- Mỗi lần thử có `attempt_id` riêng.
- Read-only/transient có thể retry theo policy.
- Idempotent mutation retry dùng cùng key và executor deduplicate.
- Non-idempotent call timeout/crash sau dispatch phải vào
  `UNKNOWN_OUTCOME`; probe/reconcile hoặc yêu cầu người dùng, không replay mù.
- Mutation ưu tiên dry-run/diff/approval/commit khi tool hỗ trợ.
- Persist/audit phải thành công trước khi dispatch mutation; nếu không thì fail
  closed.

### 5.10 Error taxonomy

Tối thiểu phải phân biệt:

- Decision: invalid, unknown tool, not allowed, loop.
- Arguments: missing, ambiguous, invalid, resolver exhausted.
- Policy: denied, approval required/denied, scope violation, tainted input.
- Execution: unavailable, rate-limited, pre-dispatch timeout,
  post-dispatch unknown, failed, cancelled, partial, rollback failed.
- Result: schema invalid, normalization failed, too large/truncated,
  insufficient, stale.
- Runtime: illegal transition, state conflict, persistence failure, deadline
  hoặc budget exhausted.
- Acceptance: failed hoặc unverifiable.

Mỗi error mang `retryable`, `retry_scope` và `side_effect_state`.

### 5.11 Skill boundary

`skills/atomic-worker` hiện tại được giữ làm no-tool baseline. Tool-using agent
có skill/prompt riêng cho decision và synthesis; không âm thầm đổi semantics
của Atomic Worker hiện tại.

### 5.12 Trình tự normative của một tool turn

Một lượt `call_tool` phải đi qua đúng thứ tự sau:

1. Load durable `AgentRun` snapshot và giữ task lease/version để chống hai
   executor cùng sửa một state.
2. Lọc catalog theo role, skill, tenant, task scope và approval policy; tạo
   `tool_catalog_digest`.
3. Build bounded decision context và gọi `InferencePort` với ToolIntent schema
   của turn này.
4. Inference kết thúc hoàn toàn; sequence/KV được release. Từ đây không còn giữ
   GPU trong khi xử lý tool.
5. Strict-parse ToolIntent, validate schema rồi validate cross-field invariant.
6. Tạo stable `action_id`; persist event `ToolProposed` trước mọi side effect.
7. Chạy deterministic compiler. Chỉ khi thiếu semantic binding mới tạo một
   `ArgumentResolver` turn riêng, chỉ thấy một tool và không được execute.
8. Validate native args, scope, taint, permission, effect class, deadline và
   approval. Tạo `invocation_id`, `idempotency_key` và persist
   `ToolAuthorized`/`ApprovalRequested`.
9. Executor dispatch với timeout/cancel token. Mutation chỉ dispatch sau khi
   audit persistence thành công.
10. Normalize output, redact secret, validate output schema, cap byte/token,
    lưu raw payload thành immutable artifact và tạo `ToolResultEnvelope`.
11. Persist terminal invocation event trước khi FlowController đi tiếp.
12. FlowController bằng code chọn direct renderer, synthesize-no-tools,
    replan, verify, approval, handoff hoặc stop.
13. Nếu cần LLM tiếp, tạo inference request mới với bounded observation
    envelope; không nối raw output và không resume hidden reasoning.
14. Deterministic acceptance/reviewer mới được chuyển task lifecycle.

Nếu crash ở bước 8–10, replay đọc durable state và effect class. Read-only hoặc
idempotent mutation có thể retry theo policy; non-idempotent mutation đã
dispatch nhưng chưa xác định kết quả phải vào `UNKNOWN_OUTCOME`.

### 5.13 Context pack và correlation

Context cho model nhỏ chỉ gồm phần cần cho turn hiện tại, theo thứ tự ổn định:

1. Fixed decision/synthesis instruction có version.
2. Task objective, acceptance criteria và bounded state summary.
3. Allowlisted state/artifact refs mà compiler có thể bind.
4. Shortlisted semantic tool cards và effect label.
5. Observation envelope gần nhất nếu đây là replan/synthesis turn.
6. ToolIntent hoặc synthesis output contract.

Không đưa full event log, raw tool payload, secret, permission implementation,
provider metadata hoặc transcript vô hạn vào prompt. Mỗi phần có byte/token cap
và digest; overflow phải summarize/store-by-ref hoặc fail closed, không truncate
JSON/schema tùy ý.

Mọi event mang ít nhất:

~~~text
workflow_id → task_id → agent_run_id → turn_id
                                      ├→ inference request_id / attempt_id
                                      └→ action_id → invocation_id → tool attempt_id
~~~

`request_id` của inference và `attempt_id` của tool là hai namespace khác nhau.
Event log/artifact store là source of truth; transcript, provider call ID và KV
cache chỉ là derived/ephemeral state.

## 6. Roadmap Inference Runtime

Ước lượng dành cho một developer full-time. Đây là effort band, không phải cam
kết lịch. Critical path đến multi-agent song song thực dụng dự kiến 14–20 tuần;
production hardening đầy đủ dài hơn.

| Mốc | Nội dung | Exit gate |
|---|---|---|
| **M0 — Worker v1 thật sự, 1–2 tuần** | Sửa queue expiry, cancel race, bounded startup/restart, readiness state, IPC corruption recovery, graceful shutdown, contract instructions, reasoning-none, UTF-8/grammar acceptance. Chốt commit sạch và chạy lại release gate. | Mọi cancel/timeout kết thúc trong deadline + grace; `/ready=503` khi load/restart; GPU + fault + soak cùng revision. |
| **M1 — Event và telemetry, 1–2 tuần** | Bounded event bus; chuyển final delta/progress/heartbeat đến SSE; slow-client backpressure; disconnect cancel; histogram hữu hạn; registry TTL; artifact cleanup job. | Stream 1.000 token không O(n²); client chậm không giữ GPU vô hạn; metrics reconcile đúng một lần. |
| **M2 — Port và scheduler simulator, 1–2 tuần** | Tách `InferencePort` high-level và `SteppableBackend` low-level. Thêm `SequenceHandle`, event sink, capability flags và fake-clock simulator. | Serial llama adapter giữ behavior; property tests cho deadline, fairness, aging và cancellation. |
| **M3 — Native sequence engine, 3–4 tuần** | Refactor generate thành per-sequence state; persistent context, seq ID, explicit release và resource ledger. Round-robin, chưa cache. | 2/4/8 sequence interleaved không rò state; cancel một sequence không ảnh hưởng sequence khác; VRAM/RAM ổn định. |
| **M4 — Continuous batching, 3–4 tuần** | Prefill/decode queues, chunked prefill, decode-first với starvation bound, dynamic microbatch và token budget mỗi tick. | Ở concurrency 4: mục tiêu ban đầu ≥1,5× serial throughput và ≥90% llama-server baseline, với p95 ITL trong SLO. |
| **M5 — Priority và admission, 2–3 tuần** | Hierarchical workflow→agent→request queues; DRR/WFQ, aging, deadline slack, quota, VRAM/KV admission và load shedding. | Không starvation; priority workload cải thiện TTFT mà throughput tổng giảm không quá 10%; cancel release trong một tick + decode hiện tại. |
| **M6 — Prefix/session cache, 3–4 tuần** | Tách immutable shared prefix và mutable session; exact-token key, model/template/tokenizer/adapter/context digest, refcount/COW, scope, TTL/LRU và byte budget. | Không cross-agent leak; invalidation đúng; đo hit rate, saved prefill, TTFT và VRAM; không giảm contract/semantic correctness. |
| **M7 — Multi-model/backend, 3–5 tuần** | Capability registry, load/unload/keepalive và routing. Steppable adapter cho llama.cpp; managed adapter cho MLX-LM/vLLM/SGLang khi public API sở hữu full generation loop. Chỉ nâng backend thành steppable khi provider có lifecycle/step API ổn định. | Conformance suite chung; harness không phụ thuộc API KV/token-step riêng của llama.cpp và không advertise capability backend không có. |

### 6.1 Implementation/release checkpoint 2026-07-16

| Mốc | Code hiện tại | Exit gate |
|---|---|---|
| M0 | Functional hardening, native fault, restart recovery, semantic GPU và 500-request soak cùng clean identity. | **Released:** consolidated gate pass tại `b38b6df32755c55e668ac11e6c8f3e8b1c2ad46b`, 500/500 và 0 failure. |
| M1 | Event/telemetry/backpressure/cleanup đã implement và nằm trong cùng attested revision. | **Engineering pass;** clean M0 attestation đã hoàn tất. |
| M2 | Ports, contracts và simulator đã implement; unit/property pass. | **Engineering pass.** |
| M3 | Native 2/4/8 sequence, cancel isolation/slot reuse và soak 15 phút pass trên RTX 3080. | **Pass:** 1.228 request, 0 lỗi, RSS drift 1.024.000 byte, process VRAM drift 0 MiB, clean shutdown. |
| M4 | Continuous batching và token timing evidence đã benchmark với cùng workload/binary. | **Pass:** `2,274732×` serial; `1,122704×` llama-server request; `1,064309×` token; ITL SLO pass. |
| M5 | Governance/admission và 3 paired priority runs pass; cancel release có GPU test. | **Pass portable gate;** heterogeneous-prompt range `0,874348–0,936995×` còn là order-sensitive optimization backlog. |
| M6 | Scoped longest-safe-prefix cache và immutable private session COW pass GPU/unit; cache on/off benchmark đo hit rate, saved prefill, TTFT, RSS/VRAM và cleanup. | **Pass:** correctness/performance đều đạt; WDDM dedicated process VRAM delta 4 MiB và cache bytes về 0 sau clear. |
| M7 | Registry/router, llama.cpp steppable và managed vLLM/SGLang/MLX-LM pass portable conformance. | **Portable pass; deployment pending:** real provider matrix không khả dụng trên host này. |

Trạng thái `pass` trong bảng này không thay thế Definition of Done. M0 chỉ được
ký bởi `summary.json` của revision sạch nêu trên; M2–M7 vẫn là engineering/
portable gates theo đúng cột giới hạn và không được hiểu là mọi deployment
backend đều đã production-certify.

## 7. Roadmap Agent Harness

| Mốc | Nội dung | Exit gate |
|---|---|---|
| **H0 — Durable state + contracts** | Tạo `agent_runtime`, event-sourced run state, ToolIntent/CompiledToolCall/ToolResultEnvelope/FlowPolicy, budgets, redaction và idempotency. Dùng v1 flat decision contract trước; chỉ thêm structured-output.v2 khi tool schema thực tế cần nested/array/optional. | Crash/replay giữ cùng state và stable action claim; illegal transition/property tests pass; budget và audit reconcile. |
| **H1 — Atomic tool agent** | Decision turn, deterministic fast path, optional ArgumentResolver, registry/compiler/permission/executor/normalizer, synthesize-no-tools và deterministic acceptance. | One action/turn; denied scope chặn trước executor; malformed/insufficient result/retry/cancel/idempotency suites pass; model không tự accept. |
| **H2 — HarnessTaskScheduler** | Task Splitter, DAG, reviewer, handoff, replan, Product/BA/Architect pipeline và durable audit. | Chỉ dependency accepted mới chạy; reviewer reject/retry/replan/blocked cases end-to-end pass. |
| **H3 — Parallel agent executor** | Chạy DAG node độc lập song song, dùng capacity/admission signal từ InferenceScheduler. | Mixed Planner/Coder/Reviewer/Tester workload không starvation, không vượt quota và không làm sai dependency/acceptance. |

### 7.1 H0 implementation checkpoint — 2026-07-15

File map production:

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

Tests H0 nằm ở `tests/unit/agent_runtime_h0/` (dùng hậu tố `_h0` để tránh
module test trùng tên với `tests/unit/test_contracts.py`) và hai file property
`tests/property/test_agent_runtime_lifecycle.py`,
`tests/property/test_agent_runtime_replay.py`.

Kết quả chính xác của checkpoint:

~~~text
python -m pytest tests/unit tests/property -q
→ 327 passed, 2 skipped in 5.69s

python -m coverage run --branch --source=agent_runtime -m pytest \
  tests/unit/agent_runtime_h0 \
  tests/property/test_agent_runtime_lifecycle.py \
  tests/property/test_agent_runtime_replay.py -q
python -m coverage report -m
→ 64 passed in 3.70s; TOTAL 97% (595 statements, 168 branches)

python -m ruff check .
→ All checks passed!
~~~

Hai skip vẫn là symlink privilege Windows đã ghi ở checkpoint trước. Một lần
chạy toàn suite đã làm lộ lost-wakeup race thật trong idle wait của continuous
scheduler (`queue_timeout` thay vì `output_invalid`). Scheduler hiện re-check
work dưới cùng condition lock trước khi wait; reproduction cũ đã pass 100/100
process độc lập và có deterministic regression test cho đúng interleaving.

Evidence máy đọc được:
`artifacts/inference-runtime/2026-07-15-h0-verification.json`. Artifact ghi
`working_tree_dirty=true` và `release_attestation=false`; nó không đóng hay thay
đổi trạng thái các gate M0/M3–M7.

### 7.2 H1 atomic read-only checkpoint — 2026-07-15

H1 read-only slice đã implement các phase H1.0–H1.8 cho hai adapter
`read_file` và `search_text`. Invariant đã chứng minh gồm one-action-per-turn,
terminal inference trước tool I/O, durable dispatch claim trước executor,
stable IDs/idempotency, fail-closed scope/taint/permission, bounded untrusted
result, synthesis không có tool và chỉ AcceptanceGate mới được ghi
`SUCCEEDED`.

Verification tập trung:

~~~text
H0/H1 unit + property + integration: 88 passed, 1 skipped
agent_runtime branch coverage:       92% (1.208 statements, 332 branches)
~~~

Skip duy nhất là symlink privilege trên Windows; traversal và resolved-scope
checks vẫn chạy. Evidence máy đọc được:
`artifacts/inference-runtime/2026-07-15-h1-atomic-readonly-verification.json`.

Đây chưa phải full H1 pass. Gate còn lại là một real local-model decision đi
qua allowlisted read/search tool đến deterministic acceptance và artifact riêng.
Mutation/approval execution chưa được bật; nếu mở scope này sau này thì toàn bộ
approval-token, idempotent retry và unknown-outcome matrix phải pass trước.

Dependency:

~~~text
M0 → M1 → M2 → M3 → M4 → M5 → M6 → M7
           └→ H0 → H1 → H2 ─────→ H3
                                  ↑
                                  M5
~~~

H0/H1 có thể bắt đầu sau M0/M2 với serial backend. H3 bắt buộc đợi H2 và M5.

## 8. Benchmark và SLO gate

Mỗi thay đổi scheduler/cache phải benchmark trên cùng model digest, runtime,
hardware và workload manifest.

Workload tối thiểu:

- Serial correctness baseline.
- Short-prompt/short-output interactive.
- Long-prefill/short-decode.
- Short-prefill/long-decode.
- Mixed Planner/Coder/Reviewer/Tester.
- Prefix-heavy.
- Cancel storm, queue saturation và worker crash.
- Tool flow: compiler fast path, resolver fallback, slow tool, mutation unknown
  outcome và insufficient result.

Metrics tối thiểu:

- TTFT, inter-token latency/TPOT và end-to-end p50/p95/p99.
- Prompt/decode tokens per second.
- Scheduler wait, active/queued sequences và batch occupancy.
- Prefill/decode tokens mỗi tick, preemptions và starvation time.
- KV bytes, headroom, fragmentation, cache hit/eviction và saved prefill tokens.
- Cancellation latency, deadline miss và failure class.
- Fairness theo workflow/agent/service class.
- Agent success, acceptance fail, loop rate, tool calls/run.
- Compiler fast-path rate, ArgumentResolver rate/pass rate.
- Retry/idempotency/unknown-outcome counts.
- Peak/stable RAM, VRAM, file descriptors và disk.

Resource evidence phải ưu tiên process tree. Trên Windows WDDM, khi
`nvidia-smi --query-compute-apps` trả memory `N/A`, dùng performance counter
`GPU Process Memory(*)\\Dedicated Usage` và lọc đúng PID tree; chỉ dùng
`total_system_fallback` khi cả hai nguồn process-scoped không khả dụng.

Không tuyên bố “thay vLLM” trước khi custom runtime đạt gate trên workload agent
thực tế. llama-server là baseline bắt buộc vì đã có continuous batching và
prompt reuse. Nếu unified KV của llama.cpp không đạt concurrency/VRAM mục tiêu,
ưu tiên managed vLLM/SGLang adapter thay vì tự viết paged attention/KV allocator.

## 9. Invariant không được đánh đổi

- Không áp grammar lên reasoning.
- Reasoning phase vẫn được generate/đếm cho protocol nhưng không stream/persist
  mặc định.
- Không dùng reasoning làm final fallback.
- Không để model quyết định system completion, permission hoặc retry mutation.
- Không concurrency trước sequence/KV isolation.
- Không cache trước ownership, scope, invalidation và cleanup.
- Không execute partial/invalid tool call.
- Không đưa raw/untrusted tool output vào system prompt.
- Không replay non-idempotent mutation khi side-effect state chưa biết.
- Không giữ GPU sequence trong lúc chờ tool I/O.
- Không mở rộng contract version mà thiếu compiler/validator differential tests.
- Không hardcode role semantics trong native scheduler.
- Không buộc managed backend phải giả lập step-level APIs.

## 10. Definition of Done cho parallel-agent MVP

- Model Worker M0–M1 có evidence cùng revision/runtime.
- Multi-sequence isolation và continuous batching M3–M4 pass.
- Priority/admission M5 không starvation và có resource bound.
- H0 durable state có crash/replay/idempotency evidence.
- H1 hoàn thành một task nhỏ qua tool với deterministic acceptance.
- H2 DAG/reviewer/replan end-to-end pass.
- H3 chạy task độc lập song song mà không vi phạm dependency, permission,
  budgets hoặc acceptance.
- Toàn bộ inference, tool và task transitions có correlation IDs và audit.
- Không có state truth chỉ tồn tại trong transcript, reasoning hoặc KV cache.

## 11. Artifact và tài liệu liên quan

- `01-vision.md`: nguyên tắc sản phẩm.
- `02-architecture.md`: ranh giới LLM/code và role.
- `03-progress-and-roadmap.md`: lịch sử nghiên cứu ban đầu.
- `04-session-handoff.md`: index cho session mới.
- `05-controlled-inference-handoff.md`: archive Phase A–G.
- `06-model-worker-v1-plan.md`: baseline/release contract Model Worker v1.
- `07-inference-runtime-and-agent-roadmap.md`: roadmap hiện hành.
- `08-m0-implementation-handoff.md`: historical artifact lúc bắt đầu M0.
- `09-inference-runtime-implementation-handoff.md`: checkpoint code/evidence và
  các gate/authority còn lại cho session kế tiếp.
- `artifacts/inference-runtime/2026-07-15-m0-m7-final-verification.json`:
  machine-readable index hiện hành cho identity, test slices, milestone status,
  giới hạn release và SHA-256 của artifact M3–M7/H0/H1.
- `artifacts/inference-runtime/2026-07-15-m0-m7-requirement-audit.json`:
  requirement-by-requirement traceability cho từng exit gate M0–M7 và trạng
  thái clean-release còn lại.
- `artifacts/inference-runtime/2026-07-15-m3-soak-final-stable-shape.json`:
  15-minute correctness/resource soak với fixed shapes warm từ wave đầu và
  process-scoped WDDM VRAM.
- `artifacts/inference-runtime/2026-07-15-m4-benchmark-final.json`: benchmark
  custom runtime so với serial và llama-server; M4 gate pass.
- `artifacts/inference-runtime/2026-07-15-m5-priority-gate-aggregate.json`:
  paired priority/fairness/throughput evidence và heterogeneous backlog.
- `artifacts/inference-runtime/2026-07-15-m6-cache-gate.json`: prefix/session
  cache scope, COW và GPU verification.
- `artifacts/inference-runtime/2026-07-15-m6-cache-performance.json`: cache
  on/off hit-rate, saved-prefill, TTFT, RSS/VRAM và cleanup evidence.
- `artifacts/inference-runtime/2026-07-15-m7-backend-conformance.json`: portable
  backend matrix; real-provider deployment matrix được ghi rõ là unavailable.
- `artifacts/inference-runtime/2026-07-15-h1-atomic-readonly-verification.json`:
  H1 read-only vertical slice, invariant/test coverage và full-H1 limits.
- `model-worker-runbook.md`: vận hành worker.
- `model-worker-release.md`: release gate.
