# Session handoff index

Ngày cập nhật: 2026-07-15.

Tài liệu này là điểm vào cho session mới. Nó không chứa implementation plan
chi tiết; artifact có thẩm quyền cho công việc tiếp theo là
`09-inference-runtime-implementation-handoff.md`.

## Đọc theo thứ tự

1. `PROJECT.md`
2. `docs/01-vision.md`
3. `docs/02-architecture.md`
4. `docs/06-model-worker-v1-plan.md`
5. `docs/07-inference-runtime-and-agent-roadmap.md`
6. `docs/09-inference-runtime-implementation-handoff.md`
7. `docs/model-worker-runbook.md`
8. `docs/model-worker-release.md`

Chỉ đọc `docs/03-progress-and-roadmap.md` và
`docs/05-controlled-inference-handoff.md` và
`docs/08-m0-implementation-handoff.md` khi cần lịch sử/quyết định cũ.
`controlled_inference/` là archive của prototype A–G.

## Trạng thái kiến trúc

Production path:

~~~text
HTTP/service control plane: model_worker/
Inference control plane:      inference_runtime/
Native llama.cpp runtimes:    native/
Current roadmap:              docs/07-inference-runtime-and-agent-roadmap.md
Current checkpoint/handoff:   docs/09-inference-runtime-implementation-handoff.md
~~~

Model Worker v1 vẫn là compatibility path single-model/fresh-context. Working
tree hiện đã có multi-sequence native runtime, continuous scheduler,
priority/admission, prefix/session cache và backend registry/router. H0 trong
`agent_runtime/` đã có durable contracts, reducer và SQLite event store. H1
atomic read-only slice đã nối decision, compiler, permission, bounded executor,
normalization, flow và deterministic acceptance cho `read_file`/`search_text`;
chưa có durable DAG. Full H1 vẫn chờ real-local-model acceptance artifact và
mutation/approval execution vẫn bị khóa. M3–M6 đã đạt engineering gate; M7 đạt
portable architecture gate nhưng chưa có real-provider execution trên host
hiện tại. Bảng trạng thái và giới hạn release nằm trong docs 07.

Hai scheduler phải được tách:

- `HarnessTaskScheduler`: DAG, role, dependency, tool, acceptance, retry,
  reviewer và handoff.
- `InferenceScheduler`: sequence, prefill/decode, admission, priority, KV,
  batching và backend routing.

## Quyết định không được làm mất

- LLM quyết định semantic; code tất định kiểm soát syntax, state, permission,
  budgets, retry và acceptance.
- Model không được tự xác nhận system completion.
- Task Splitter chỉ chia việc; Atomic Worker chỉ thực hiện một task nhỏ.
- Reasoning phase vẫn được generate/đếm cho protocol nhưng private mặc định.
- Grammar chỉ áp lên final, không áp lên reasoning.
- Không concurrency trước sequence/KV isolation.
- Không cache trước ownership/scope/invalidation/cleanup.
- Không đưa tool execution vào `model_worker` hoặc native llama process.
- Tool-using agent v1 chỉ một action mỗi turn; code authorize/execute.
- Transcript, reasoning và KV cache không phải source of truth.

## Working tree

Tại thời điểm cập nhật handoff, repo có nhiều thay đổi code/test/docs chưa
commit thuộc về người dùng. Session mới phải chạy:

~~~powershell
git status --short
git diff --stat
~~~

Không reset, checkout hoặc ghi đè các thay đổi đó. Docs 07/09 và
`artifacts/inference-runtime/2026-07-15-m0-m7-final-verification.json` định danh
checkpoint hiện tại; consolidated evidence cũ không đại diện cho revision chưa
chốt.

Requirement-by-requirement audit nằm ở
`artifacts/inference-runtime/2026-07-15-m0-m7-requirement-audit.json`; audit này
ghi rõ M1–M7 engineering pass và M0 release attestation vẫn pending.

## Công việc còn lại cho session kế tiếp

Không viết lại H1 read-only slice. Bắt đầu từ evidence hiện có và thực hiện theo
thứ tự:

1. Review working tree; chỉ khi người dùng cho phép mới commit và chạy clean
   consolidated M0 release gate.
2. Đóng full H1 bằng một real local-model decision → allowlisted read/search
   tool → normalized observation → deterministic acceptance artifact.
3. Chỉ triển khai mutation/approval matrix nếu người dùng chủ động mở scope;
   mutation execution hiện phải tiếp tục fail closed.

Không ghép các việc trên với:

- Task DAG/parallel agent (H2/H3).
- Tuning kernel/KV allocator.
- Tuyên bố M0 released trước clean consolidated gate.
- Tuyên bố M7 real-provider matrix pass khi môi trường chưa có provider thật.

H1 phải tiếp tục dùng H0 làm source of truth và giữ pipeline: decision → strict
validation → deterministic argument compile → permission/approval → persist
claim → execute → normalize/redact → persist terminal event → flow policy →
deterministic acceptance. Không expose raw shell hoặc đường dẫn tùy ý cho model
nhỏ.
