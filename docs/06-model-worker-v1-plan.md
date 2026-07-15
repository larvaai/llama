# Kế hoạch Model Worker v1

> **Vai trò tài liệu:** baseline, trust boundary và release contract cho
> single-model worker. Roadmap post-v1 hiện hành nằm ở
> `07-inference-runtime-and-agent-roadmap.md`; checkpoint triển khai hiện hành
> nằm ở `09-inference-runtime-implementation-handoff.md`. File 08 chỉ còn là
> historical artifact lúc bắt đầu M0.

> **Release status (2026-07-16):** contract M0 trong tài liệu này đã được ký từ
> clean revision `b38b6df32755c55e668ac11e6c8f3e8b1c2ad46b`. Xem
> `model-worker-release.md` và
> `release-evidence/b38b6df32755c55e668ac11e6c8f3e8b1c2ad46b/summary.json`. Nội dung
> bên dưới tiếp tục là normative baseline; đổi manifest/runtime/model/native
> identity phải chạy gate mới.

## 1. Quyết định phạm vi

Mục tiêu của mốc tiếp theo là biến controlled-inference spike hiện tại thành một **single-model worker đáng tin cậy** trên llama.cpp.

Model Worker v1 phải làm được:

- Load đúng một model từ manifest và giữ model resident.
- Nhận request inference có contract được version hóa.
- Áp chat template, tokenize, preflight context và decode prompt đúng giới hạn batch.
- Kiểm soát reasoning protocol bằng state machine tất định.
- Chỉ bật grammar ở phase final.
- Parse và validate structured output bằng cùng một contract đã dùng để sinh grammar.
- Có queue hữu hạn, cancellation, deadline, watchdog và crash recovery.
- Stream final an toàn, cung cấp telemetry và lưu artifact có lifecycle rõ ràng.
- Trả lỗi có taxonomy ổn định để caller tự quyết định retry.

Model Worker v1 **không** làm:

- Tool calling hoặc thực thi command/file/network.
- Agent loop `observe → act → observe`.
- Permission gate cho tool.
- Planner, task splitter, scheduler, reviewer hoặc multi-agent.
- Nghiệm thu semantic của task.
- Tự retry task hoặc tự thay đổi prompt/goal.
- Session chat, KV reuse, prompt cache, continuous batching hoặc multi-model scheduling.
- Durable task queue hoặc replay request sau khi cả service chết; registry v1 là state vận hành trong memory.

Fresh context cho mỗi request tiếp tục là invariant bắt buộc. Các mục bị loại khỏi v1 chỉ được xem lại sau khi worker vượt toàn bộ release gate ở cuối tài liệu.

## 2. Trust boundary đích

Worker chỉ được kết luận về những gì nó có thể kiểm chứng bằng code:

| Câu hỏi | Model worker trả lời? | Chủ sở hữu |
|---|---:|---|
| Model có load và sẵn sàng không? | Có | Worker runtime |
| Request có hợp lệ và nằm trong resource envelope không? | Có | HTTP/service layer |
| Reasoning protocol có đi đúng state machine không? | Có | Native decoder |
| Final có đúng output contract không? | Có | Contract compiler/validator |
| Task có được giải đúng không? | Không | Caller hoặc harness sau này |
| Có được gọi tool hay mutation không? | Không có tool trong worker | Agent harness sau này |
| Có nên retry/replan không? | Không; worker chỉ trả `retryable` cho lỗi hạ tầng | Caller hoặc harness sau này |

Hệ quả bắt buộc:

- Xóa `accepted` khỏi production response.
- Xóa `expected_result` khỏi production request.
- Không diễn giải field do model sinh như `status=completed` thành trạng thái hệ thống.
- Phân biệt ba khái niệm: `termination`, `protocol_valid`, `output_valid`.
- Test semantic vẫn có thể biết đáp án, nhưng assertion phải nằm trong test runner, không nằm trong worker API.

## 3. Contract API v1

Tạo endpoint mới `POST /v1/model/generate`. Không âm thầm đổi nghĩa endpoint prototype hiện tại. `/v1/controlled/generate` chỉ được giữ dưới nhãn legacy trong thời gian migration và không phải release gate.

### 3.1 Request

```json
{
  "protocol_version": "model-worker.v1",
  "model_id": "qwen35-9b-local",
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."}
  ],
  "output_contract": {
    "version": "structured-output.v1",
    "schema": {},
    "instructions": "Optional bounded instruction describing field semantics"
  },
  "limits": {
    "reasoning_tokens": 768,
    "final_tokens": 256,
    "total_tokens": 1024,
    "queue_timeout_ms": 5000,
    "execution_timeout_ms": 120000
  },
  "stream": {
    "enabled": false,
    "include_reasoning": false
  },
  "metadata": {
    "client_request_id": "optional-opaque-correlation-id"
  }
}
```

Quy tắc:

- `request_id` và `attempt_id` do server sinh; client ID không được dùng làm path.
- `model_id` phải khớp model đã load. V1 không tự đổi hoặc load model khác.
- `messages` chỉ cho phép role đã khai báo; mọi field và type đều được validate trước queue.
- Client chỉ được hạ limits so với manifest/server policy, không được nâng trần.
- Budget phải thỏa `reasoning_tokens <= total_tokens`, `final_tokens <= total_tokens` và `total_tokens <= reasoning_tokens + final_tokens`; mọi marker/EOG sampled đều tính vào total.
- `stream.enabled` và `include_reasoning` phải là boolean thật; không dùng coercion như `bool("false")`.
- `include_reasoning=false` là mặc định. Khi false, reasoning text không được stream hoặc persist.
- Schema phải được normalize/compile hoàn toàn trước khi request vào queue hoặc SSE gửi header.
- `output_contract.instructions` có byte cap và là nơi mô tả semantic field; worker không lờ JSON Schema keyword để lấy description.

### 3.2 Response

```json
{
  "protocol_version": "model-worker.v1",
  "request_id": "server-generated",
  "attempt_id": "server-generated",
  "termination": "completed",
  "protocol_valid": true,
  "output_valid": true,
  "output": {},
  "usage": {
    "prompt_tokens": 0,
    "reasoning_tokens": 0,
    "final_tokens": 0,
    "sampled_tokens": 0,
    "context_limit": 0,
    "context_headroom": 0
  },
  "timing": {
    "queue_ms": 0,
    "prompt_decode_ms": 0,
    "generation_ms": 0,
    "total_ms": 0
  },
  "model": {
    "id": "qwen35-9b-local",
    "manifest_digest": "sha256:...",
    "runtime_build": "b10012",
    "process_generation": 1
  },
  "error": null
}
```

`termination=completed` chỉ hợp lệ khi `protocol_valid=true` và `output_valid=true`. Một JSON đúng shape nhưng sai task vẫn có thể là output hợp lệ; worker tuyệt đối không gắn `accepted=true`.

### 3.3 Error taxonomy

Taxonomy tối thiểu:

| Code | Retryable | HTTP trước khi stream | Ý nghĩa |
|---|---:|---:|---|
| `invalid_request` | Không | 400 | JSON/type/field sai |
| `request_too_large` | Không | 413 | Vượt byte/message/schema cap |
| `unsupported_contract` | Không | 422 | Contract dùng keyword chưa hỗ trợ |
| `context_overflow` | Không | 422 | Prompt + output reserve vượt context |
| `queue_full` | Có | 429 | Hết slot queue |
| `queue_timeout` | Có | 408 | Hết deadline khi còn chờ |
| `cancelled` | Không | 409 | Caller/client disconnect yêu cầu hủy |
| `deadline_exceeded` | Có | 504 | Execution vượt deadline |
| `protocol_violation` | Có điều kiện | 422 | Marker/state reasoning sai |
| `output_invalid` | Có điều kiện | 422 | Final parse/contract validation fail |
| `decode_failed` | Có | 502 | llama decode lỗi |
| `worker_crashed` | Có | 502 | Native process chết |
| `worker_not_ready` | Có | 503 | Model chưa resident |
| `shutdown` | Có | 503 | Service đang drain |

Không tự replay request trong v1. Sau crash, service restart worker để nhận request kế tiếp và trả lỗi typed cho request đang chạy. Việc retry request thuộc caller.

## 4. Hai state machine cần tách riêng

### 4.1 Request lifecycle

```text
RECEIVED
  → PREFLIGHTED
  → QUEUED
  → RUNNING
  → COMPLETED | FAILED | CANCELLED | TIMED_OUT
```

Mọi transition phải được code kiểm tra, có timestamp và chỉ xảy ra một lần. Registry của request được tạo trước khi enqueue để request đang chờ cũng cancel được.

### 4.2 Model phase

Với manifest yêu cầu reasoning marker:

```text
EXPECT_REASONING_START
  → REASONING
  → FINAL
  → DONE
```

Các transition bị cấm:

- Gặp end marker khi chưa gặp start marker.
- EOG trước end marker.
- Start marker lặp lại.
- Quay lại reasoning sau khi vào final.
- Grammar active trước khi end marker được consume hoàn chỉnh.
- Final kết thúc khi grammar chưa ở accepting state.

Budget trước start marker vẫn phải được tính vào reasoning/total budget để không thể bypass bằng cách ở mãi state đầu. Manifest có thể khai báo `reasoning.mode=none`; worker không được tự suy đoán protocol từ output.

## 5. Kế hoạch triển khai theo phase

### Phase 0 — Baseline có thể tái tạo

Mục tiêu: biến thư mục lab thành source tree có version và build lặp lại được trước khi đổi behavior.

Thay đổi:

1. Khởi tạo top-level Git cho authored project; vendored llama.cpp trở thành submodule hoặc dependency pin rõ ràng.
2. Thêm `.gitignore` loại `build/`, `artifacts/`, `__pycache__/`, model và archive runtime.
3. Không commit hai bản CUDA/runtime trùng nhau. Runtime được fetch từ manifest có URL, version và SHA-256 hoặc được cấp qua path ngoài repo.
4. Thêm `pyproject.toml` với Python version, test/lint dependencies và entry point service.
5. Chuyển native build sang CMake target `model-worker-native`; không sinh `.def` động từ một DLL không được xác minh.
6. Ghi pin llama.cpp `b10012`, header/runtime ABI và hash của DLL dùng lúc build/run.
7. Tách cấu hình máy cá nhân thành `config/model.example.json`; path thật qua CLI/env/config local không commit.

File dự kiến:

- `.gitignore`
- `pyproject.toml`
- `CMakeLists.txt`
- `config/model.example.json`
- `scripts/fetch_runtime.ps1`
- `scripts/verify_runtime.py`

Test/gate:

- Checkout sạch có thể cài Python dependencies và build native target từ tài liệu.
- Runtime/header mismatch fail trước compile hoặc startup với lỗi rõ ràng.
- `rg` không thấy username, absolute model path hoặc token ID hardcode trong source production.
- Source tree không chứa model, ZIP CUDA hoặc generated binary.

### Phase 1 — Sửa trust boundary và API contract

Mục tiêu: worker chỉ báo inference/output validity, không giả làm task validator.

Thay đổi:

1. Tạo dataclass/Pydantic-equivalent nội bộ cho `GenerateRequest`, `GenerateResult`, `WorkerError` mà không dựa vào coercion ngầm.
2. Tạo endpoint `/v1/model/generate` versioned.
3. Loại `accepted` và `expected_result` khỏi production contract.
4. Đổi `validate_schema_subset` thành `validate_output_contract`; kết quả gồm danh sách lỗi có JSON path.
5. Không hardcode bốn field `status/result/evidence/reason` trong prompt builder.
6. `response_language` không thuộc worker. Caller thể hiện ngôn ngữ trong messages.
7. Preflight xong toàn bộ body/schema/limits trước khi enqueue và trước SSE headers.
8. Chuẩn hóa HTTP/error response; exception không được làm handler đóng connection im lặng.
9. Legacy endpoint có warning và bị loại khỏi release gate.

File dự kiến:

- `model_worker/contracts.py`
- `model_worker/http_api.py`
- `model_worker/errors.py`
- `model_worker/prompt.py`
- `controlled_inference/controlled_service.py` chỉ còn adapter legacy trong migration.

Test/gate:

- Output `status=blocked` không bao giờ trở thành system completion vì response không còn field `accepted`.
- Request list/string/null, `stream="false"`, integer request ID và field lạ đều nhận lỗi JSON xác định.
- Test runner vẫn kiểm tra được đáp án bên ngoài API.
- Mọi preflight failure chứng minh queue depth và model token count không đổi.

### Phase 2 — Model manifest và capability verification

Mục tiêu: bỏ toàn bộ assumption gắn với máy và model hiện tại khỏi code.

Manifest tối thiểu:

```json
{
  "manifest_version": "model-manifest.v1",
  "id": "qwen35-9b-local",
  "gguf_path": "...",
  "gguf_sha256": "...",
  "runtime_build": "b10012",
  "context": {"n_ctx": 4096, "n_batch": 1024, "n_ubatch": 512},
  "gpu": {"layers": 99},
  "sampling": {"profile": "greedy-v1"},
  "reasoning": {
    "mode": "required_marker_sequence",
    "start_text": "<think>",
    "end_text": "</think>",
    "require_start": true
  },
  "limits": {
    "input_bytes": 262144,
    "schema_bytes": 65536,
    "max_messages": 32,
    "max_reasoning_tokens": 1024,
    "max_final_tokens": 512,
    "max_total_tokens": 1536
  }
}
```

Thay đổi:

1. Resolve model/runtime/config từ CLI hoặc manifest, không từ constant Python/C++.
2. Hash manifest và trả digest trong mọi response/artifact.
3. Startup tokenize `start_text/end_text` với tokenizer thật; lưu token **sequence**, không giả định một token.
4. Verify start/end sequence khác nhau, không rỗng và chat template tồn tại.
5. Verify `n_ctx` với model metadata; rope scaling hoặc context vượt training context phải được khai báo rõ, không suy đoán.
6. Sampling profile do server/manifest sở hữu. V1 hỗ trợ `greedy-v1`; không cho client lắp sampler chain tùy ý.
7. Chạy startup protocol probe tùy chọn; readiness fail nếu capability không khớp manifest.
8. Request `model_id` khác model resident nhận lỗi, không trigger JIT load.

Test/gate:

- Đổi model path/config không cần rebuild C++.
- Marker một token và nhiều token đều đi qua cùng state controller test.
- Manifest sai hash, runtime build, marker hoặc context bị reject trước khi service ready.
- Health trả model ID, manifest digest, runtime build và process generation.

### Phase 3 — Native decoder correctness và context envelope

Mục tiêu: biến token loop thành module có invariant kiểm thử được, không phải logic gắn trong một hàm lớn.

Thay đổi:

1. Tách `ReasoningPhaseController` thuần khỏi llama sampling; input là token ID, output là transition/action/error.
2. Hỗ trợ marker sequence, không chỉ token ID đơn.
3. Bắt buộc start marker theo manifest; end-before-start là `protocol_violation`.
4. Tính mọi sampled token trước final vào reasoning/total budget, kể cả token trước start.
5. Chỉ switch grammar sau khi consume toàn bộ end sequence.
6. Kiểm tra EOG chỉ được chấp nhận khi phase final và grammar ở accepting state.
7. Kiểm tra return value của cả hai lần `llama_chat_apply_template` và mọi allocation/sampler init.
8. Tokenize prompt trước, tính:

   ```text
   prompt_tokens + reserved_generation_tokens + safety_margin <= n_ctx
   ```

9. Decode prompt theo chunk `<= n_batch`; cancellation/deadline được kiểm tra giữa các chunk.
10. Không đưa batch prompt lớn hơn `n_batch` vào một `llama_decode`.
11. Summary native chứa termination typed, phase cuối, marker counts, prompt/final usage và context headroom.
12. Dùng RAII cho context, sampler và output handle để mọi error path giải phóng tài nguyên.

File dự kiến:

- `native/reasoning_phase_controller.h/.cpp`
- `native/request_decoder.h/.cpp`
- `native/model_worker_main.cpp`
- `native/tests/test_reasoning_phase_controller.cpp`

Test/gate:

- Synthetic token tests bao phủ start đúng, missing start, end trước start, duplicate start, missing end, EOG sớm và marker nhiều token.
- Prompt dài hơn `n_batch` nhưng còn trong context phải chạy thành công.
- Prompt vượt context bị reject trước `llama_decode`.
- Boundary đúng tại các mép reasoning/final/total budget.
- Cancellation trong prompt ingestion và generation đều kết thúc có bound.
- Sanitizer/debug build không leak context/sampler trên mọi injected failure.

### Phase 4 — Output contract compiler fail-closed

Mục tiêu: grammar và validator không thể bất đồng hoặc lờ keyword.

Thiết kế:

1. Parse schema thành một normalized AST versioned.
2. Grammar compiler và validator cùng đọc AST này; không duy trì hai cách diễn giải độc lập.
3. `structured-output.v1` chỉ hỗ trợ subset được công bố:
   - Root object phẳng.
   - Tất cả property required và thứ tự canonical.
   - `string`, `integer`, `boolean`, `null`.
   - Union primitive với `null`.
   - Primitive enum có type tương thích.
   - `additionalProperties=false`.
4. Mọi keyword khác, gồm `maxLength`, `pattern`, bounds, array, nested object và optional field, phải fail `unsupported_contract`; tuyệt đối không bỏ qua.
5. `enum` bắt buộc có `type`; từng enum value phải validate với type lúc compile.
6. Reject duplicate/invalid types, empty enum, non-finite number, property/rule vượt cap.
7. JSON parser ở strict mode: reject `NaN`, `Infinity`, duplicate key và trailing data.
8. Canonicalize property order một lần; validator không phụ thuộc tình cờ vào insertion order từ caller.
9. Prompt chỉ nói về contract đã normalize hoặc dùng description từ caller; không hardcode field của schema cũ.
10. Tạo differential helper: feed serialized candidate qua llama grammar acceptor/reference recognizer rồi so kết quả với validator.
11. Chỉ thêm constraint mới ở version sau khi grammar compiler **và** validator cùng có test đối xứng.

File dự kiến:

- `model_worker/output_contract/ast.py`
- `model_worker/output_contract/parser.py`
- `model_worker/output_contract/gbnf.py`
- `model_worker/output_contract/validator.py`
- `tests/property/test_contract_equivalence.py`

Test/gate:

- Regression cho `maxLength` bị lờ và `enum` thiếu type.
- Fuzz malformed schema không được crash service.
- Property suite: mọi fixture grammar-valid phải validator-valid; mọi validator-invalid fixture không được coi `output_valid`.
- Unsupported contract fail trước queue/model activity.
- Ít nhất 10.000 generated schema/value cases trong CI không tìm thấy compiler/validator divergence.

### Phase 5 — Dispatcher, IPC, cancellation, timeout và watchdog

Mục tiêu: một request treo hoặc disconnect không được khóa service vô hạn.

Thay đổi service:

1. Thay `worker_lock.acquire()` trong HTTP thread bằng explicit bounded queue và một dispatcher thread.
2. Tạo request registry trước enqueue với lifecycle, cancel event và absolute deadlines.
3. Cancel request ở `QUEUED` bằng cách loại/tombstone queue item; không chờ nó thành active.
4. Client disconnect áp policy rõ: mặc định cancel request nếu không có consumer khác.
5. Tách `queue_timeout_ms`, `execution_timeout_ms` và watchdog grace.
6. Native process có control-reader thread hoặc control pipe riêng; frame `cancel` đặt atomic flag để inference loop quan sát dù data-plane đang bận generation.
7. Phân biệt process heartbeat với progress event. Heartbeat không được che decoder stall; absolute execution deadline luôn có quyền kill process.
8. Khi execution deadline hoặc progress deadline vượt: gửi cancel, chờ grace, kill worker, đánh dấu typed error và restart process.
9. Graceful shutdown: ngừng nhận request, cancel/drain theo policy, shutdown worker, có hard deadline.

Thay đổi IPC:

1. Bỏ tab-delimited protocol; dùng NDJSON hoặc length-prefixed JSON có `protocol_version`, `request_id`, `attempt_id`, `sequence`.
2. Worker phát frame `ready`, `started`, `phase`, `final_delta`, `heartbeat`, `completed`, `failed`.
3. Service verify response ID, attempt ID và sequence; desync là worker protocol failure.
4. Không dùng việc reread toàn bộ `tokens.jsonl` mỗi 15 ms làm transport streaming.
5. Artifact writer subscribe frame và ghi buffered/asynchronous; live transport không phụ thuộc disk.
6. Reasoning text frame chỉ phát khi request và server policy cùng cho phép.

File dự kiến:

- `model_worker/dispatcher.py`
- `model_worker/request_registry.py`
- `model_worker/worker_process.py`
- `model_worker/ipc.py`
- `native/ipc_protocol.h/.cpp`

Test/gate:

- Cancel queued trả trong bound mà model không nhận request.
- Cancel running được quan sát trong một prompt chunk/token iteration; watchdog xử lý trường hợp native không phản hồi.
- Fake worker hang, crash, gửi JSON lỗi, sai ID, sai sequence hoặc thiếu completion đều không khóa request sau.
- Queue full/timeout không double-count metrics.
- SSE preflight error dùng HTTP status đúng; error sau header dùng SSE `error` event hợp lệ.
- Sau crash, readiness chỉ về `ready` sau khi model load lại hoàn tất.

### Phase 6 — Resource envelope và HTTP security

Mục tiêu: mọi request có trần CPU/RAM/VRAM/disk/time trước khi chạm model.

Thay đổi:

1. Giới hạn Content-Length, message count, từng message bytes, tổng input bytes, schema bytes, property count và queue length.
2. Giới hạn min/max cho mọi budget; server clamp theo manifest, không tin client.
3. Đặt socket read/header timeout và giới hạn concurrent HTTP handlers.
4. Server tự sinh UUID/ULID path-safe. `client_request_id` chỉ là metadata có length cap và không dùng làm filename.
5. Mặc định bind loopback. Startup phải từ chối non-loopback nếu chưa có TLS termination **và** bearer authentication/reverse-proxy trust mode rõ ràng.
6. `/metrics` và debug endpoints tuân cùng exposure policy; test-only crash hook không có trong production build/config.
7. Không log authorization header, full prompt hoặc raw schema ở application log.
8. Rate limit đơn giản theo token/client chỉ cần khi bật external mode; local mode vẫn có queue/body/resource cap.

Test/gate:

- Oversized body nhận 413 mà không parse/ghi disk/enqueue.
- Slow/incomplete body bị đóng theo timeout, không giữ thread vô hạn.
- Unicode/reserved-name/siêu dài client ID không ảnh hưởng path.
- Non-loopback startup không auth phải fail closed.
- Fuzz HTTP body/type không tạo traceback uncaught hoặc làm health fail.

### Phase 7 — Artifact, privacy và observability

Mục tiêu: có bằng chứng điều tra được mà không biến reasoning log thành data leak hoặc disk DoS.

Artifact layout:

```text
artifacts/
  YYYY-MM-DD/
    <request_id>/
      <attempt_id>/
        manifest.json
        result.json
        events.jsonl        # optional, redacted
```

Thay đổi:

1. Request/attempt directory immutable; không reuse hoặc overwrite.
2. `manifest.json` chứa request hash, contract hash, model manifest digest, runtime build, limits và timestamps; không mặc định chứa raw prompt.
3. `result.json` ghi atomic bằng temp + rename và chứa terminal state duy nhất.
4. Token/reasoning log mặc định tắt; debug opt-in có redaction và retention ngắn.
5. Thiết lập max artifact bytes/request, tổng disk quota, retention TTL và cleanup job.
6. Artifact root và file dùng ACL chỉ cho service account/user hiện tại; cleanup phải resolve path và không follow symlink/reparse point ra ngoài root.
7. Buffer event writes; không flush/re-read toàn bộ file mỗi token.
8. Liveness và readiness tách riêng.
9. Metrics tối thiểu:
   - Request count theo termination/error class.
   - Queue depth/wait histogram.
   - Prompt decode, generation và total latency histogram.
   - Prompt/reasoning/final token count.
   - Prompt/decode tokens per second.
   - Model loads, worker restarts và watchdog kills.
   - Context headroom.
10. Không đưa request/attempt/client ID vào metric labels để tránh cardinality explosion.
11. Structured service log có request/attempt correlation nhưng không chứa chain-of-thought.

Test/gate:

- Reuse cùng `client_request_id` tạo hai immutable request/attempt khác nhau.
- Crash giữa lúc ghi không tạo `result.json` giả hoàn tất.
- `include_reasoning=false` không để reasoning text xuất hiện ở SSE, log hoặc artifact.
- Quota/TTL cleanup không xóa active attempt và không đi ra ngoài artifact root.
- Streaming 1.000 token không có tăng I/O O(n²).

### Phase 8 — Test architecture và reliability matrix

Mục tiêu: thay số đếm “8/8, 10/10” bằng coverage gắn với invariant và failure mode.

Tầng test:

1. **Unit Python**: request parsing, limits, error mapping, registry, queue, artifact paths, contract AST/validator.
2. **Unit native**: phase controller, marker sequence, budgets, UTF-8 accumulator và termination mapping.
3. **Property/fuzz**: schema parser/compiler/validator, malformed IPC/HTTP, token state sequences.
4. **Integration fake worker**: timeout, hang, crash, protocol desync, queued/running cancellation, shutdown.
5. **Native integration**: prompt chunking, context preflight và grammar activation với test model/runtime.
6. **GPU model suite**: model hiện tại, tiếng Anh/Việt/Unicode, không ghi đáp án vào prompt.
7. **Soak**: hàng trăm request tuần tự, crash/restart xen kẽ, theo dõi VRAM/RAM/file descriptors/disk.

Các case bắt buộc:

- Output schema-valid nhưng semantic sai: worker trả `output_valid=true`, test runner đánh semantic fail; chứng minh trust boundary đúng.
- Model trả `blocked` hoặc `completed` không được đổi thành system acceptance.
- Missing start, end-before-start, missing end, duplicate marker, EOG sớm.
- Prompt `> n_batch` nhưng `< n_ctx`; prompt vượt context; budget vượt cap.
- Unsupported schema keyword, enum/type mismatch, duplicate JSON key và invalid UTF-8.
- Body sai top-level type, coercion boolean, oversized/slow body.
- Duplicate client correlation ID và path edge cases.
- Cancel queued, prompt-decode, reasoning, final và client disconnect.
- Worker hang, crash trước ready, crash giữa request, malformed response, restart fail.
- 100+ sequential contexts cho kết quả độc lập; task không tiết lộ đáp án trong prompt.

Target command sau khi cấu trúc test được tạo:

```powershell
python -m pytest tests/unit tests/property
python -m pytest tests/integration -m "not gpu"
python -m pytest tests/gpu --model-manifest config/model.local.json
python scripts/soak_worker.py --model-manifest config/model.local.json --requests 500
```

Gate:

- Không merge khi unit/property/integration fake-worker fail.
- GPU suite là release gate cho từng model manifest được support.
- Soak không có state leakage, worker deadlock hoặc tăng tài nguyên đơn điệu sau warmup.
- Timeout/cancellation hoàn tất trong deadline + watchdog grace đã cấu hình.
- Test report nhóm theo invariant/error class, không chỉ tổng số case pass.

### Phase 9 — Release packaging và operational gate

Mục tiêu: phát hành một worker v1 có contract ổn định, không phát hành lab scripts như production entrypoint.

Thay đổi:

1. Entry point duy nhất cho production service và một CLI `validate-manifest`.
2. Version API, IPC, manifest và output contract độc lập.
3. Startup self-check: runtime hash/ABI, model hash, template, markers, context, artifact directory và port exposure.
4. Graceful shutdown/drain được tài liệu hóa.
5. Viết operator runbook cho model load failure, watchdog restart, disk quota và GPU OOM.
6. Legacy Phase A–G chuyển vào `experiments/` hoặc archive; không được import bởi production package.
7. Benchmark baseline latency, prompt throughput, generation throughput, peak VRAM/RAM và restart time.

Release gate cuối:

- Checkout/build/start từ tài liệu trên máy sạch phù hợp.
- Không còn absolute user path, hardcoded marker token hoặc test-only endpoint trong production config.
- Mọi request có finite byte/token/time/disk bounds.
- Không có đường code nào trả `accepted` hoặc semantic task completion.
- Phase controller fail closed trên mọi protocol violation đã liệt kê.
- Grammar compiler và validator dùng cùng normalized contract.
- Queue, cancellation, watchdog và recovery đã qua fault-injection suite.
- Reasoning không stream/persist mặc định.
- Artifact immutable, atomic, có retention/quota.
- Unit/property/fake-worker/GPU/soak gates đều có evidence mới từ cùng commit/runtime manifest.

## 6. Thứ tự và dependency bắt buộc

```text
Phase 0: reproducible baseline
  ↓
Phase 1: trust boundary + API
  ↓
Phase 2: model manifest
  ↓
Phase 3: native protocol/context correctness
  ↓
Phase 4: output contract fail-closed
  ↓
Phase 5: dispatcher/IPC/deadline/watchdog
  ↓
Phase 6: resource/security envelope
  ↓
Phase 7: artifact/observability
  ↓
Phase 8: full reliability evidence
  ↓
Phase 9: release gate
```

Không bắt đầu tool calling giữa các phase. Đặc biệt, không xây agent loop sau Phase 4 chỉ vì structured output đã ổn; serving và failure containment ở Phase 5–8 vẫn là điều kiện của model worker.

## 7. Traceability từ review sang kế hoạch

| Finding đã nêu | Phase xử lý | Bằng chứng đóng finding |
|---|---:|---|
| False acceptance, `blocked/null/empty evidence` vẫn accepted | 1 | Production API không còn `accepted/expected_result`; semantic assertion ở caller test |
| Chưa có agent/tool loop | Ngoài scope | Tài liệu và package worker không chứa tool executor; agent bị hoãn rõ ràng |
| End marker có thể bypass start marker | 2–3 | Manifest + pure FSM reject end-before-start/missing-start |
| Worker hang khóa service, queued request không cancel được | 5 | Explicit dispatcher/registry, deadlines, heartbeat và watchdog fault tests |
| Body/budget/context không có trần | 3, 6 | Preflight formula, prompt chunking, byte/token/time caps và 413/422 tests |
| Schema bỏ qua constraint, enum thiếu type làm crash | 4 | Normalized AST fail-closed + regression/property/fuzz suite |
| Model path/token IDs/GPU layers hardcode | 0, 2 | Manifest/config và startup capability verification |
| Reasoning bị stream/persist; file tail O(n²) | 5, 7 | IPC frames, buffered artifact writer, reasoning off mặc định |
| Request ID reuse phá audit | 1, 7 | Server-generated request/attempt ID và immutable directories |
| Test happy path tiết lộ đáp án | 8 | GPU semantic suite không chứa đáp án, invariant/fault matrix |
| Không top-level Git/dependency/CI, repo chứa binary lớn | 0 | Clean source checkout, pinned dependencies, generated assets ignored |
| Thiếu auth/rate/resource controls | 6 | Loopback fail-closed hoặc authenticated external mode + caps |
| Metrics quá yếu | 7 | Histograms, error classes, usage/headroom/restart metrics |
| Chưa multi-model/batching/KV reuse | Hoãn sau v1 | Không phải correctness gate của single-model worker; fresh context giữ nguyên |

## 8. Những việc cố ý hoãn sau Model Worker v1

Chỉ xem xét sau khi Phase 9 pass:

1. Multi-model registry và load/unload scheduling.
2. Continuous batching và nhiều context song song.
3. Prompt cache hoặc KV reuse có session lifecycle.
4. Speculative decoding và tuning sampler.
5. OpenAI-compatible facade rộng hơn.
6. Tool-use action schema và agent loop.
7. Task state, permission gate, retry/replan, planner/reviewer và multi-agent.

Thứ tự sau v1 phải là: đo correctness/throughput của worker trước, rồi mới quyết định batching/cache; xây agent chỉ sau khi worker contract và failure semantics đã ổn định.

## 9. Definition of Done

Project chỉ được đổi trạng thái từ “controlled-inference spike” sang “Model Worker v1” khi có evidence cùng một revision chứng minh toàn bộ:

- Contract API/IPC/manifest/output-contract được version hóa.
- Không còn semantic task acceptance trong worker.
- Model capability được khai báo và verify lúc startup.
- Reasoning FSM fail closed, không bypass marker/budget.
- Prompt ingestion đúng `n_batch`, context được preflight.
- Schema/compiler/validator fail closed và không divergence trong property suite.
- Queue/cancel/deadline/watchdog/crash recovery có bound và fault tests.
- HTTP/resource/security envelope hữu hạn.
- Reasoning private mặc định; artifact immutable, atomic và có retention.
- Metrics đủ để phân biệt queue, model load, prompt decode, generation và failure class.
- Clean build, unit, property, integration, GPU và soak đều pass.

Cho tới lúc đó, tên đúng của code hiện tại vẫn là **controlled-inference prototype**, không phải agent harness core và chưa phải production model worker.
