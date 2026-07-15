# Controlled inference — trạng thái và handoff

Ngày cập nhật: 2026-07-15.

> **Trạng thái tài liệu:** archive/handoff của prototype Phase A–G. Các
> invariant controlled inference vẫn có giá trị, nhưng thứ tự triển khai ở cuối
> file không còn là roadmap hiện hành. Đọc `06-model-worker-v1-plan.md` cho
> baseline worker, `07-inference-runtime-and-agent-roadmap.md` cho roadmap và
> `09-inference-runtime-implementation-handoff.md` cho session tiếp theo.

## Mục tiêu

Xây một lớp controlled inference mỏng trên llama.cpp cho agent harness:

```text
THINKING: sampling tự do
→ bắt token kết thúc reasoning
→ FINAL: bật grammar sinh từ JSON Schema
→ parse JSON
→ deterministic schema validation
→ caller/harness thực hiện task acceptance bên ngoài model worker
```

Mục tiêu không phải xây lại toàn bộ LM Studio hoặc Ollama. llama.cpp tiếp tục phụ trách model inference; project xây phần kiểm soát phase và output contract. Semantic task acceptance thuộc caller/harness, không thuộc model worker.

## Quyết định đã chốt

- Reasoning phase vẫn được generate và đếm để kiểm soát protocol; nội dung
  reasoning không stream hoặc persist mặc định.
- Grammar không được tác động lên thinking.
- Chỉ final content được ép schema.
- Chỉ chấp nhận final content; dữ liệu nằm riêng trong reasoning không được tính là output.
- `token_id` là dữ liệu chuẩn trong token log; `piece_display` chỉ phục vụ debug.
- Raw bytes chỉ tồn tại trong bộ nhớ, không lưu thành artifact riêng.
- Ghép toàn bộ bytes của final trước, sau đó mới validate/decode UTF-8.
- JSON parser và validator chỉ đọc final đã ghép, không tái dựng final từ token log.
- Schema, grammar và validator không phụ thuộc ngôn ngữ.
- Task có `response_language: auto | vi | en`; mặc định dự kiến là `auto`.
- Các key contract giữ ổn định bằng tiếng Anh, ví dụ `status`, `result`, `evidence`, `reason`.

## Runtime hiện tại

- Engine: llama.cpp official `b10012`.
- Model: `Qwen3.5-9B.Q4_K_M.gguf`.
- GPU: NVIDIA GeForce RTX 3080 10 GB.
- Full offload: `33/33` layer.
- CUDA model buffer quan sát được: `4810.28 MiB`.
- Runtime DLL đóng gói của LM Studio không được dùng vì access violation khi gọi trực tiếp ngoài host LM Studio.

## Kết quả các phase

### Phase A — HTTP/token observation

- Quan sát được một lần chuyển `thinking → final` ở lớp parser.
- Xác định candidate token: `<think> = 248068`, `</think> = 248069`.
- HTTP parser tiêu thụ reasoning marker, nên chưa đủ để điều khiển sampling.

### Phase B — direct llama.cpp C API

- Bắt trực tiếp token `248069 = </think>` trong sampling loop.
- State chuyển `THINKING → FINAL` trên chính token đó.
- Chạy full GPU, native exit code `0`.

### Phase C — fixed grammar switch

- Thinking dùng free greedy sampler.
- Sau `</think>`, controller đổi sang fixed JSON grammar + greedy.
- Final bị ép chính xác thành `{"result":2}`.
- Grammar không hoạt động ở bất kỳ thinking token nào.

### Phase D — JSON Schema subset compiler

- `compile_schema_subset.py` sinh GBNF từ JSON Schema.
- C++ controller đọc grammar artifact lúc runtime; schema không viết cứng trong decoder.
- Contract Atomic Worker `status/result/evidence/reason` đã pass.
- Unit test compiler: 4/4 pass.
- Ma trận ngôn ngữ GPU `en`, `vi`, `auto_en`, `auto_vi`: 4/4 pass.

Artifact tổng hợp ngôn ngữ: `controlled_inference/artifacts/phase-d-language-matrix.json`.

### Phase E — stress test

Trạng thái: prototype suite đã pass `8/8` trên GPU.

| Case | Expected termination | Exit | Kết quả |
|---|---|---:|---|
| Boundary thiếu | `missing_reasoning_boundary` | 23 | Pass |
| Hết reasoning budget | `reasoning_budget_exhausted` | 21 | Pass |
| Hết final budget | `final_budget_exhausted` | 22 | Pass |
| Unicode tiếng Việt | `completed` | 0 | Pass |
| Quote/backslash/newline escape | `completed` | 0 | Pass |
| Prompt injection | `completed`, result vẫn là 2 | 0 | Pass |
| Yêu cầu thêm field | `completed`, field bị grammar chặn | 0 | Pass |
| Cancellation giữa generation | `cancelled` | 20 | Pass |

Chi tiết quan sát:

- Boundary thiếu kết thúc sau 308 sampled token, grammar chưa bật.
- Reasoning budget 16 dừng đúng ở 16 thinking token.
- Final budget 3 dừng đúng ở 3 final token, grammar đã bật.
- Escape string round-trip đúng quote, backslash và newline sau JSON parse.
- Prompt injection không đổi `result=2`, không làm lộ reasoning vào final.
- Yêu cầu field `debug` bị chặn; final vẫn chỉ có 4 field contract.
- Cancellation flag được phát hiện giữa generation và dừng sạch sau 23 sampled token.
- Tất cả case dùng full GPU `33/33`; không crash hoặc timeout.

Artifact tổng hợp: `controlled_inference/artifacts/phase-e-stress-results.json`.

### Phase F — service hóa

Trạng thái: single-worker service prototype đã hoàn thành, integration test `10/10` pass.

Đã có:

- `POST /v1/controlled/generate` cho non-stream và SSE streaming.
- SSE event tách riêng `thinking`, `final`, `result`.
- `POST /v1/controlled/cancel/{request_id}`.
- `GET /health` trả readiness, busy state và queue depth.
- `GET /metrics` theo Prometheus text format.
- Một active request và tối đa một waiting request; request thứ ba nhận HTTP 429.
- Worker C++ chạy trong process riêng.
- Worker crash trả HTTP 502 nhưng service tiếp tục sống.
- Request thực sau worker crash đã chạy thành công.
- Schema unsupported và budget không hợp lệ fail trước khi vào queue.

Integration checks đã pass:

1. Health ready.
2. Queue capacity đúng một waiting request.
3. Streaming có đủ thinking/final/result.
4. Ghép final stream thành JSON hoàn chỉnh đúng bằng final result.
5. Queued request hoàn tất.
6. Worker crash được cô lập.
7. Service còn sống sau crash.
8. Request sau crash thành công.
9. Metrics phản ánh queue rejection và worker crash.
10. Cancellation qua API dừng worker với termination `cancelled` và metrics được cập nhật.

Phase F tiếp tục pass regression test 10/10 sau khi backend chuyển sang persistent worker ở Phase G.

### Phase G — persistent model, nhiều request tuần tự

Trạng thái: GPU integration test `10/10` pass.

```text
HTTP service
→ queue: 1 active + 1 waiting
→ persistent C++ worker process
→ model object giữ trong VRAM
→ request 1: context mới → generate → free context
→ request 2: context mới → generate → free context
→ ...
```

Model weights chỉ load một lần trong một worker generation. Context mới là ranh giới cô lập request, vì vậy KV và token history không được reuse ngầm.

Ba request test tuần tự trả kết quả `2 → 3 → 1` với:

- `process_generation`: `1,1,1`.
- `model_loads_total`: `1,1,1`.
- `request_ordinal`: `1,2,3`.
- `context_fresh=true` cho cả ba.
- Tiếng Việt round-trip không có replacement character.
- Model vẫn resident sau request cuối.

Thời gian generation quan sát được, không tính model load lúc service startup: `8.562s`, `7.391s`, `5.109s`.

Worker vẫn là process riêng. Nếu crash, request hiện tại fail; service tạo worker generation mới và load model lại. Phase F regression xác nhận crash isolation, recovery, streaming, queue và cancellation vẫn pass.

Artifact: `controlled_inference/artifacts/phase-g-persistent-results.json`.

Giới hạn còn lại: chỉ xử lý tuần tự; chưa continuous batching, prompt cache/KV reuse, timeout service-level đầy đủ hoặc multi-model scheduling.

## Phạm vi JSON Schema hiện hỗ trợ

- Root là object phẳng.
- Tất cả properties bắt buộc và theo thứ tự khai báo.
- `string`, `integer`, `boolean`, `null`.
- `enum`.
- Union một primitive với `null`.
- Bắt buộc `additionalProperties=false`.

Chưa hỗ trợ:

- Optional property.
- Array.
- Object lồng nhau.
- `number`/floating point.
- `oneOf`, `anyOf`, `$ref`.
- `minLength`, `maxLength`, numeric bounds, regex pattern.
- Object property không cố định thứ tự.

## Ngôn ngữ

Pipeline tiếng Anh và tiếng Việt dùng cùng kiến trúc:

```text
UTF-8 input
→ model chat template
→ tokenizer
→ token IDs
→ llama.cpp inference
→ ghép final bytes
→ decode UTF-8
→ parse/validate JSON
```

Ngôn ngữ chỉ ảnh hưởng prompt và nội dung string, không ảnh hưởng grammar/state machine. `prepare_phase_d_prompt.py` chuyển `response_language` thành prompt artifact. C++ schema-mode không chứa prompt tiếng Anh hoặc tiếng Việt.

`auto` hiện dựa vào model làm theo chỉ dẫn “dùng cùng ngôn ngữ với yêu cầu”. Đây chưa phải language detector tất định.

## So với LM Studio và Ollama

### Không thua đáng kể trong phạm vi hẹp hiện tại

- Cùng dùng tokenizer/model/llama.cpp cho GGUF nên không có engine tiếng Việt riêng.
- Full GPU inference đã hoạt động.
- Thinking và final được tách ở sampling loop.
- Final JSON được grammar ép sau reasoning boundary.
- Có grammar và shape/type validation bước đầu; chưa có deterministic semantic task acceptance.
- Với contract `thinking tự do → final constrained`, ta kiểm soát sâu hơn API tổng quát của LM Studio/Ollama trên đúng model đã thử.

### Còn thua xa về serving và vận hành production

- Chỉ thử một model, một reasoning protocol và một GPU.
- Token reasoning boundary đang gắn với model Qwen hiện tại.
- Chưa có model registry/capability detection.
- Persistent single-model worker, HTTP API, SSE, queue nhỏ và cancellation đã có ở mức prototype.
- Chưa có concurrency hoặc continuous batching; hiện cố ý chạy tuần tự.
- Context isolation đã có bằng fresh context; timeout service-level vẫn chưa hoàn chỉnh.
- Chưa có KV-cache/session lifecycle.
- Chưa có model download, load/unload, keep-alive, JIT load hoặc auto-evict.
- Chưa có multi-model/multi-GPU scheduling.
- Chưa có tool calling, MCP, embeddings, vision hoặc chat state.
- Chưa có authentication, rate limit, health supervision và automatic restart.
- Chưa có metrics production: latency percentiles, tokens/s, VRAM, queue depth, failure classes.
- Chưa có compatibility suite trên nhiều model/template/tokenizer.
- Schema compiler còn rất hẹp.

LM Studio hiện cung cấp model management, REST/OpenAI/Anthropic-compatible APIs, structured output, stateful chat, headless daemon, JIT loading và continuous batching. Ollama cung cấp API, structured output, thinking field, keep-alive, queue/concurrency và quản lý model lifecycle. Tham khảo:

- https://lmstudio.ai/docs/developer
- https://lmstudio.ai/docs/developer/openai-compat/structured-output
- https://lmstudio.ai/docs/app/advanced/parallel-requests
- https://docs.ollama.com/capabilities/structured-outputs
- https://docs.ollama.com/capabilities/thinking
- https://docs.ollama.com/faq

## Lệnh chạy lại

```powershell
cd D:\zalollm\agent-harness-lab
.\controlled_inference\run_phase_d.ps1
python .\controlled_inference\run_phase_e.py
.\controlled_inference\run_phase_f.ps1
python .\controlled_inference\test_phase_f_service.py
.\controlled_inference\run_phase_g.ps1
```

Lệnh trên:

1. Chạy unit test schema compiler.
2. Compile JSON Schema thành GBNF.
3. Build C++ controller.
4. Chạy bốn language cases trên GPU.
5. Validate từng case.
6. Ghi ma trận tổng hợp.

`run_phase_e.py` build decoder, chạy 8 stress case độc lập và ghi kết quả tổng hợp.

`run_phase_f.ps1` build decoder rồi chạy service tại `127.0.0.1:8090`. `test_phase_f_service.py` chạy integration test với service riêng tại port test.

## Artifacts quan trọng

- `controlled_inference/phase_b_sample.cpp`
- `controlled_inference/build_phase_b.ps1`
- `controlled_inference/compile_schema_subset.py`
- `controlled_inference/prepare_phase_d_prompt.py`
- `controlled_inference/analyze_phase_d.py`
- `controlled_inference/aggregate_phase_d_languages.py`
- `controlled_inference/phase_d_schema.json`
- `controlled_inference/phase_d_cases/`
- `controlled_inference/artifacts/phase-d.gbnf`
- `controlled_inference/artifacts/phase-d-language-matrix.json`
- `controlled_inference/artifacts/phase-e-stress-results.json`
- `controlled_inference/controlled_service.py`
- `controlled_inference/test_phase_f_service.py`
- `controlled_inference/artifacts/phase-f-service-results.json`
- `controlled_inference/persistent_worker.cpp`
- `controlled_inference/test_phase_g_persistent.py`
- `controlled_inference/artifacts/phase-g-persistent-results.json`
- `controlled_inference/README.md`

## Thứ tự triển khai sau này

### Bước 1 — làm sạch inference core

- Tách code Phase B/C/D khỏi file thử nghiệm thành module `model`, `prompt`, `sampler`, `phase_controller`, `output`.
- Không hardcode model path, token IDs hoặc GPU layers.
- Thêm manifest mô tả reasoning protocol theo model.
- Thêm budgets riêng cho thinking và final.

### Bước 2 — mở rộng schema compiler có kiểm soát

- Optional properties.
- Array và nested object.
- `number`, bounds và string constraints.
- Fuzz/property tests: mọi output grammar cho phép đều phải pass validator.
- Schema unsupported phải fail-fast trước khi load model.

### Bước 3 — persistent single-model service

- Đã có HTTP API tối thiểu, streaming reasoning/final, cancellation, health, metrics và single-worker queue.
- Đã chuyển từ process-per-request sang worker giữ model trong VRAM.
- Mỗi request tạo context mới và free sau khi xong; không reuse KV ngầm.
- Thêm timeout ở cấp service và graceful worker shutdown.
- Thiết kế prompt cache/KV/session lifecycle trước khi cho phép reuse context.

### Bước 4 — production controls

- Request queue và backpressure.
- Context/KV isolation.
- Metrics, tracing và audit log.
- Crash recovery, watchdog và graceful shutdown.
- Security/authentication nếu bind ra ngoài localhost.

### Bước 5 — compatibility và hiệu năng

- Nhiều model/template/reasoning protocol.
- Continuous batching khi correctness đã ổn định.
- Prompt cache/KV reuse.
- Benchmark latency, throughput, VRAM và schema reliability.

## Tiêu chí không được đánh đổi khi triển khai tiếp

- Không tắt thinking để làm structured output dễ hơn.
- Không áp grammar lên thinking.
- Không dùng reasoning làm final fallback.
- Không để model tự tuyên bố system completion.
- Không mở rộng schema/compiler mà thiếu deterministic validation.
- Không thêm concurrency trước khi có request/KV isolation.
- Không xây GUI/model catalog trước controlled inference service và harness contract.
