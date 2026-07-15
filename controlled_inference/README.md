# Controlled inference spike

## Phase A — token observation

Mục tiêu: quan sát riêng luồng thinking và final khi chưa bật grammar.

Chạy:

```powershell
python .\controlled_inference\phase_a_observe.py
```

Artifacts:

- `artifacts/phase-a-observation.json`: delta, thời gian, phase và token ID.
- `artifacts/phase-a-server.log`: log của llama-server.

Giới hạn có chủ ý: API `deepseek` đã bóc và loại marker kết thúc reasoning trước khi trả HTTP. Phase A xác nhận boundary ở lớp parser. Muốn bắt chính xác control token trong vòng sampling phải chuyển sang C API ở Phase B.

### Kết quả lần chạy đầu

- Thinking: 124 token.
- Final: 29 token.
- Chỉ có một chuyển phase: event 122, `thinking → final`.
- Token marker theo tokenizer của model: `<think>` = `248068`, `</think>` = `248069`.
- Delta cuối của thinking là newline token `198`.
- Delta đầu của final là code-fence token `71093`.
- Marker `248069` không xuất hiện trong HTTP output vì chat parser đã tiêu thụ nó.

Kết luận: đã quan sát được ranh giới logic ở output parser. Chưa chứng minh được token `248069` chính là tín hiệu chuyển state trong sampling loop; đó là tiêu chí mở của Phase B/C API.

## Phase B — direct C API boundary

`phase_b_sample.cpp` gọi trực tiếp `llama.dll`, greedy-sample từng token và ghi state trước/sau token vào JSONL. Không dùng HTTP server hoặc reasoning parser.

Kết quả:

- Token đầu tiên: `248068 = <think>`, state `START → THINKING`.
- Token 142: `248069 = </think>`, state `THINKING → FINAL`.
- Token 143 trở đi được quan sát trong state `FINAL`.
- Có đúng một reasoning boundary trong lần chạy.
- Tổng cộng 160 sampled token và kết thúc bằng `<|im_end|>`.

Artifacts:

- `artifacts/phase-b-tokens.jsonl`: log gốc từng sampled token.
- `artifacts/phase-b-observation.json`: kết quả acceptance đã rút gọn.
- `artifacts/phase-b-stderr.log`: log runtime.

### GPU verification

Runtime đóng gói của LM Studio bị access-violation khi DLL được gọi ngoài host của nó. Spike đã chuyển sang llama.cpp official `b10012` và build lại theo header/ABI cùng release.

- Native exit code: `0`.
- Offload: `33/33` layer lên GPU.
- CUDA model buffer: `4810.28 MiB`.
- CUDA KV buffer: `128.00 MiB`.
- CUDA compute buffer: `505.02 MiB`.
- 198 sampled token trong khoảng 2.25 giây generation loop.
- Boundary `248069 = </think>` vẫn được bắt trực tiếp và state chuyển sang `FINAL`.

Artifact xác minh: `artifacts/phase-b-gpu-observation.json`.

## Phase C — switch grammar after thinking

Controller dùng hai sampler được khởi tạo độc lập:

1. `free greedy` trong state `THINKING`.
2. `fixed JSON grammar + greedy` trong state `FINAL`.

Token `248069 = </think>` kích hoạt việc đổi sampler. Grammar cố định chỉ cho phép object có dạng `{"result":<integer>}`.

Kết quả GPU đầu tiên:

- Thinking kết thúc tại token 140.
- Sampler switch xảy ra ngay sau token 140.
- Tất cả token thinking có `grammar_active=false`.
- Tất cả token final có `grammar_active=true`.
- Final chính xác: `{"result":2}`.
- Không Markdown, không whitespace thừa, không field thừa.
- Native exit code `0`, full GPU `33/33` layer.

Chạy lại bằng `run_phase_c.ps1`. Artifact acceptance: `artifacts/phase-c-observation.json`.

## Phase D — JSON Schema subset compiler

`compile_schema_subset.py` chuyển JSON Schema hẹp sang GBNF. Controller đọc grammar artifact lúc runtime; grammar không còn viết cứng trong C++.

Subset hiện tại:

- Root là object phẳng, không lồng object/array.
- Tất cả properties phải nằm trong `required` và đúng thứ tự khai báo.
- Hỗ trợ `string`, `integer`, `boolean`, `null`.
- Hỗ trợ `enum` và union một primitive với `null`.
- Bắt buộc `additionalProperties=false`.

Schema thử nghiệm là contract Atomic Worker gồm `status`, `result`, `evidence`, `reason`. Kết quả final được grammar ép đúng schema và deterministic validator kiểm tra lại acceptance task.

Chạy toàn bộ: `run_phase_d.ps1`. Artifact acceptance: `artifacts/phase-d-observation.json`.

### Quy ước UTF-8 và token log

- `token_id` là trường chuẩn trong token log.
- `piece_display` chỉ để con người debug; nếu một piece riêng chưa tạo thành UTF-8 hợp lệ thì ghi `null`.
- Raw bytes chỉ tồn tại tạm trong bộ nhớ decoder, không lưu thành artifact riêng.
- Final được tạo bằng cách ghép toàn bộ bytes trước, sau đó mới kiểm tra và decode UTF-8.
- JSON parser và schema validator chỉ đọc `final_text` đã ghép hoàn chỉnh, không ghép lại nội dung từ token log.

Test tiếng Việt đã pass với evidence chứa đầy đủ dấu Unicode.

### Ngôn ngữ phản hồi

Ngôn ngữ không nằm trong decoder, grammar hoặc schema. Mỗi task config khai báo:

- `response_language: "en"`: yêu cầu English.
- `response_language: "vi"`: yêu cầu tiếng Việt.
- `response_language: "auto"`: dùng cùng ngôn ngữ với yêu cầu người dùng.

`prepare_phase_d_prompt.py` chuyển policy này thành prompt artifact. C++ schema-mode chỉ đọc system/user prompt từ file nên không bị gắn với ngôn ngữ cụ thể.

Ma trận GPU `en`, `vi`, `auto_en`, `auto_vi` đều pass. Xem `artifacts/phase-d-language-matrix.json`.

## Phase E — stress test

Decoder có termination contract và exit code riêng:

- `completed` → `0`.
- `cancelled` → `20`.
- `reasoning_budget_exhausted` → `21`.
- `final_budget_exhausted` → `22`.
- `missing_reasoning_boundary` → `23`.
- `total_budget_exhausted` → `24`.
- Decode/internal failure → `25`.

Stress suite GPU gồm 8 case:

1. Thiếu reasoning boundary.
2. Hết reasoning budget.
3. Hết final budget.
4. Unicode tiếng Việt.
5. JSON string có quote, backslash và newline.
6. Prompt injection yêu cầu phá contract và đổi kết quả.
7. Model được yêu cầu thêm field ngoài schema.
8. Cancellation giữa generation.

Kết quả hiện tại: `8/8 passed`, không crash và không timeout. Chạy bằng:

```powershell
python .\controlled_inference\run_phase_e.py
```

Artifact tổng hợp: `artifacts/phase-e-stress-results.json`.

## Phase F — service hóa

Prototype một worker đã hoàn thành:

- `POST /v1/controlled/generate`: inference non-stream hoặc SSE.
- `POST /v1/controlled/cancel/{request_id}`: yêu cầu cancellation.
- `GET /health`: readiness, worker busy và queue depth.
- `GET /metrics`: Prometheus text metrics.
- Một active request và tối đa một waiting request; request thứ ba nhận HTTP 429.
- Mỗi worker là process riêng; worker crash không làm chết service.
- Streaming tách event `thinking`, `final`, `result`.

Chạy service:

```powershell
.\controlled_inference\run_phase_f.ps1
```

Ví dụ request:

```powershell
$body = @{
  task = 'Đếm nhãn bắt đầu bằng A: Alpha, beta, atlas, Gamma. Kết quả là 2.'
  response_language = 'auto'
  expected_result = 2
  stream = $false
} | ConvertTo-Json

Invoke-RestMethod http://127.0.0.1:8090/v1/controlled/generate `
  -Method Post -ContentType 'application/json' -Body $body
```

Integration test:

```powershell
python .\controlled_inference\test_phase_f_service.py
```

Kết quả hiện tại: 10/10 checks pass, bao gồm cancellation qua API. Artifact: `artifacts/phase-f-service-results.json`.

Phase F hiện tiếp tục pass regression test 10/10 sau khi service chuyển sang worker lâu dài.

## Phase G — persistent worker, nhiều request tuần tự

Service dùng một process C++ giữ model trong VRAM. Mỗi request dùng cùng model object đã load, nhưng tạo context và sampler mới. Sau generation, context được hủy để KV không rò sang request sau.

Worker chết không làm chết HTTP service. Service khởi động worker mới và load lại model; `process_generation` và `model_loads_total` cho phép quan sát recovery.

Chạy test:

```powershell
.\controlled_inference\run_phase_g.ps1
```

Kết quả hiện tại: 10/10 checks pass trên ba request tuần tự Anh → Việt → Anh:

- Cùng `process_generation=1`.
- `model_loads_total=1` trong cả ba request.
- `request_ordinal=1,2,3`.
- `context_fresh=true` cho từng request.
- Kết quả độc lập `2,3,1`; không lẫn state.
- Model vẫn resident sau request cuối.

Artifact: `artifacts/phase-g-persistent-results.json`.

Giới hạn: hiện chỉ chạy một request tại một thời điểm, queue tối đa một request chờ. Chưa có continuous batching, context reuse, prompt cache hoặc nhiều model.
