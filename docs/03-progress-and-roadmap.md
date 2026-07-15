# Tiến độ và roadmap

Ngày ghi nhận: 2026-07-14.

## Trạng thái hiện tại

Project: `D:\zalollm\agent-harness-lab`

Model thử nghiệm qua LM Studio:

```text
qwen3.5-9b-claude-4.6-opus-uncensored-distilled
```

### Đã hoàn thành

- [x] Thống nhất tầm nhìn local-first.
- [x] Phác thảo mô hình nhiều phòng ban/vai trò.
- [x] Tách trách nhiệm LLM và code tất định.
- [x] Xác định nguyên tắc Task Splitter không thực hiện task.
- [x] Xác định nguyên tắc Atomic Worker không tự chia việc.
- [x] Tạo project riêng.
- [x] Tạo skill `atomic-worker` bản đầu tiên.
- [x] Validate cấu trúc skill bằng `quick_validate.py`.
- [x] Tạo một atomic task có input và acceptance criteria.
- [x] Gọi model thật qua LM Studio OpenAI-compatible API.
- [x] Kiểm tra output bằng schema tất định sau khi nhận `content`.
- [x] Thêm deterministic validator cho task thử nghiệm.
- [x] Lưu task, output, usage và kết quả validation.
- [x] Chạy thành công: kết quả `2`, validation passed.

### Phát hiện từ thử nghiệm đầu tiên

1. Model có thể hoàn thành task nhỏ đúng kết quả.
2. LM Studio có thể trả structured output của model distilled trong `reasoning_content`, còn `content` rỗng.
3. Theo policy hiện tại, adapter chỉ chấp nhận `content`; output chỉ có trong `reasoning_content` bị đánh fail.
4. Giới hạn `max_tokens=160` và `512` từng làm model hết budget trước khi có output hữu dụng.
5. `max_tokens=2048` đủ cho lần thử hiện tại.
6. LM Studio structured-output grammar từng đặt JSON vào `reasoning_content`; request hiện không dùng `response_format` và validator kiểm tra schema sau khi nhận `content`.
7. `maxLength=80` làm evidence ngắn nhưng có thể bị cắt câu máy móc.
8. Validator hiện vẫn gắn cứng với đáp án của một task; chưa phải validator tổng quát.

### Chẩn đoán llama.cpp: thinking kết hợp JSON Schema

Runtime đã thử: llama.cpp commit `0eca4d4`, LM Studio runtime `2.24.0`, Qwen 3.5 9B Q4, `--reasoning on`, `--reasoning-format deepseek`.

- Chạy 12 schema độc lập, mỗi case dùng một server sạch và streaming.
- Các case gồm một boolean, thêm dần bốn field, enum, nullable union, `maxLength` và `additionalProperties`.
- Kết quả: 0 passed, 12 failed.
- Mỗi case sinh khoảng 750 ký tự reasoning rồi bắt đầu final content.
- Grammar không được áp dụng: schema một field `ok` vẫn sinh thêm field ngoài schema.
- Budget 256 token bị reasoning dùng phần lớn, khiến final JSON bị cắt giữa chuỗi.
- Timeout quan sát trước đó là biểu hiện của request/grammar không đạt contract, không phải một constraint đơn lẻ gây exponential blowup đã được chứng minh.

Kết luận hiện tại: runtime llama.cpp này không phù hợp với contract bắt buộc “thinking bật + reasoning riêng + final content bị ép JSON Schema”. Xem `llama-schema-diagnostic.json`.

### Chưa hoàn thành

- [ ] Chưa thử nhiều lần để đo độ ổn định.
- [ ] Chưa thử task phải trả `blocked`.
- [ ] Chưa thử prompt injection hoặc yêu cầu mở rộng phạm vi.
- [ ] Chưa có Task Splitter skill.
- [ ] Chưa có schema validator dùng chung.
- [ ] Chưa có scheduler hoặc state machine.
- [ ] Chưa có handoff gate giữa các vai trò.
- [ ] Chưa có Reviewer/QA skill.
- [ ] Chưa có retry/replan policy.
- [ ] Chưa so sánh nhiều cách viết skill hoặc nhiều local model.

## Các giai đoạn

### Giai đoạn 0 — Tầm nhìn và nguyên tắc

Trạng thái: **hoàn thành bản nháp**.

Đầu ra: tầm nhìn local-first, phân vai công ty, ranh giới LLM/code và nguyên tắc atomic task.

### Giai đoạn 1 — Atomic Worker Lab

Trạng thái: **đang thực hiện**.

Mục tiêu: chốt cách viết skill khiến model làm đúng một việc, nói ít và biết dừng.

Việc tiếp theo:

1. Tách runner, LM Studio adapter và validator thành module riêng.
2. Tạo bộ test khoảng 20 task nhỏ.
3. Chạy mỗi task nhiều lần với temperature cố định và khác nhau.
4. Đo schema compliance, scope compliance, false completion và verbosity.
5. So sánh các phiên bản wording của `SKILL.md`.
6. Chốt Atomic Worker skill v1 khi đạt ngưỡng ổn định.

Điều kiện hoàn thành giai đoạn:

- Output đúng schema ≥ 99% trên test suite.
- Không mở rộng phạm vi ≥ 95%.
- Task thiếu dữ liệu trả `blocked` ≥ 95%.
- Không false-complete trên test âm.
- Output nằm trong giới hạn độ dài đã định.

Các ngưỡng trên là đề xuất ban đầu, chưa được xác nhận bằng dữ liệu.

### Giai đoạn 2 — Task Splitter

Trạng thái: **chưa bắt đầu**.

Mục tiêu: chia yêu cầu thành atomic task nhưng tuyệt đối không thực hiện task.

Đầu ra dự kiến: danh sách task, dependency, input/output, acceptance criteria và lý do task đã đủ nhỏ.

### Giai đoạn 3 — Harness Core

Trạng thái: **chưa bắt đầu**.

Mục tiêu: state machine, scheduler, permission gate, budgets, retry và audit log.

### Giai đoạn 4 — Reviewer và Handoff Gates

Trạng thái: **chưa bắt đầu**.

Mục tiêu: kiểm tra artifact từng vai trò và chỉ cho phép handoff khi contract đạt.

### Giai đoạn 5 — Product/BA/Architect pipeline

Trạng thái: **chưa bắt đầu**.

Mục tiêu: biến yêu cầu lớn, mơ hồ thành artifact cụ thể mà không đi sớm vào triển khai.

### Giai đoạn 6 — Integration end-to-end

Trạng thái: **chưa bắt đầu**.

Mục tiêu: chạy một yêu cầu vừa phải qua toàn bộ pipeline, có retry, replan và nghiệm thu.

## Thứ tự xây dựng đã thống nhất

```text
Atomic Worker
→ Deterministic Validator
→ Task Splitter
→ Handoff Gate
→ Reviewer
→ Orchestrator
→ Product/BA/Architect pipeline
```

## Cập nhật controlled inference — 2026-07-15

- [x] Phase E stress test: 8/8.
- [x] Phase F service API/streaming/queue/recovery/cancellation: 10/10 regression.
- [x] Phase G persistent worker: model giữ trong VRAM qua ba request tuần tự, 10/10.
- [x] Fresh context cho từng request; không reuse KV ngầm.
- [ ] Service-level timeout và watchdog đầy đủ.
- [ ] Continuous batching/concurrency.
- [ ] Prompt cache hoặc KV reuse có lifecycle rõ ràng.
- [ ] Multi-model/model capability registry.

Hiệu chỉnh sau adversarial review: các kết quả trên chứng minh controlled-inference spike trong phạm vi hẹp, không chứng minh deterministic task acceptance hoặc production model-worker readiness. Mốc ưu tiên hiện tại là [Model Worker v1](06-model-worker-v1-plan.md); agent/tool work được hoãn.

## Mức sẵn sàng hiện tại

### Mức 1 — Controlled-inference prototype

Trạng thái: **đủ để tiếp tục nghiên cứu, chưa đủ làm production model worker**.

Model đã có thể nhận một task nhỏ, thinking và trả final có cấu trúc. Lớp hiện tại đã chứng minh:

- Reasoning boundary được quan sát trên model hiện tại; grammar chỉ áp dụng sau end marker.
- Chỉ final content được chấp nhận, không dùng reasoning làm fallback.
- JSON Schema subset, JSON parse và shape/type validation bước đầu.
- Reasoning/final/total budget, cancellation và termination code.
- API, SSE streaming, queue nhỏ, health, metrics và crash recovery.
- Model giữ trong VRAM qua nhiều request tuần tự.
- Fresh context cho từng request; không rò KV/token history.
- Cùng pipeline cho tiếng Anh và tiếng Việt.

Chưa được tuyên bố: model-independent protocol, fail-closed contract validation, task acceptance, service deadline/watchdog, bounded resources hoặc production security.

### Mức 2 — Tool-using atomic agent

Trạng thái: **chưa hoàn thành và chưa phải mốc tiếp theo**.

Chỉ bắt đầu mốc này sau khi Model Worker v1 vượt release gate. Agent phải làm được một task nhỏ bằng tool nhưng vẫn bị code tất định kiểm soát từng hành động.

Các thành phần cần xây:

1. **Tool contract**
   - Bắt đầu với 2–3 tool nhỏ: `read_file`, `write_artifact`, `run_test`.
   - Input/output mỗi tool có schema riêng.
   - Mỗi lượt model chỉ được chọn đúng một action.
   - Model không trực tiếp thực thi; deterministic runtime parse và gọi tool.

2. **Agent loop tối thiểu**
   - Luồng: `observe → choose one action → execute → inspect result → continue/stop`.
   - Có giới hạn số vòng, token, thời gian và số lần gọi tool.
   - Chỉ code runtime được quyết định loop đã hoàn tất hay chưa.

3. **Permission gate**
   - Allowlist tool theo skill/role.
   - Giới hạn đường dẫn, workspace và thao tác đọc/ghi.
   - Chặn command, network hoặc mutation ngoài phạm vi.
   - Hành động nhạy cảm phải fail trước khi tool chạy.

4. **Task state và artifact store**
   - Lưu task input, acceptance criteria, action history và tool result.
   - Lưu artifact đầu ra độc lập với context của model.
   - Có request ID, attempt ID, timestamps và audit trail.
   - Có thể phục hồi hoặc điều tra mà không dựa vào reasoning text.

5. **Acceptance validator theo loại task**
   - File cần tạo/sửa có tồn tại.
   - Nội dung hoặc cấu trúc đạt contract.
   - Test/lint cần thiết pass.
   - Không sửa file ngoài phạm vi.
   - Model không được tự đặt `accepted=true` hoặc tự tuyên bố system completion.

6. **Skills runtime**
   - Load đúng skill theo role/task type.
   - Skill quy định phạm vi nhỏ, tool được phép, budgets và output contract.
   - Prompt được tạo từ skill + task artifact bằng code tất định.
   - Thử nhiều wording và đo scope compliance, verbosity, blocked behavior.

7. **Retry và recovery policy**
   - Phân loại: schema fail, tool fail, acceptance fail, hết budget, cancelled, blocked.
   - Chỉ retry lỗi được phép retry; có giới hạn attempt.
   - Retry dùng state/artifact rõ ràng, không dựa vào ký ức ngầm của model.
   - Không tự mở rộng phạm vi task khi retry.

Điều kiện để gọi là tool-using atomic agent:

- Hoàn thành một task file nhỏ end-to-end bằng tool.
- Mỗi lượt chỉ có một action hợp lệ.
- Tool ngoài allowlist và path ngoài scope bị chặn tất định.
- Acceptance do validator quyết định.
- Retry/cancel/budget đều có termination rõ ràng.
- Toàn bộ action và artifact có audit log.

### Mức 3 — Multi-agent harness

Trạng thái: **chưa bắt đầu triển khai**.

Các lớp cần bổ sung sau khi Mức 2 ổn định:

8. **Task Splitter**
   - Chỉ chia việc, tuyệt đối không thực hiện.
   - Mỗi step có input, output, dependency, role và acceptance criteria.
   - Code kiểm tra step có đủ nhỏ trước khi giao Atomic Worker.

9. **Scheduler và dependency graph**
   - Chỉ chạy step khi dependency đã accepted.
   - Mỗi step giao cho đúng role/subagent.
   - Quản lý trạng thái `pending/running/blocked/failed/accepted`.
   - Không dùng lời tuyên bố của model làm trạng thái hệ thống.

10. **Reviewer và handoff gate**
    - Reviewer kiểm tra artifact, không tin summary của worker.
    - Handoff chỉ mở khi schema và acceptance criteria đều pass.
    - Reviewer fail phải trả lỗi cụ thể cho retry hoặc replan.

11. **Replan policy**
    - Chỉ Task Splitter/Planner được sửa plan.
    - Worker không tự tạo step mới hoặc đổi acceptance criteria.
    - Replan có version, lý do và audit trail.

12. **Product/BA/Architect pipeline**
    - Product giữ ở mục tiêu và giá trị cần đạt.
    - BA chuyển mục tiêu thành requirement/acceptance rõ ràng.
    - Architect tạo boundary và technical plan, chưa thực hiện code.
    - Chỉ sau handoff gate mới chuyển sang Task Splitter và worker triển khai.

13. **Integration end-to-end**
    - Chạy một yêu cầu vừa phải qua toàn pipeline.
    - Có blocked case, retry, reviewer reject và replan.
    - Integrator chỉ ghép artifact đã accepted.

### Mức 4 — Production hardening

Các việc không chặn thử nghiệm atomic agent nhưng cần trước production:

- Service-level timeout, watchdog và graceful shutdown đầy đủ.
- Authentication, rate limit và giới hạn tài nguyên.
- Metrics latency percentile, tokens/s, VRAM và failure class.
- Model/reasoning-protocol capability registry.
- Compatibility suite trên nhiều model/template/tokenizer.
- Mở rộng schema compiler khi contract thực tế yêu cầu.
- Continuous batching/concurrency chỉ sau khi isolation đã được kiểm chứng.
- Prompt cache/KV reuse chỉ sau khi có session lifecycle rõ ràng.

## Thứ tự triển khai tiếp theo

```text
1. Hoàn thành Phase 0–9 của Model Worker v1
2. Thu evidence unit/property/fault/GPU/soak cùng revision
3. Chỉ sau release gate mới thiết kế tool schemas và one-action agent loop
4. Permission gate, task state, validator và retry policy
5. Task Splitter, scheduler, reviewer và replan
6. Product/BA/Architect pipeline và end-to-end integration
```

Quy tắc xuyên suốt: việc lớn luôn được vai trò chuyên chia nhỏ; người chia việc không thực hiện; worker chỉ nhận một việc nhỏ; code tất định giữ quyền tool, state, acceptance và system completion.
