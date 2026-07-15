# Session handoff

## Đọc trước khi tiếp tục

Đọc theo thứ tự:

1. `docs/01-vision.md`
2. `docs/02-architecture.md`
3. `docs/03-progress-and-roadmap.md`
4. `docs/05-controlled-inference-handoff.md`
5. `skills/atomic-worker/SKILL.md`
6. `run_experiment.py`
7. `experiment-task.json`
8. `experiment-result.json`

## Project hiện tại

```text
D:\zalollm\agent-harness-lab
```

LM Studio API:

```text
http://localhost:1234/v1/chat/completions
```

Model đã thử:

```text
qwen3.5-9b-claude-4.6-opus-uncensored-distilled
```

Lệnh chạy lại:

```powershell
python .\run_experiment.py
```

Kết quả gần nhất: validation passed, `result = 2`.

## Quyết định không được làm mất

- Local LLM không nhận một yêu cầu lớn để tự xử lý toàn bộ.
- Yêu cầu lớn đi qua Product/BA/Architect/Planner trước Dev.
- Mỗi vai trò có skill, context và output contract riêng.
- Task Splitter chỉ chia việc, không thực hiện.
- Atomic Worker chỉ làm một task nhỏ, không lập kế hoạch.
- Code tất định kiểm soát state, schema, permission, budget và acceptance.
- Model không được tự xác nhận hoàn thành ở cấp hệ thống.

## Điểm kỹ thuật cần nhớ

Model distilled này có thể trả JSON trong `reasoning_content` thay vì `content`. Policy hiện tại chỉ chấp nhận `content`; trường hợp đó phải bị đánh fail và lưu raw response để debug.

Không thêm lại `response_format=json_schema` mà chưa kiểm thử: với model và LM Studio hiện tại, structured-output grammar khiến JSON bị phân loại vào `reasoning_content`. Runner đang yêu cầu JSON thuần bằng skill rồi validate tất định sau khi nhận `content`.

Standalone llama.cpp cũng đã được kiểm thử với thinking bật và `--reasoning-format deepseek`. Diagnostic 12 schema cho kết quả 0/12 đạt: grammar không giới hạn final content đúng schema và JSON thường bị cắt do reasoning chiếm token budget. Artifact: `diagnose_llama_schema.py` và `llama-schema-diagnostic.json`. Không lặp lại thử nghiệm này nếu chưa đổi runtime/parser/backend.

Evidence 80 ký tự ở lần chạy gần nhất bị cắt câu. Không nên coi đây là wording cuối cùng của skill. Cần thử một trong các hướng:

- Giới hạn số trường hoặc số token thay vì `maxLength` cứng.
- Ép evidence theo mẫu dữ liệu ngắn.
- Để code tự sinh evidence cho tiêu chí tất định.
- Không yêu cầu LLM evidence khi validator có thể tự tính.

## Task nên làm ngay trong session mới

Thiết kế test harness cho Atomic Worker trước khi viết Task Splitter:

1. Tạo manifest chứa nhiều test case.
2. Có test dương, test thiếu dữ liệu, test quá lớn và test vượt phạm vi.
3. Chạy lặp mỗi case.
4. Lưu raw response và normalized response riêng.
5. Tính metric thay vì đánh giá bằng mắt.
6. Dùng kết quả để sửa wording của skill.

Không xây orchestrator ở bước kế tiếp; Atomic Worker chưa đủ dữ liệu để chốt.
