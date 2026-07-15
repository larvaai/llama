# Kiến trúc và ranh giới kiểm soát

## Các vai trò dự kiến

| Vai trò | Được làm | Không được làm |
|---|---|---|
| Product | Làm rõ vấn đề, người dùng và giá trị | Thiết kế code |
| BA | Viết yêu cầu, luồng và acceptance criteria | Chọn cách triển khai |
| Architect | Xác định component, interface và ràng buộc | Viết toàn bộ tính năng |
| Task Splitter | Chia việc và dependency | Thực hiện task |
| Atomic Worker | Làm đúng một task nhỏ | Chia việc hoặc mở rộng phạm vi |
| Reviewer/QA | Kiểm tra tiêu chí bằng bằng chứng | Tự sửa rồi tự duyệt |
| Integrator | Ghép các artifact đã đạt chuẩn | Bỏ qua contract |
| Final Gate | Kiểm tra mục tiêu tổng thể | Chấp nhận bằng cảm tính |

## Ranh giới LLM và code tất định

### LLM phụ trách

- Hiểu ngữ nghĩa yêu cầu.
- Làm rõ mục tiêu trong phạm vi vai trò.
- Tạo nội dung artifact.
- Chia task khi đang mang vai Task Splitter.
- Thực hiện một task khi đang mang vai Atomic Worker.
- Đánh giá phần ngữ nghĩa khi test tất định không đủ.

### Code tất định phụ trách

- Validate input và output schema.
- Quản lý ID, trạng thái và dependency.
- Chặn task chưa đủ điều kiện chạy.
- Chọn skill, model, tool và permission.
- Giới hạn token, timeout, retry và độ dài output.
- Chạy test, lint, type-check và kiểm tra artifact.
- Lưu prompt, response, usage và audit trail.
- Quyết định state transition.
- Chặn dependency cycle và vòng lặp retry.
- Pause, cancel, rollback hoặc yêu cầu người dùng duyệt.

## State machine đề xuất

```text
pending → ready → running → reviewing → passed
                      ↓          ↓
                    failed ← rejected
                      ↓
                retry | replan | blocked
```

Chỉ scheduler/harness được đổi trạng thái. Chuỗi `completed` do model trả về chỉ là một đề nghị, không phải sự thật hệ thống.

## Handoff contract

Mỗi bước chuyển vai trò phải có artifact rõ ràng:

```yaml
handoff:
  from: BA
  to: Architect
  artifact: requirements_spec
  required_fields:
    - scope
    - user_flows
    - constraints
    - acceptance_criteria
  forbidden_fields:
    - source_code
    - implementation_details
```

Harness kiểm tra contract trước khi nạp artifact cho vai trò tiếp theo.

## Contract của một atomic task

```yaml
goal: một kết quả duy nhất
context: chỉ dữ liệu cần thiết
allowed_actions: danh sách hành động được phép
forbidden_actions: danh sách hành động bị cấm
expected_output: một artifact cụ thể
acceptance_criteria: điều kiện kiểm chứng được
```

Output tối thiểu:

```json
{
  "status": "completed|blocked",
  "result": null,
  "evidence": "short factual evidence",
  "reason": null
}
```

## Chuỗi kiểm tra

```text
Schema validator
→ deterministic checks
→ automated tests
→ semantic reviewer LLM khi cần
→ final system gate
```

Không dùng LLM reviewer cho điều mà code có thể kiểm tra chính xác.

## Tiêu chí một task đủ nhỏ

Task được giao cho Atomic Worker chỉ khi:

- Có đúng một goal.
- Có đúng một output chính.
- Không cần tự chia bước.
- Context đã đủ.
- Allowed và forbidden actions rõ ràng.
- Acceptance criteria có thể kiểm chứng.
- Có thể hoàn thành trong một lần gọi model với budget giới hạn.

Nếu không đạt, Task Splitter phải tiếp tục chia hoặc trả `blocked`.
