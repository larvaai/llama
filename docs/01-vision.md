# Tầm nhìn Agent Harness

## Mục tiêu

Xây một agent harness chạy hoàn toàn bằng local LLM. Harness nhận yêu cầu lớn nhưng không đưa toàn bộ yêu cầu đó cho một model xử lý trong một lần gọi.

Hệ thống mô phỏng cách một công ty làm việc:

```text
Ý tưởng lớn
→ Product discovery
→ BA làm rõ yêu cầu
→ Architect thiết kế cấp cao
→ Planner chia việc
→ Worker làm một việc nhỏ
→ QA kiểm chứng
→ Integrator ghép kết quả
→ Final gate nghiệm thu
```

Mỗi vai trò chỉ thực hiện một loại tư duy và chỉ tạo một loại artifact.

## Nguyên tắc nền tảng

1. LLM quyết định nội dung; code tất định kiểm soát hành vi.
2. Harness chọn vai trò, context, tool và output cho LLM.
3. LLM không tự cấp quyền hoặc tự mở rộng phạm vi.
4. Một lần gọi LLM chỉ làm một việc nhỏ.
5. Người chia việc chỉ chia việc, không thực hiện.
6. Worker chỉ thực hiện, không lập kế hoạch lớn.
7. Người làm không được tự chấm và tự đánh dấu hoàn thành.
8. Mọi kết luận hoàn thành phải có bằng chứng kiểm tra được.
9. Yêu cầu lớn phải đi qua các artifact trung gian trước khi thành task code.
10. Khi task vẫn chứa nhiều hành động, tiếp tục chia nhỏ.

## Lý do chọn thiết kế này

Local LLM nhỏ có chất lượng tốt hơn khi:

- Context ngắn và liên quan trực tiếp.
- Vai trò rõ ràng.
- Output bị giới hạn bằng schema.
- Task có một mục tiêu và một kết quả.
- Acceptance criteria cụ thể.
- Code kiểm tra thay vì tin lời model.

Mục tiêu không phải tạo một “siêu agent”, mà tạo một tổ chức gồm nhiều vai nhỏ, được điều phối bằng quy trình tất định.

## Phạm vi ban đầu

Ưu tiên nghiên cứu cách viết skill để local model:

- Nói ít.
- Không đi ngoài phạm vi.
- Biết từ chối task quá lớn.
- Trả đúng schema.
- Không khai hoàn thành khi thiếu bằng chứng.
- Ổn định qua nhiều lần chạy.

Chưa ưu tiên xây UI, chạy nhiều agent song song hoặc tự động triển khai dự án lớn.
