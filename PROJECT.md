# Agent Harness Lab

Project nghiên cứu agent harness local-first: chia yêu cầu lớn qua nhiều vai trò và chỉ giao atomic task cho local LLM.

Tài liệu:

- [Tầm nhìn](docs/01-vision.md)
- [Kiến trúc](docs/02-architecture.md)
- [Tiến độ và roadmap](docs/03-progress-and-roadmap.md)
- [Session handoff](docs/04-session-handoff.md)
- [Controlled inference handoff](docs/05-controlled-inference-handoff.md)
- [Kế hoạch Model Worker v1](docs/06-model-worker-v1-plan.md)

Trạng thái hiện tại: source tree Model Worker v1 đã được tách khỏi controlled-inference prototype, có API/IPC/manifest/output-contract version độc lập và reliability suite. Bản phát hành chỉ được ký khi `scripts/release_gate.ps1` có đủ evidence native, GPU và soak từ cùng manifest/runtime. Prototype Phase A–G trong `controlled_inference/` là archive, không được production package import.
