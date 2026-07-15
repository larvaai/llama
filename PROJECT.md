# Agent Harness Lab

Project nghiên cứu agent harness local-first: chia yêu cầu lớn qua nhiều vai trò và chỉ giao atomic task cho local LLM.

Tài liệu:

- [Tầm nhìn](docs/01-vision.md)
- [Kiến trúc](docs/02-architecture.md)
- [Tiến độ lịch sử](docs/03-progress-and-roadmap.md)
- [Session handoff index](docs/04-session-handoff.md)
- [Controlled inference handoff — archive A–G](docs/05-controlled-inference-handoff.md)
- [Kế hoạch Model Worker v1](docs/06-model-worker-v1-plan.md)
- [Roadmap hiện hành: Inference Runtime và Agent Harness](docs/07-inference-runtime-and-agent-roadmap.md)
- [Historical artifact: M0 hardening start](docs/08-m0-implementation-handoff.md)
- [Implementation checkpoint và handoff còn lại](docs/09-inference-runtime-implementation-handoff.md)

Trạng thái hiện tại: Model Worker M0 đã được release-attest từ revision sạch
`b38b6df32755c55e668ac11e6c8f3e8b1c2ad46b` trên RTX 3080. Consolidated gate
đã build đúng native binary, kiểm tra unit/property/integration/native/GPU/fault,
chứng minh crash → `DEGRADED` → request-triggered recovery và chạy soak 500/500
không lỗi cùng resource series. Inference Runtime multi-sequence, continuous
batching, governance, prefix/session cache và backend registry/router đã đạt
M1–M7 engineering gate; M7 real-provider matrix vẫn là deployment gate theo
môi trường. H0 durable agent state và H1 atomic read-only tool slice đã đạt
engineering gate; full H1 chưa được tuyên bố vì chưa có real-local-model
acceptance artifact và mutation/approval execution vẫn bị khóa. `docs/07` là
roadmap có thẩm quyền, `docs/09` là handoff hiện hành và
`docs/model-worker-release.md` ghi release attestation. Prototype Phase A–G
trong `controlled_inference/` là archive, không được production package import.
