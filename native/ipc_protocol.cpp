#include "ipc_protocol.h"

namespace model_worker {
bool valid_frame_identity(const FrameIdentity & frame) noexcept {
    return frame.protocol_version == "model-worker-ipc.v1" &&
           !frame.request_id.empty() && !frame.attempt_id.empty();
}
}  // namespace model_worker
