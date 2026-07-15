#pragma once

#include <cstdint>
#include <string>

namespace model_worker {

struct FrameIdentity {
    std::string protocol_version;
    std::string request_id;
    std::string attempt_id;
    std::uint64_t sequence = 0;
};

bool valid_frame_identity(const FrameIdentity & frame) noexcept;

}  // namespace model_worker
