#include "generation_safety.h"

#include <cstdint>
#include <utility>

namespace model_worker {

bool grammar_acceptance_proven(bool is_eog, GrammarSampleSource source) noexcept {
    return is_eog && source == GrammarSampleSource::final_grammar;
}

namespace {

bool continuation(unsigned char byte) {
    return byte >= 0x80 && byte <= 0xBF;
}

}  // namespace

Utf8AppendResult Utf8Accumulator::append(std::string_view bytes) {
    if (!valid_) return {false, {}};
    pending_.append(bytes);
    std::size_t offset = 0;
    while (offset < pending_.size()) {
        const auto first = static_cast<unsigned char>(pending_[offset]);
        std::size_t width = 0;
        std::uint32_t code_point = 0;
        std::uint32_t minimum = 0;
        if (first <= 0x7F) {
            width = 1; code_point = first;
        } else if (first >= 0xC2 && first <= 0xDF) {
            width = 2; code_point = first & 0x1F; minimum = 0x80;
        } else if (first >= 0xE0 && first <= 0xEF) {
            width = 3; code_point = first & 0x0F; minimum = 0x800;
        } else if (first >= 0xF0 && first <= 0xF4) {
            width = 4; code_point = first & 0x07; minimum = 0x10000;
        } else {
            valid_ = false;
            return {false, {}};
        }
        if (pending_.size() - offset < width) break;
        for (std::size_t index = 1; index < width; ++index) {
            const auto byte = static_cast<unsigned char>(pending_[offset + index]);
            if (!continuation(byte)) {
                valid_ = false;
                return {false, {}};
            }
            code_point = (code_point << 6) | (byte & 0x3F);
        }
        if (code_point < minimum || code_point > 0x10FFFF ||
            (code_point >= 0xD800 && code_point <= 0xDFFF)) {
            valid_ = false;
            return {false, {}};
        }
        offset += width;
    }
    std::string completed = pending_.substr(0, offset);
    pending_.erase(0, offset);
    return {true, std::move(completed)};
}

}  // namespace model_worker
