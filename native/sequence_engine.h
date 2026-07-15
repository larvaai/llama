#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace model_worker {

enum class SequenceLifecycle {
    admitted,
    prefill,
    decode,
    terminal,
    released,
};

enum class SequenceFinishReason {
    none,
    stop,
    length,
    cancelled,
    failed,
};

struct SequenceHandle {
    std::int32_t slot{-1};
    std::uint64_t generation{0};
    std::string request_id;
    std::string attempt_id;

    friend bool operator==(const SequenceHandle &, const SequenceHandle &) = default;
};

enum class AdmissionStatus {
    admitted,
    invalid_request,
    duplicate,
    sequence_capacity,
    token_capacity,
};

struct AdmissionResult {
    AdmissionStatus status{AdmissionStatus::invalid_request};
    std::optional<SequenceHandle> handle;
};

enum class SequenceOperationStatus {
    applied,
    stale_handle,
    invalid_state,
    invalid_amount,
};

enum class SequenceReleaseStatus {
    released,
    already_released,
    stale_handle,
    invalid_state,
};

struct SequenceReleaseResult {
    SequenceReleaseStatus status{SequenceReleaseStatus::stale_handle};
    std::size_t released_tokens{0};
};

struct SequenceSnapshot {
    SequenceHandle handle;
    SequenceLifecycle lifecycle{SequenceLifecycle::released};
    SequenceFinishReason finish_reason{SequenceFinishReason::none};
    std::size_t prompt_tokens{0};
    std::size_t output_token_budget{0};
    std::size_t prompt_tokens_processed{0};
    std::size_t decoded_tokens{0};
    std::size_t reserved_tokens{0};
};

class SequenceEngine {
public:
    SequenceEngine(std::size_t max_sequences, std::size_t token_capacity);

    AdmissionResult admit(
        std::string request_id,
        std::string attempt_id,
        std::size_t prompt_tokens,
        std::size_t output_token_budget
    );

    SequenceOperationStatus start_prefill(const SequenceHandle & handle);
    SequenceOperationStatus advance_prefill(
        const SequenceHandle & handle,
        std::size_t tokens
    );
    SequenceOperationStatus advance_decode(
        const SequenceHandle & handle,
        std::size_t tokens
    );
    SequenceOperationStatus finish(
        const SequenceHandle & handle,
        SequenceFinishReason reason
    );
    SequenceOperationStatus cancel(const SequenceHandle & handle);
    SequenceReleaseResult release(const SequenceHandle & handle);

    std::optional<SequenceHandle> next_runnable();
    std::optional<SequenceSnapshot> snapshot(const SequenceHandle & handle) const;

    std::size_t max_sequences() const noexcept { return slots_.size(); }
    std::size_t token_capacity() const noexcept { return token_capacity_; }
    std::size_t active_sequences() const noexcept { return active_sequences_; }
    std::size_t reserved_tokens() const noexcept { return reserved_tokens_; }

private:
    struct Slot {
        std::uint64_t generation{0};
        std::string request_id;
        std::string attempt_id;
        SequenceLifecycle lifecycle{SequenceLifecycle::released};
        SequenceFinishReason finish_reason{SequenceFinishReason::none};
        std::size_t prompt_tokens{0};
        std::size_t output_token_budget{0};
        std::size_t prompt_tokens_processed{0};
        std::size_t decoded_tokens{0};

        std::size_t reservation() const noexcept {
            return prompt_tokens + output_token_budget;
        }
    };

    bool matches(const Slot & slot, const SequenceHandle & handle) const noexcept;
    SequenceHandle make_handle(std::size_t index, const Slot & slot) const;
    Slot * find(const SequenceHandle & handle) noexcept;
    const Slot * find(const SequenceHandle & handle) const noexcept;
    bool duplicate(const std::string & request_id, const std::string & attempt_id) const;
    static bool runnable(SequenceLifecycle lifecycle) noexcept;

    std::vector<Slot> slots_;
    std::size_t token_capacity_;
    std::size_t reserved_tokens_{0};
    std::size_t active_sequences_{0};
    std::size_t round_robin_cursor_{0};
};

}  // namespace model_worker
