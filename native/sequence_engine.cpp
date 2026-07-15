#include "sequence_engine.h"

#include <limits>
#include <stdexcept>

namespace model_worker {

SequenceEngine::SequenceEngine(
    const std::size_t max_sequences,
    const std::size_t token_capacity
) : slots_(max_sequences), token_capacity_(token_capacity) {
    if (max_sequences == 0) {
        throw std::invalid_argument("max_sequences must be positive");
    }
    if (token_capacity == 0) {
        throw std::invalid_argument("token_capacity must be positive");
    }
    if (max_sequences > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max())) {
        throw std::invalid_argument("max_sequences exceeds sequence ID range");
    }
}

AdmissionResult SequenceEngine::admit(
    std::string request_id,
    std::string attempt_id,
    const std::size_t prompt_tokens,
    const std::size_t output_token_budget
) {
    if (request_id.empty() || attempt_id.empty() || prompt_tokens == 0 || output_token_budget == 0) {
        return {AdmissionStatus::invalid_request, std::nullopt};
    }
    if (prompt_tokens > std::numeric_limits<std::size_t>::max() - output_token_budget) {
        return {AdmissionStatus::invalid_request, std::nullopt};
    }
    if (duplicate(request_id, attempt_id)) {
        return {AdmissionStatus::duplicate, std::nullopt};
    }
    if (active_sequences_ == slots_.size()) {
        return {AdmissionStatus::sequence_capacity, std::nullopt};
    }
    const auto reservation = prompt_tokens + output_token_budget;
    if (reservation > token_capacity_ - reserved_tokens_) {
        return {AdmissionStatus::token_capacity, std::nullopt};
    }

    for (std::size_t index = 0; index < slots_.size(); ++index) {
        auto & slot = slots_[index];
        if (slot.lifecycle != SequenceLifecycle::released) {
            continue;
        }
        ++slot.generation;
        if (slot.generation == 0) {
            ++slot.generation;
        }
        slot.request_id = std::move(request_id);
        slot.attempt_id = std::move(attempt_id);
        slot.lifecycle = SequenceLifecycle::admitted;
        slot.finish_reason = SequenceFinishReason::none;
        slot.prompt_tokens = prompt_tokens;
        slot.output_token_budget = output_token_budget;
        slot.prompt_tokens_processed = 0;
        slot.decoded_tokens = 0;
        reserved_tokens_ += reservation;
        ++active_sequences_;
        return {AdmissionStatus::admitted, make_handle(index, slot)};
    }
    return {AdmissionStatus::sequence_capacity, std::nullopt};
}

SequenceOperationStatus SequenceEngine::start_prefill(const SequenceHandle & handle) {
    auto * slot = find(handle);
    if (slot == nullptr) {
        return SequenceOperationStatus::stale_handle;
    }
    if (slot->lifecycle != SequenceLifecycle::admitted) {
        return SequenceOperationStatus::invalid_state;
    }
    slot->lifecycle = SequenceLifecycle::prefill;
    return SequenceOperationStatus::applied;
}

SequenceOperationStatus SequenceEngine::advance_prefill(
    const SequenceHandle & handle,
    const std::size_t tokens
) {
    auto * slot = find(handle);
    if (slot == nullptr) {
        return SequenceOperationStatus::stale_handle;
    }
    if (slot->lifecycle != SequenceLifecycle::prefill) {
        return SequenceOperationStatus::invalid_state;
    }
    const auto remaining = slot->prompt_tokens - slot->prompt_tokens_processed;
    if (tokens == 0 || tokens > remaining) {
        return SequenceOperationStatus::invalid_amount;
    }
    slot->prompt_tokens_processed += tokens;
    if (slot->prompt_tokens_processed == slot->prompt_tokens) {
        slot->lifecycle = SequenceLifecycle::decode;
    }
    return SequenceOperationStatus::applied;
}

SequenceOperationStatus SequenceEngine::advance_decode(
    const SequenceHandle & handle,
    const std::size_t tokens
) {
    auto * slot = find(handle);
    if (slot == nullptr) {
        return SequenceOperationStatus::stale_handle;
    }
    if (slot->lifecycle != SequenceLifecycle::decode) {
        return SequenceOperationStatus::invalid_state;
    }
    const auto remaining = slot->output_token_budget - slot->decoded_tokens;
    if (tokens == 0 || tokens > remaining) {
        return SequenceOperationStatus::invalid_amount;
    }
    slot->decoded_tokens += tokens;
    return SequenceOperationStatus::applied;
}

SequenceOperationStatus SequenceEngine::finish(
    const SequenceHandle & handle,
    const SequenceFinishReason reason
) {
    auto * slot = find(handle);
    if (slot == nullptr) {
        return SequenceOperationStatus::stale_handle;
    }
    if (slot->lifecycle != SequenceLifecycle::decode) {
        return SequenceOperationStatus::invalid_state;
    }
    if (
        reason != SequenceFinishReason::stop
        && reason != SequenceFinishReason::length
        && reason != SequenceFinishReason::failed
    ) {
        return SequenceOperationStatus::invalid_amount;
    }
    slot->lifecycle = SequenceLifecycle::terminal;
    slot->finish_reason = reason;
    return SequenceOperationStatus::applied;
}

SequenceOperationStatus SequenceEngine::cancel(const SequenceHandle & handle) {
    auto * slot = find(handle);
    if (slot == nullptr) {
        return SequenceOperationStatus::stale_handle;
    }
    if (
        slot->lifecycle != SequenceLifecycle::admitted
        && slot->lifecycle != SequenceLifecycle::prefill
        && slot->lifecycle != SequenceLifecycle::decode
    ) {
        return SequenceOperationStatus::invalid_state;
    }
    slot->lifecycle = SequenceLifecycle::terminal;
    slot->finish_reason = SequenceFinishReason::cancelled;
    return SequenceOperationStatus::applied;
}

SequenceReleaseResult SequenceEngine::release(const SequenceHandle & handle) {
    auto * slot = find(handle);
    if (slot == nullptr) {
        if (
            handle.slot >= 0
            && static_cast<std::size_t>(handle.slot) < slots_.size()
            && slots_[static_cast<std::size_t>(handle.slot)].generation == handle.generation
            && slots_[static_cast<std::size_t>(handle.slot)].lifecycle == SequenceLifecycle::released
        ) {
            return {SequenceReleaseStatus::already_released, 0};
        }
        return {SequenceReleaseStatus::stale_handle, 0};
    }
    if (slot->lifecycle != SequenceLifecycle::terminal) {
        return {SequenceReleaseStatus::invalid_state, 0};
    }
    const auto released = slot->reservation();
    reserved_tokens_ -= released;
    --active_sequences_;
    slot->request_id.clear();
    slot->attempt_id.clear();
    slot->lifecycle = SequenceLifecycle::released;
    slot->finish_reason = SequenceFinishReason::none;
    slot->prompt_tokens = 0;
    slot->output_token_budget = 0;
    slot->prompt_tokens_processed = 0;
    slot->decoded_tokens = 0;
    return {SequenceReleaseStatus::released, released};
}

std::optional<SequenceHandle> SequenceEngine::next_runnable() {
    for (std::size_t offset = 0; offset < slots_.size(); ++offset) {
        const auto index = (round_robin_cursor_ + offset) % slots_.size();
        const auto & slot = slots_[index];
        if (!runnable(slot.lifecycle)) {
            continue;
        }
        round_robin_cursor_ = (index + 1) % slots_.size();
        return make_handle(index, slot);
    }
    return std::nullopt;
}

std::optional<SequenceSnapshot> SequenceEngine::snapshot(const SequenceHandle & handle) const {
    const auto * slot = find(handle);
    if (slot == nullptr) {
        return std::nullopt;
    }
    return SequenceSnapshot{
        handle,
        slot->lifecycle,
        slot->finish_reason,
        slot->prompt_tokens,
        slot->output_token_budget,
        slot->prompt_tokens_processed,
        slot->decoded_tokens,
        slot->reservation(),
    };
}

bool SequenceEngine::matches(const Slot & slot, const SequenceHandle & handle) const noexcept {
    return slot.lifecycle != SequenceLifecycle::released
        && slot.generation == handle.generation
        && slot.request_id == handle.request_id
        && slot.attempt_id == handle.attempt_id;
}

SequenceHandle SequenceEngine::make_handle(
    const std::size_t index,
    const Slot & slot
) const {
    return {
        static_cast<std::int32_t>(index),
        slot.generation,
        slot.request_id,
        slot.attempt_id,
    };
}

SequenceEngine::Slot * SequenceEngine::find(const SequenceHandle & handle) noexcept {
    if (handle.slot < 0 || static_cast<std::size_t>(handle.slot) >= slots_.size()) {
        return nullptr;
    }
    auto & slot = slots_[static_cast<std::size_t>(handle.slot)];
    return matches(slot, handle) ? &slot : nullptr;
}

const SequenceEngine::Slot * SequenceEngine::find(const SequenceHandle & handle) const noexcept {
    if (handle.slot < 0 || static_cast<std::size_t>(handle.slot) >= slots_.size()) {
        return nullptr;
    }
    const auto & slot = slots_[static_cast<std::size_t>(handle.slot)];
    return matches(slot, handle) ? &slot : nullptr;
}

bool SequenceEngine::duplicate(
    const std::string & request_id,
    const std::string & attempt_id
) const {
    for (const auto & slot : slots_) {
        if (
            slot.lifecycle != SequenceLifecycle::released
            && slot.request_id == request_id
            && slot.attempt_id == attempt_id
        ) {
            return true;
        }
    }
    return false;
}

bool SequenceEngine::runnable(const SequenceLifecycle lifecycle) noexcept {
    return lifecycle == SequenceLifecycle::prefill || lifecycle == SequenceLifecycle::decode;
}

}  // namespace model_worker
