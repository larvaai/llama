#include "reasoning_phase_controller.h"

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace model_worker {

ReasoningPhaseController::ReasoningPhaseController(
    std::vector<std::int32_t> start_marker,
    std::vector<std::int32_t> end_marker,
    std::size_t reasoning_budget,
    std::size_t final_budget,
    std::size_t total_budget,
    bool require_start)
    : start_(std::move(start_marker)), end_(std::move(end_marker)),
      reasoning_budget_(reasoning_budget), final_budget_(final_budget),
      total_budget_(total_budget), require_start_(require_start),
      phase_(require_start ? ReasoningPhase::expect_reasoning_start : ReasoningPhase::reasoning) {
    if (start_.empty() || end_.empty() || start_ == end_) {
        throw std::invalid_argument("marker sequences must be non-empty and distinct");
    }
    if (reasoning_budget_ > total_budget_ || final_budget_ > total_budget_ ||
        total_budget_ > reasoning_budget_ + final_budget_) {
        throw std::invalid_argument("invalid token budget envelope");
    }
}

void ReasoningPhaseController::remember(std::int32_t token) {
    tail_.push_back(token);
    const auto limit = std::max(start_.size(), end_.size());
    if (tail_.size() > limit) tail_.erase(tail_.begin());
}

bool ReasoningPhaseController::suffix_is(const std::vector<std::int32_t> & marker) const {
    return tail_.size() >= marker.size() &&
           std::equal(marker.rbegin(), marker.rend(), tail_.rbegin());
}

ControllerResult ReasoningPhaseController::fail(std::string reason) {
    phase_ = ReasoningPhase::error;
    return {ControllerAction::failed, phase_, std::move(reason)};
}

ControllerResult ReasoningPhaseController::consume(std::int32_t token, bool is_eog, bool grammar_accepting) {
    if (phase_ == ReasoningPhase::done || phase_ == ReasoningPhase::error) {
        return fail("token received after terminal state");
    }
    ++sampled_;
    if (sampled_ > total_budget_) return fail("total_budget_exhausted");
    if (phase_ != ReasoningPhase::final) {
        ++reasoning_;  // Includes every marker token and all tokens before the required start.
        if (reasoning_ > reasoning_budget_) return fail("reasoning_budget_exhausted");
    } else if (!is_eog) {
        ++final_;
        if (final_ > final_budget_) return fail("final_budget_exhausted");
    }
    remember(token);

    if (is_eog) {
        if (phase_ != ReasoningPhase::final) return fail("eog_before_reasoning_end");
        if (!grammar_accepting) return fail("eog_before_grammar_accepting");
        phase_ = ReasoningPhase::done;
        return {ControllerAction::completed, phase_, {}};
    }

    const bool start = suffix_is(start_);
    const bool end = suffix_is(end_);
    if (phase_ == ReasoningPhase::expect_reasoning_start) {
        if (end) return fail("reasoning_end_before_start");
        if (start) {
            phase_ = ReasoningPhase::reasoning;
            return {ControllerAction::reasoning_started, phase_, {}};
        }
    } else if (phase_ == ReasoningPhase::reasoning) {
        if (start) return fail("duplicate_reasoning_start");
        if (end) {
            phase_ = ReasoningPhase::final;
            return {ControllerAction::activate_final_grammar, phase_, {}};
        }
    } else if (phase_ == ReasoningPhase::final && (start || end)) {
        return fail("reasoning_marker_after_final");
    }
    return {ControllerAction::none, phase_, {}};
}

}  // namespace model_worker
