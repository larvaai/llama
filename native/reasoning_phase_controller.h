#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace model_worker {

enum class ReasoningPhase { expect_reasoning_start, reasoning, final, done, error };
enum class ControllerAction { none, reasoning_started, activate_final_grammar, completed, failed };

struct ControllerResult {
    ControllerAction action = ControllerAction::none;
    ReasoningPhase phase = ReasoningPhase::expect_reasoning_start;
    std::string error;
};

class ReasoningPhaseController {
public:
    ReasoningPhaseController(std::vector<std::int32_t> start_marker,
                             std::vector<std::int32_t> end_marker,
                             std::size_t reasoning_budget,
                             std::size_t final_budget,
                             std::size_t total_budget,
                             bool require_start = true);

    ControllerResult consume(std::int32_t token, bool is_eog, bool grammar_accepting);
    ReasoningPhase phase() const noexcept { return phase_; }
    std::size_t sampled_tokens() const noexcept { return sampled_; }
    std::size_t reasoning_tokens() const noexcept { return reasoning_; }
    std::size_t final_tokens() const noexcept { return final_; }

private:
    bool suffix_is(const std::vector<std::int32_t> & marker) const;
    ControllerResult fail(std::string reason);
    void remember(std::int32_t token);

    std::vector<std::int32_t> start_;
    std::vector<std::int32_t> end_;
    std::vector<std::int32_t> tail_;
    std::size_t reasoning_budget_;
    std::size_t final_budget_;
    std::size_t total_budget_;
    bool require_start_;
    ReasoningPhase phase_;
    std::size_t sampled_ = 0;
    std::size_t reasoning_ = 0;
    std::size_t final_ = 0;
};

}  // namespace model_worker
