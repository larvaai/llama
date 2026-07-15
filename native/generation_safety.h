#pragma once

#include <string>
#include <string_view>

namespace model_worker {

enum class GrammarSampleSource { unconstrained, final_grammar };

bool grammar_acceptance_proven(bool is_eog, GrammarSampleSource source) noexcept;

struct Utf8AppendResult {
    bool valid = true;
    std::string completed;
};

class Utf8Accumulator {
public:
    Utf8AppendResult append(std::string_view bytes);
    bool finish() const noexcept { return valid_ && pending_.empty(); }
    bool valid() const noexcept { return valid_; }
    const std::string & pending() const noexcept { return pending_; }

private:
    bool valid_ = true;
    std::string pending_;
};

}  // namespace model_worker
