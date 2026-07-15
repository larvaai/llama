#include "ipc_protocol.h"
#include "generation_safety.h"
#include "pending_cancel_registry.h"
#include "reasoning_phase_controller.h"
#include "sequence_engine.h"

#include <chrono>
#include <cstdint>
#include <functional>
#include <initializer_list>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using model_worker::ControllerAction;
using model_worker::FrameIdentity;
using model_worker::GrammarSampleSource;
using model_worker::PendingCancelRegistry;
using model_worker::ReasoningPhase;
using model_worker::ReasoningPhaseController;
using model_worker::AdmissionStatus;
using model_worker::SequenceEngine;
using model_worker::SequenceFinishReason;
using model_worker::SequenceLifecycle;
using model_worker::SequenceOperationStatus;
using model_worker::SequenceReleaseStatus;
using model_worker::Utf8Accumulator;

static void check(bool condition, const std::string & message) {
    if (!condition) throw std::runtime_error(message);
}

static void expect_invalid(std::function<void()> operation, const std::string & name) {
    try {
        operation();
    } catch (const std::invalid_argument &) {
        return;
    }
    throw std::runtime_error(name + " did not reject invalid construction");
}

static void happy_multi_token() {
    ReasoningPhaseController fsm({1, 2}, {3, 4}, 8, 4, 10);
    check(fsm.consume(1, false, false).action == ControllerAction::none, "partial start");
    check(fsm.consume(2, false, false).action == ControllerAction::reasoning_started, "start");
    check(fsm.consume(9, false, false).action == ControllerAction::none, "reasoning token");
    check(fsm.consume(3, false, false).action == ControllerAction::none, "partial end");
    check(fsm.consume(4, false, false).action == ControllerAction::activate_final_grammar, "end");
    check(fsm.consume(8, false, false).action == ControllerAction::none, "final token");
    const auto completed = fsm.consume(0, true, true);
    check(completed.action == ControllerAction::completed, "accepting EOG");
    check(completed.phase == ReasoningPhase::done, "completed phase");
    check(fsm.sampled_tokens() == 7 && fsm.reasoning_tokens() == 5 && fsm.final_tokens() == 1,
          "token accounting");
}

static void protocol_violations() {
    ReasoningPhaseController end_first({1}, {2}, 5, 3, 7);
    check(end_first.consume(2, false, false).error == "reasoning_end_before_start", "end before start");

    ReasoningPhaseController duplicate({1}, {2}, 5, 3, 7);
    duplicate.consume(1, false, false);
    check(duplicate.consume(1, false, false).error == "duplicate_reasoning_start", "duplicate start");

    ReasoningPhaseController missing_start({1}, {2}, 5, 3, 7);
    check(missing_start.consume(0, true, false).error == "eog_before_reasoning_end", "missing start");

    ReasoningPhaseController missing_end({1}, {2}, 5, 3, 7);
    missing_end.consume(1, false, false);
    check(missing_end.consume(0, true, false).error == "eog_before_reasoning_end", "missing end");

    ReasoningPhaseController grammar({1}, {2}, 5, 3, 7);
    grammar.consume(1, false, false);
    grammar.consume(2, false, false);
    check(grammar.consume(0, true, false).error == "eog_before_grammar_accepting", "non-accepting EOG");

    ReasoningPhaseController marker_after_final({1}, {2}, 5, 3, 7);
    marker_after_final.consume(1, false, false);
    marker_after_final.consume(2, false, false);
    check(marker_after_final.consume(1, false, false).error == "reasoning_marker_after_final",
          "marker after final");
}

static void budget_and_terminal_boundaries() {
    ReasoningPhaseController exact({1}, {2}, 3, 1, 4);
    exact.consume(1, false, false);
    exact.consume(2, false, false);
    exact.consume(9, false, false);
    check(exact.consume(0, true, true).action == ControllerAction::completed, "exact budgets");
    check(exact.consume(7, false, false).error == "token received after terminal state", "terminal state");

    ReasoningPhaseController reasoning_exhausted({1}, {2}, 2, 2, 4);
    reasoning_exhausted.consume(1, false, false);
    reasoning_exhausted.consume(9, false, false);
    check(reasoning_exhausted.consume(9, false, false).error == "reasoning_budget_exhausted",
          "reasoning boundary");

    ReasoningPhaseController final_exhausted({1}, {2}, 3, 1, 4);
    final_exhausted.consume(1, false, false);
    final_exhausted.consume(2, false, false);
    final_exhausted.consume(9, false, false);
    check(final_exhausted.consume(8, false, false).error == "final_budget_exhausted", "final boundary");

    ReasoningPhaseController total_exhausted({1}, {2}, 3, 3, 3);
    total_exhausted.consume(1, false, false);
    total_exhausted.consume(2, false, false);
    total_exhausted.consume(9, false, false);
    check(total_exhausted.consume(0, true, true).error == "total_budget_exhausted", "total boundary");
}

static void invalid_construction_and_ipc_identity() {
    expect_invalid([] { ReasoningPhaseController({}, {2}, 2, 1, 3); }, "empty start marker");
    expect_invalid([] { ReasoningPhaseController({1}, {}, 2, 1, 3); }, "empty end marker");
    expect_invalid([] { ReasoningPhaseController({1}, {1}, 2, 1, 3); }, "equal markers");
    expect_invalid([] { ReasoningPhaseController({1}, {2}, 4, 1, 3); }, "reasoning over total");

    check(model_worker::valid_frame_identity({"model-worker-ipc.v1", "request", "attempt", 0}),
          "valid IPC identity");
    check(!model_worker::valid_frame_identity({"wrong", "request", "attempt", 0}), "IPC version");
    check(!model_worker::valid_frame_identity({"model-worker-ipc.v1", "", "attempt", 0}), "empty request ID");
    check(!model_worker::valid_frame_identity({"model-worker-ipc.v1", "request", "", 0}), "empty attempt ID");
}

static std::string bytes(std::initializer_list<unsigned int> values) {
    std::string result;
    for (const auto value : values) result.push_back(static_cast<char>(value));
    return result;
}

static void grammar_termination_and_utf8_safety() {
    check(!model_worker::grammar_acceptance_proven(true, GrammarSampleSource::unconstrained),
          "unconstrained EOG is not grammar acceptance");
    check(!model_worker::grammar_acceptance_proven(false, GrammarSampleSource::final_grammar),
          "non-EOG is not termination");
    check(model_worker::grammar_acceptance_proven(true, GrammarSampleSource::final_grammar),
          "final grammar sampled EOG proves acceptance");

    Utf8Accumulator split;
    check(split.append(bytes({0xE2})).valid && !split.finish(), "split prefix incomplete");
    check(split.append(bytes({0x82})).valid && !split.finish(), "split continuation incomplete");
    const auto completed = split.append(bytes({0xAC}));
    check(completed.valid && completed.completed == bytes({0xE2, 0x82, 0xAC}), "split euro complete");
    check(split.append(bytes({0xF0, 0x9F, 0x98, 0x80})).valid && split.finish(), "valid scalar");

    const std::vector<std::string> invalid = {
        bytes({0x80}),                   // stray continuation
        bytes({0xC0, 0x80}),             // overlong two-byte
        bytes({0xE0, 0x80, 0x80}),       // overlong three-byte
        bytes({0xED, 0xA0, 0x80}),       // UTF-16 surrogate
        bytes({0xF4, 0x90, 0x80, 0x80}), // above U+10FFFF
        bytes({0xE2, 0x28, 0xA1}),       // invalid continuation
    };
    for (const auto & candidate : invalid) {
        Utf8Accumulator accumulator;
        check(!accumulator.append(candidate).valid && !accumulator.finish(), "invalid UTF-8 rejected");
    }
    Utf8Accumulator incomplete;
    check(incomplete.append(bytes({0xF0, 0x9F})).valid && !incomplete.finish(),
          "trailing incomplete UTF-8 rejected at finish");
}

static void pending_cancel_registry_is_bounded_and_expires() {
    using namespace std::chrono_literals;
    PendingCancelRegistry registry(2, 10ms);
    const auto now = PendingCancelRegistry::Clock::time_point{};
    registry.add({"request-1", "attempt-1"}, now);
    registry.add({"request-2", "attempt-2"}, now + 1ms);
    registry.add({"request-3", "attempt-3"}, now + 2ms);
    check(registry.size(now + 2ms) == 2, "pending cancel registry stays bounded");
    check(!registry.consume({"request-1", "attempt-1"}, now + 2ms), "oldest pending cancel evicted");
    check(registry.consume({"request-2", "attempt-2"}, now + 2ms), "pending cancel consumed once");
    check(!registry.consume({"request-2", "attempt-2"}, now + 2ms), "consumed cancel removed");
    check(registry.size(now + 20ms) == 0, "stale pending cancel expires");

    expect_invalid([] { PendingCancelRegistry invalid(0); }, "zero pending cancel capacity");
}

static void sequence_engine_interleaves_and_releases() {
    SequenceEngine engine(8, 512);
    std::vector<model_worker::SequenceHandle> handles;
    for (int index = 0; index < 8; ++index) {
        const auto admitted = engine.admit(
            "request-" + std::to_string(index),
            "attempt-" + std::to_string(index),
            16,
            16
        );
        check(admitted.status == AdmissionStatus::admitted && admitted.handle.has_value(),
              "sequence admitted");
        handles.push_back(*admitted.handle);
        check(engine.start_prefill(handles.back()) == SequenceOperationStatus::applied,
              "sequence enters prefill");
    }
    check(engine.active_sequences() == 8 && engine.reserved_tokens() == 256,
          "sequence resource ledger reserves capacity");

    std::vector<int> visits(8, 0);
    for (int iteration = 0; iteration < 16; ++iteration) {
        const auto next = engine.next_runnable();
        check(next.has_value(), "round robin returns runnable sequence");
        ++visits[static_cast<std::size_t>(next->slot)];
        const auto snapshot = engine.snapshot(*next);
        check(snapshot.has_value(), "round robin handle remains live");
        if (snapshot->lifecycle == SequenceLifecycle::prefill) {
            check(engine.advance_prefill(*next, 8) == SequenceOperationStatus::applied,
                  "chunked prefill advances independently");
        }
    }
    for (const auto count : visits) {
        check(count == 2, "round robin does not starve any of eight sequences");
    }

    for (std::size_t index = 0; index < handles.size(); ++index) {
        if (index == 3) {
            check(engine.cancel(handles[index]) == SequenceOperationStatus::applied,
                  "one sequence cancels");
        } else {
            check(engine.advance_decode(handles[index], 1) == SequenceOperationStatus::applied,
                  "other sequence decodes after peer cancellation");
            check(engine.finish(handles[index], SequenceFinishReason::stop)
                      == SequenceOperationStatus::applied,
                  "other sequence finishes independently");
        }
    }

    for (const auto & handle : handles) {
        const auto released = engine.release(handle);
        check(released.status == SequenceReleaseStatus::released && released.released_tokens == 32,
              "terminal sequence releases exact reservation");
        check(engine.release(handle).status == SequenceReleaseStatus::already_released,
              "release is idempotent for current generation");
    }
    check(engine.active_sequences() == 0 && engine.reserved_tokens() == 0,
          "all sequence resources reconcile to zero");
}

static void sequence_engine_rejects_capacity_and_stale_handles() {
    SequenceEngine engine(2, 12);
    const auto first = engine.admit("r1", "a1", 4, 2);
    check(first.status == AdmissionStatus::admitted, "first capacity admission");
    check(engine.admit("r1", "a1", 1, 1).status == AdmissionStatus::duplicate,
          "duplicate request attempt rejected");
    check(engine.admit("r2", "a2", 5, 2).status == AdmissionStatus::token_capacity,
          "token ledger rejects overcommit");
    const auto second = engine.admit("r2", "a2", 3, 3);
    check(second.status == AdmissionStatus::admitted, "second exact-capacity admission");
    check(engine.admit("r3", "a3", 1, 1).status == AdmissionStatus::sequence_capacity,
          "sequence slot capacity enforced");

    check(engine.start_prefill(*first.handle) == SequenceOperationStatus::applied,
          "first starts");
    check(engine.advance_prefill(*first.handle, 4) == SequenceOperationStatus::applied,
          "first prefill completes");
    check(engine.finish(*first.handle, SequenceFinishReason::stop)
              == SequenceOperationStatus::applied,
          "first terminates");
    check(engine.release(*first.handle).status == SequenceReleaseStatus::released,
          "first releases");

    const auto reused = engine.admit("r3", "a3", 2, 2);
    check(reused.status == AdmissionStatus::admitted, "released slot can be reused");
    check(reused.handle->slot == first.handle->slot
              && reused.handle->generation != first.handle->generation,
          "slot reuse increments generation");
    check(engine.cancel(*first.handle) == SequenceOperationStatus::stale_handle,
          "stale handle cannot cancel new occupant");
    check(engine.snapshot(*reused.handle)->lifecycle == SequenceLifecycle::admitted,
          "new occupant state is not poisoned by stale handle");

    expect_invalid([] { SequenceEngine invalid(0, 1); }, "zero sequence capacity");
    expect_invalid([] { SequenceEngine invalid(1, 0); }, "zero token capacity");
}

int main() {
    try {
        happy_multi_token();
        protocol_violations();
        budget_and_terminal_boundaries();
        invalid_construction_and_ipc_identity();
        grammar_termination_and_utf8_safety();
        pending_cancel_registry_is_bounded_and_expires();
        sequence_engine_interleaves_and_releases();
        sequence_engine_rejects_capacity_and_stale_handles();
        std::cout << "reasoning controller tests passed\n";
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "reasoning controller test failed: " << error.what() << "\n";
        return 1;
    }
}
