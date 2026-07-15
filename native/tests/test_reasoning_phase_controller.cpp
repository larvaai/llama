#include "reasoning_phase_controller.h"

#include <cassert>
#include <iostream>
#include <stdexcept>

using model_worker::ControllerAction;
using model_worker::ReasoningPhaseController;

static void happy_multi_token() {
    ReasoningPhaseController fsm({1, 2}, {3, 4}, 8, 4, 10);
    assert(fsm.consume(1, false, false).action == ControllerAction::none);
    assert(fsm.consume(2, false, false).action == ControllerAction::reasoning_started);
    assert(fsm.consume(9, false, false).action == ControllerAction::none);
    assert(fsm.consume(3, false, false).action == ControllerAction::none);
    assert(fsm.consume(4, false, false).action == ControllerAction::activate_final_grammar);
    assert(fsm.consume(8, false, false).action == ControllerAction::none);
    assert(fsm.consume(0, true, true).action == ControllerAction::completed);
}

static void violations() {
    ReasoningPhaseController end_first({1}, {2}, 5, 3, 7);
    assert(end_first.consume(2, false, false).action == ControllerAction::failed);
    ReasoningPhaseController duplicate({1}, {2}, 5, 3, 7);
    duplicate.consume(1, false, false);
    assert(duplicate.consume(1, false, false).action == ControllerAction::failed);
    ReasoningPhaseController early_eog({1}, {2}, 5, 3, 7);
    assert(early_eog.consume(0, true, false).action == ControllerAction::failed);
    ReasoningPhaseController grammar({1}, {2}, 5, 3, 7);
    grammar.consume(1, false, false); grammar.consume(2, false, false);
    assert(grammar.consume(0, true, false).action == ControllerAction::failed);
}

int main() {
    happy_multi_token();
    violations();
    std::cout << "reasoning controller tests passed\n";
}
