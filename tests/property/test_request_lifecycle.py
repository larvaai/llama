from __future__ import annotations

from hypothesis import given, strategies as st

from model_worker.request_registry import Lifecycle, RequestRegistry, TERMINAL


@given(
    st.lists(
        st.sampled_from(("start", "queue_expire", "cancel")),
        min_size=1,
        max_size=20,
    )
)
def test_queue_expiry_cas_never_crosses_a_competing_lifecycle_winner(actions):
    registry = RequestRegistry()
    record = registry.create({}, 100, 200)
    registry.transition(record, Lifecycle.PREFLIGHTED)
    registry.transition(record, Lifecycle.QUEUED)

    for action in actions:
        if action == "start":
            registry.compare_and_transition(
                record,
                Lifecycle.QUEUED,
                Lifecycle.RUNNING,
            )
        elif action == "queue_expire":
            registry.compare_and_transition(
                record,
                Lifecycle.QUEUED,
                Lifecycle.TIMED_OUT,
                error="queue_timeout",
            )
        else:
            registry.cancel(record.request_id)

        terminal_timestamps = set(record.timestamps) & {
            lifecycle.value for lifecycle in TERMINAL
        }
        assert len(terminal_timestamps) <= 1

    if record.lifecycle == Lifecycle.RUNNING:
        terminal = Lifecycle.CANCELLED if record.cancel_event.is_set() else Lifecycle.COMPLETED
        registry.compare_and_transition(record, Lifecycle.RUNNING, terminal)

    assert record.lifecycle in TERMINAL
    terminal_timestamps = set(record.timestamps) & {
        lifecycle.value for lifecycle in TERMINAL
    }
    assert len(terminal_timestamps) == 1
    if record.error == "queue_timeout":
        assert Lifecycle.RUNNING.value not in record.timestamps
