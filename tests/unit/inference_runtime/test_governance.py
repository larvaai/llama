from __future__ import annotations

from dataclasses import dataclass

import pytest

from inference_runtime import (
    AdmissionLimits,
    AdmissionRejection,
    HierarchicalFairSelector,
    ResourceAdmissionController,
    RuntimeSchedulingPolicy,
    SchedulingMetadata,
    ServiceClassPolicy,
)


@dataclass
class Entry:
    scheduling: SchedulingMetadata
    admitted_at: float
    order: int


def metadata(
    request_id,
    *,
    workflow="workflow",
    agent="agent",
    service_class="throughput",
    weight=1,
    deadline=None,
):
    return SchedulingMetadata(
        request_id,
        workflow,
        agent,
        service_class,
        weight,
        deadline,
    )


def policy(**overrides):
    values = {
        "service_classes": (
            ServiceClassPolicy("emergency", 16, True),
            ServiceClassPolicy("interactive", 4),
            ServiceClassPolicy("throughput", 1),
            ServiceClassPolicy("background", 0),
        ),
        "aging_interval_seconds": 1,
        "deadline_urgency_window_seconds": 2,
        "deadline_priority_boost": 8,
        "emergency_burst_cap": 2,
    }
    values.update(overrides)
    return RuntimeSchedulingPolicy(**values)


def limits(**overrides):
    values = {
        "max_pending": 8,
        "max_sequences": 4,
        "kv_token_budget": 400,
        "max_pending_per_workflow": 4,
        "max_pending_per_agent": 3,
        "max_sequences_per_workflow": 3,
        "max_sequences_per_agent": 2,
        "max_kv_tokens_per_workflow": 300,
        "max_kv_tokens_per_agent": 200,
        "load_shed_pending_threshold": 4,
        "load_shed_min_priority": 1,
    }
    values.update(overrides)
    return AdmissionLimits(**values)


def test_hierarchical_weighted_service_rotates_workflows_agents_and_requests():
    selector = HierarchicalFairSelector(policy())
    entries = [
        Entry(metadata("a1", workflow="a", agent="one"), 0, 0),
        Entry(metadata("a2", workflow="a", agent="two"), 0, 1),
        Entry(metadata("b1", workflow="b", agent="one"), 0, 2),
    ]
    chosen = []
    for _ in range(6):
        entry = selector.select(entries, 0)
        chosen.append(entry.scheduling.request_id)
        selector.charge(entry.scheduling, 1)
    assert chosen == ["a1", "b1", "a2", "b1", "a1", "b1"]
    snapshot = selector.snapshot()
    assert dict(snapshot.workflow_service) == {"a": 3, "b": 3}
    assert dict(snapshot.request_service) == {"a1": 2, "a2": 1, "b1": 3}


def test_priority_deadline_and_aging_are_explicit_and_starvation_bounded():
    selector = HierarchicalFairSelector(policy(aging_interval_seconds=1))
    background = Entry(metadata("old", service_class="background"), 0, 0)
    interactive = Entry(metadata("new", service_class="interactive"), 9, 1)
    assert selector.select([background, interactive], 9) is background

    urgent = Entry(
        metadata("urgent", service_class="throughput", deadline=10),
        9,
        2,
    )
    assert selector.select([interactive, urgent], 9) is urgent


def test_emergency_lane_is_capped_when_normal_work_is_eligible():
    selector = HierarchicalFairSelector(policy(emergency_burst_cap=2))
    emergency = Entry(metadata("e", service_class="emergency"), 0, 0)
    normal = Entry(metadata("n", service_class="throughput"), 0, 1)
    selected = []
    for _ in range(6):
        entry = selector.select([emergency, normal], 0)
        selected.append(entry.scheduling.request_id)
        selector.charge(entry.scheduling, 1)
    assert selected == ["e", "e", "n", "e", "e", "n"]


def test_admission_enforces_queue_hierarchy_load_shedding_and_kv_ledger():
    controller = ResourceAdmissionController(limits(), policy())
    for index in range(3):
        controller.admit(
            metadata(f"a-{index}", workflow="w", agent="a"),
            90,
        )
    with pytest.raises(AdmissionRejection) as agent_quota:
        controller.admit(metadata("a-3", workflow="w", agent="a"), 90)
    assert agent_quota.value.code == "agent_queue_quota"

    controller.admit(metadata("b-0", workflow="w", agent="b"), 90)
    with pytest.raises(AdmissionRejection) as workflow_quota:
        controller.admit(metadata("other", workflow="w", agent="c"), 90)
    assert workflow_quota.value.code == "workflow_queue_quota"

    assert controller.can_activate("a-0")
    controller.activate("a-0", 80)
    assert controller.can_activate("a-1")
    controller.activate("a-1", 80)
    assert not controller.can_activate("a-2")
    snapshot = controller.snapshot()
    assert snapshot.active_sequences == 2
    assert snapshot.reserved_kv_tokens == 160
    assert controller.release("a-0")
    assert controller.can_activate("a-2")


def test_admission_fails_closed_on_underestimate_and_sheds_only_low_priority():
    controller = ResourceAdmissionController(
        limits(load_shed_pending_threshold=2),
        policy(),
    )
    controller.admit(metadata("one"), 100)
    controller.admit(metadata("two"), 100)
    with pytest.raises(AdmissionRejection) as shed:
        controller.admit(metadata("low", service_class="background"), 50)
    assert shed.value.code == "overloaded" and shed.value.retryable
    controller.admit(metadata("high", service_class="interactive"), 50)

    with pytest.raises(AdmissionRejection) as underestimate:
        controller.activate("one", 101)
    assert underestimate.value.code == "reservation_underestimated"


def test_admission_rejects_request_that_can_never_fit_hierarchy_kv_quota():
    agent_limited = ResourceAdmissionController(limits(), policy())
    with pytest.raises(AdmissionRejection) as agent_quota:
        agent_limited.admit(metadata("too-large-for-agent"), 201)
    assert agent_quota.value.code == "agent_kv_quota"
    assert agent_limited.snapshot().pending_requests == 0

    workflow_limited = ResourceAdmissionController(
        limits(max_kv_tokens_per_agent=300),
        policy(),
    )
    with pytest.raises(AdmissionRejection) as workflow_quota:
        workflow_limited.admit(metadata("too-large-for-workflow"), 301)
    assert workflow_quota.value.code == "workflow_kv_quota"
    assert workflow_limited.snapshot().pending_requests == 0
