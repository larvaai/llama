from __future__ import annotations

import math
import threading
from collections import Counter
from dataclasses import dataclass
from fractions import Fraction
from typing import Protocol, Sequence

from .contracts import SchedulingMetadata


class SchedulableEntry(Protocol):
    scheduling: SchedulingMetadata
    admitted_at: float
    order: int


@dataclass(frozen=True, slots=True)
class ServiceClassPolicy:
    name: str
    priority: int
    emergency: bool = False

    def __post_init__(self) -> None:
        if type(self.name) is not str or not self.name or len(self.name.encode()) > 64:
            raise ValueError("service class name must be 1..64 UTF-8 bytes")
        if type(self.priority) is not int or abs(self.priority) > 1_000_000:
            raise ValueError("service class priority is invalid")
        if type(self.emergency) is not bool:
            raise TypeError("emergency must be a bool")


@dataclass(frozen=True, slots=True)
class RuntimeSchedulingPolicy:
    service_classes: tuple[ServiceClassPolicy, ...]
    aging_interval_seconds: float = 1.0
    deadline_urgency_window_seconds: float = 2.0
    deadline_priority_boost: int = 8
    emergency_burst_cap: int = 4

    def __post_init__(self) -> None:
        if type(self.service_classes) is not tuple or not self.service_classes:
            raise ValueError("service_classes must be a non-empty tuple")
        names = set()
        emergency_count = 0
        for item in self.service_classes:
            if type(item) is not ServiceClassPolicy:
                raise TypeError("service_classes must contain ServiceClassPolicy values")
            if item.name in names:
                raise ValueError("service class names must be unique")
            names.add(item.name)
            emergency_count += int(item.emergency)
        if emergency_count > 1:
            raise ValueError("at most one service class may use the emergency lane")
        for name in (
            "aging_interval_seconds",
            "deadline_urgency_window_seconds",
        ):
            value = getattr(self, name)
            if type(value) not in {int, float} or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
            object.__setattr__(self, name, float(value))
        if type(self.deadline_priority_boost) is not int or self.deadline_priority_boost < 0:
            raise ValueError("deadline_priority_boost must be a non-negative integer")
        if type(self.emergency_burst_cap) is not int or self.emergency_burst_cap <= 0:
            raise ValueError("emergency_burst_cap must be a positive integer")

    @classmethod
    def defaults(cls) -> RuntimeSchedulingPolicy:
        return cls(
            (
                ServiceClassPolicy("emergency", 16, True),
                ServiceClassPolicy("interactive-critical", 8),
                ServiceClassPolicy("interactive", 4),
                ServiceClassPolicy("throughput", 2),
                ServiceClassPolicy("background", 0),
                ServiceClassPolicy("batch", -2),
            ),
        )

    def class_policy(self, name: str) -> ServiceClassPolicy:
        for item in self.service_classes:
            if item.name == name:
                return item
        raise AdmissionRejection(
            "unknown_service_class",
            f"service class {name!r} is not configured",
            retryable=False,
        )


@dataclass(frozen=True, slots=True)
class FairnessSnapshot:
    workflow_service: tuple[tuple[str, int], ...]
    agent_service: tuple[tuple[tuple[str, str], int], ...]
    request_service: tuple[tuple[str, int], ...]
    consecutive_emergency: int


class HierarchicalFairSelector:
    """Deterministic weighted fair selector with bounded emergency service.

    Priority, deadline urgency and aging decide which class is eligible. Within
    that class, normalized service debt is compared at workflow, agent and
    request level in that order. Roles remain opaque to the inference layer.
    """

    def __init__(self, policy: RuntimeSchedulingPolicy) -> None:
        if type(policy) is not RuntimeSchedulingPolicy:
            raise TypeError("policy must be RuntimeSchedulingPolicy")
        self.policy = policy
        self._workflow_service: Counter[str] = Counter()
        self._agent_service: Counter[tuple[str, str]] = Counter()
        self._request_service: Counter[str] = Counter()
        self._workflow_weight: dict[str, int] = {}
        self._agent_weight: dict[tuple[str, str], int] = {}
        self._consecutive_emergency = 0

    def select(
        self,
        entries: Sequence[SchedulableEntry],
        now: float,
    ) -> SchedulableEntry | None:
        if type(now) not in {int, float} or not math.isfinite(now) or now < 0:
            raise ValueError("now must be a finite non-negative timestamp")
        if not entries:
            return None
        candidates = list(entries)
        for entry in candidates:
            if type(entry.scheduling) is not SchedulingMetadata:
                raise TypeError("entry scheduling metadata is invalid")
            self.policy.class_policy(entry.scheduling.service_class)
            workflow = entry.scheduling.workflow_id
            agent = (workflow, entry.scheduling.agent_id)
            self._workflow_weight[workflow] = max(
                self._workflow_weight.get(workflow, 1),
                entry.scheduling.weight,
            )
            self._agent_weight[agent] = max(
                self._agent_weight.get(agent, 1),
                entry.scheduling.weight,
            )

        emergency = [entry for entry in candidates if self._is_emergency(entry)]
        normal = [entry for entry in candidates if not self._is_emergency(entry)]
        if emergency and (
            not normal
            or self._consecutive_emergency < self.policy.emergency_burst_cap
        ):
            candidates = emergency
        elif normal:
            candidates = normal

        selected = min(candidates, key=lambda entry: self._key(entry, float(now)))
        if self._is_emergency(selected):
            self._consecutive_emergency += 1
        else:
            self._consecutive_emergency = 0
        return selected

    def charge(self, scheduling: SchedulingMetadata, tokens: int) -> None:
        if type(scheduling) is not SchedulingMetadata:
            raise TypeError("scheduling must be SchedulingMetadata")
        if type(tokens) is not int or tokens <= 0:
            raise ValueError("tokens must be a positive integer")
        workflow = scheduling.workflow_id
        agent = (workflow, scheduling.agent_id)
        self._workflow_service[workflow] += tokens
        self._agent_service[agent] += tokens
        self._request_service[scheduling.request_id] += tokens

    def snapshot(self) -> FairnessSnapshot:
        return FairnessSnapshot(
            tuple(sorted(self._workflow_service.items())),
            tuple(sorted(self._agent_service.items())),
            tuple(sorted(self._request_service.items())),
            self._consecutive_emergency,
        )

    def forget(
        self,
        scheduling: SchedulingMetadata,
        *,
        drop_agent: bool,
        drop_workflow: bool,
    ) -> None:
        """Bound ledgers when the scheduler no longer has work for a scope."""
        self._request_service.pop(scheduling.request_id, None)
        agent = (scheduling.workflow_id, scheduling.agent_id)
        if drop_agent:
            self._agent_service.pop(agent, None)
            self._agent_weight.pop(agent, None)
        if drop_workflow:
            self._workflow_service.pop(scheduling.workflow_id, None)
            self._workflow_weight.pop(scheduling.workflow_id, None)

    def _key(self, entry: SchedulableEntry, now: float) -> tuple[object, ...]:
        scheduling = entry.scheduling
        class_policy = self.policy.class_policy(scheduling.service_class)
        waited = max(0.0, now - entry.admitted_at)
        aging = int(waited / self.policy.aging_interval_seconds)
        deadline_boost = 0
        if scheduling.deadline_monotonic is not None:
            slack = scheduling.deadline_monotonic - now
            if slack <= self.policy.deadline_urgency_window_seconds:
                deadline_boost = self.policy.deadline_priority_boost
        effective_priority = class_policy.priority + aging + deadline_boost
        workflow = scheduling.workflow_id
        agent = (workflow, scheduling.agent_id)
        workflow_debt = Fraction(
            self._workflow_service[workflow],
            self._workflow_weight.get(workflow, scheduling.weight),
        )
        agent_debt = Fraction(
            self._agent_service[agent],
            self._agent_weight.get(agent, scheduling.weight),
        )
        request_debt = Fraction(
            self._request_service[scheduling.request_id],
            scheduling.weight,
        )
        return (
            -effective_priority,
            workflow_debt,
            agent_debt,
            request_debt,
            math.inf
            if scheduling.deadline_monotonic is None
            else scheduling.deadline_monotonic,
            entry.order,
        )

    def _is_emergency(self, entry: SchedulableEntry) -> bool:
        return self.policy.class_policy(entry.scheduling.service_class).emergency


class AdmissionRejection(RuntimeError):
    def __init__(self, code: str, detail: str, *, retryable: bool) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail
        self.retryable = retryable


@dataclass(frozen=True, slots=True)
class AdmissionLimits:
    max_pending: int
    max_sequences: int
    kv_token_budget: int
    max_pending_per_workflow: int
    max_pending_per_agent: int
    max_sequences_per_workflow: int
    max_sequences_per_agent: int
    max_kv_tokens_per_workflow: int
    max_kv_tokens_per_agent: int
    load_shed_pending_threshold: int
    load_shed_min_priority: int

    def __post_init__(self) -> None:
        for name in (
            "max_pending",
            "max_sequences",
            "kv_token_budget",
            "max_pending_per_workflow",
            "max_pending_per_agent",
            "max_sequences_per_workflow",
            "max_sequences_per_agent",
            "max_kv_tokens_per_workflow",
            "max_kv_tokens_per_agent",
            "load_shed_pending_threshold",
        ):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if type(self.load_shed_min_priority) is not int:
            raise ValueError("load_shed_min_priority must be an integer")
        if self.max_sequences_per_workflow > self.max_sequences:
            raise ValueError("workflow sequence quota exceeds global capacity")
        if self.max_sequences_per_agent > self.max_sequences_per_workflow:
            raise ValueError("agent sequence quota exceeds workflow quota")
        if self.max_kv_tokens_per_workflow > self.kv_token_budget:
            raise ValueError("workflow KV quota exceeds global capacity")
        if self.max_kv_tokens_per_agent > self.max_kv_tokens_per_workflow:
            raise ValueError("agent KV quota exceeds workflow quota")
        if self.load_shed_pending_threshold > self.max_pending:
            raise ValueError("load shed threshold exceeds pending capacity")


@dataclass(frozen=True, slots=True)
class AdmissionSnapshot:
    pending_requests: int
    active_sequences: int
    reserved_kv_tokens: int
    pending_by_workflow: tuple[tuple[str, int], ...]
    active_by_workflow: tuple[tuple[str, int], ...]
    active_by_agent: tuple[tuple[tuple[str, str], int], ...]


@dataclass(frozen=True, slots=True)
class _Reservation:
    scheduling: SchedulingMetadata
    estimated_tokens: int
    reserved_tokens: int | None = None


class ResourceAdmissionController:
    """Thread-safe queue, hierarchy and KV reservation ledger."""

    def __init__(
        self,
        limits: AdmissionLimits,
        policy: RuntimeSchedulingPolicy,
    ) -> None:
        if type(limits) is not AdmissionLimits:
            raise TypeError("limits must be AdmissionLimits")
        if type(policy) is not RuntimeSchedulingPolicy:
            raise TypeError("policy must be RuntimeSchedulingPolicy")
        self.limits = limits
        self.policy = policy
        self._lock = threading.RLock()
        self._reservations: dict[str, _Reservation] = {}

    def admit(self, scheduling: SchedulingMetadata, estimated_tokens: int) -> None:
        if type(scheduling) is not SchedulingMetadata:
            raise TypeError("scheduling must be SchedulingMetadata")
        if type(estimated_tokens) is not int or estimated_tokens <= 0:
            raise ValueError("estimated_tokens must be a positive integer")
        if estimated_tokens > self.limits.kv_token_budget:
            raise AdmissionRejection(
                "context_overflow",
                "request estimate exceeds the runtime KV budget",
                retryable=False,
            )
        if estimated_tokens > self.limits.max_kv_tokens_per_workflow:
            raise AdmissionRejection(
                "workflow_kv_quota",
                "request estimate exceeds the per-workflow KV quota",
                retryable=False,
            )
        if estimated_tokens > self.limits.max_kv_tokens_per_agent:
            raise AdmissionRejection(
                "agent_kv_quota",
                "request estimate exceeds the per-agent KV quota",
                retryable=False,
            )
        class_policy = self.policy.class_policy(scheduling.service_class)
        with self._lock:
            if scheduling.request_id in self._reservations:
                raise AdmissionRejection(
                    "duplicate_request",
                    "request is already registered",
                    retryable=False,
                )
            pending = self._pending_locked()
            if len(pending) >= self.limits.max_pending:
                raise AdmissionRejection(
                    "queue_full",
                    "inference admission queue is full",
                    retryable=True,
                )
            if (
                len(pending) >= self.limits.load_shed_pending_threshold
                and class_policy.priority < self.limits.load_shed_min_priority
                and not class_policy.emergency
            ):
                raise AdmissionRejection(
                    "overloaded",
                    "low-priority work is being shed under saturation",
                    retryable=True,
                )
            workflow_pending = sum(
                item.scheduling.workflow_id == scheduling.workflow_id
                for item in pending
            )
            agent_pending = sum(
                item.scheduling.workflow_id == scheduling.workflow_id
                and item.scheduling.agent_id == scheduling.agent_id
                for item in pending
            )
            if workflow_pending >= self.limits.max_pending_per_workflow:
                raise AdmissionRejection(
                    "workflow_queue_quota",
                    "workflow pending quota is exhausted",
                    retryable=True,
                )
            if agent_pending >= self.limits.max_pending_per_agent:
                raise AdmissionRejection(
                    "agent_queue_quota",
                    "agent pending quota is exhausted",
                    retryable=True,
                )
            self._reservations[scheduling.request_id] = _Reservation(
                scheduling,
                estimated_tokens,
            )

    def can_activate(self, request_id: str) -> bool:
        with self._lock:
            item = self._reservations.get(request_id)
            if item is None or item.reserved_tokens is not None:
                return False
            active = self._active_locked()
            if len(active) >= self.limits.max_sequences:
                return False
            workflow = item.scheduling.workflow_id
            agent = (workflow, item.scheduling.agent_id)
            workflow_active = [
                current
                for current in active
                if current.scheduling.workflow_id == workflow
            ]
            agent_active = [
                current
                for current in workflow_active
                if current.scheduling.agent_id == agent[1]
            ]
            if len(workflow_active) >= self.limits.max_sequences_per_workflow:
                return False
            if len(agent_active) >= self.limits.max_sequences_per_agent:
                return False
            global_kv = sum(current.reserved_tokens or 0 for current in active)
            workflow_kv = sum(current.reserved_tokens or 0 for current in workflow_active)
            agent_kv = sum(current.reserved_tokens or 0 for current in agent_active)
            return (
                global_kv + item.estimated_tokens <= self.limits.kv_token_budget
                and workflow_kv + item.estimated_tokens
                <= self.limits.max_kv_tokens_per_workflow
                and agent_kv + item.estimated_tokens
                <= self.limits.max_kv_tokens_per_agent
            )

    def activate(self, request_id: str, reserved_tokens: int | None = None) -> None:
        with self._lock:
            item = self._reservations.get(request_id)
            if item is None:
                raise AdmissionRejection(
                    "unknown_request",
                    "request is not registered",
                    retryable=False,
                )
            if item.reserved_tokens is not None:
                raise AdmissionRejection(
                    "state_conflict",
                    "request already owns a sequence reservation",
                    retryable=False,
                )
            actual = item.estimated_tokens if reserved_tokens is None else reserved_tokens
            if type(actual) is not int or actual <= 0:
                raise ValueError("reserved_tokens must be a positive integer")
            if actual > item.estimated_tokens:
                # The estimate is deliberately conservative. Fail closed if a
                # backend reports otherwise instead of silently overcommitting.
                raise AdmissionRejection(
                    "reservation_underestimated",
                    "backend reservation exceeds the admission estimate",
                    retryable=False,
                )
            self._reservations[request_id] = _Reservation(
                item.scheduling,
                item.estimated_tokens,
                actual,
            )

    def release(self, request_id: str) -> bool:
        with self._lock:
            return self._reservations.pop(request_id, None) is not None

    def snapshot(self) -> AdmissionSnapshot:
        with self._lock:
            pending = self._pending_locked()
            active = self._active_locked()
            pending_workflow = Counter(
                item.scheduling.workflow_id for item in pending
            )
            active_workflow = Counter(
                item.scheduling.workflow_id for item in active
            )
            active_agent = Counter(
                (item.scheduling.workflow_id, item.scheduling.agent_id)
                for item in active
            )
            return AdmissionSnapshot(
                len(pending),
                len(active),
                sum(item.reserved_tokens or 0 for item in active),
                tuple(sorted(pending_workflow.items())),
                tuple(sorted(active_workflow.items())),
                tuple(sorted(active_agent.items())),
            )

    def _pending_locked(self) -> list[_Reservation]:
        return [
            item for item in self._reservations.values() if item.reserved_tokens is None
        ]

    def _active_locked(self) -> list[_Reservation]:
        return [
            item for item in self._reservations.values() if item.reserved_tokens is not None
        ]
