# Inference runtime reference contracts

`InferencePort` remains the high-level harness boundary. `ManagedBackend` owns a
whole request and its internal scheduler; it is never required to expose sequence
steps. `SteppableBackend` exposes `open_sequence`, bounded `prefill`/`decode`
steps, and explicit `release` for a control plane that owns sequence scheduling.

`DeterministicSchedulerSimulator` is a single-threaded reference policy for tests.
Each tick performs at most one backend action. Selection uses:

```text
service-class priority + floor(age / aging interval) - serviced steps / weight
```

An earlier deadline and then admission order break equal scores. Deadlines and
cancellation are hard terminal boundaries; an opened sequence is released before
its single terminal event. With bounded class priorities, finite sequence work,
and an advancing monotonic clock, the unbounded age term prevents starvation.
Fairness here is per admitted request; hierarchical workflow/agent quotas remain
deferred. The simulator does not implement production locking, continuous
batching, KV allocation, cache reuse, or native preemption.
