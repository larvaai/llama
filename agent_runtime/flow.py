from __future__ import annotations

from .contracts import AgentRunLifecycle, FlowPolicy, FlowTransition, ToolResultEnvelope


class CodeOwnedFlowController:
    def decide(self, result: ToolResultEnvelope, *, verify: bool = True) -> FlowTransition:
        if result.status.value == "unknown_outcome":
            return FlowTransition(FlowPolicy.HANDOFF, AgentRunLifecycle.PAUSED, "unknown outcome requires reconciliation")
        if result.status.value == "failed":
            return FlowTransition(FlowPolicy.REPLAN_WITH_OBSERVATION, AgentRunLifecycle.DECIDING, "tool failed")
        if verify:
            return FlowTransition(FlowPolicy.VERIFY_THEN_REPLAN, AgentRunLifecycle.VERIFYING, "deterministic verification required")
        return FlowTransition(FlowPolicy.SYNTHESIZE_NO_TOOLS, AgentRunLifecycle.SYNTHESIZING, "bounded observation is ready")
