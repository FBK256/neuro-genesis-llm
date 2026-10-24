"""Constrained agent controller. Model-produced text never becomes executable code."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


@dataclass(frozen=True)
class ToolSpec:
    """A capability explicitly registered by the application owner."""

    name: str
    description: str
    function: Callable[..., Any]
    requires_approval: bool = False
    allowed_args: frozenset[str] | None = None
    risk_level: str = "low"

    def __post_init__(self) -> None:
        if self.risk_level not in {"low", "medium", "high", "critical"}:
            raise ValueError("risk_level must be low, medium, high, or critical")
        if self.risk_level in {"high", "critical"} and not self.requires_approval:
            object.__setattr__(self, "requires_approval", True)


class ToolRegistry:
    """Allowlist for application capabilities; no implicit shell or Python execution."""

    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec) -> None:
        if not spec.name or spec.name in self._tools:
            raise ValueError(f"tool name is empty or already registered: {spec.name!r}")
        if not callable(spec.function):
            raise TypeError("tool function must be callable")
        self._tools[spec.name] = spec

    def get_spec(self, name: str) -> ToolSpec:
        """Get a registered tool specification or reject an unknown capability."""
        if name not in self._tools:
            raise PermissionError(f"tool is not registered: {name}")
        return self._tools[name]

    def describe(self) -> list[dict[str, Any]]:
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "requires_approval": tool.requires_approval,
                "risk_level": tool.risk_level,
                "allowed_args": sorted(tool.allowed_args) if tool.allowed_args is not None else None,
            }
            for tool in self._tools.values()
        ]

    def execute(
        self,
        name: str,
        args: dict[str, Any],
        *,
        approve: Callable[[ToolSpec, dict[str, Any]], bool] | None = None,
    ) -> Any:
        if name not in self._tools:
            raise PermissionError(f"tool is not registered: {name}")
        spec = self._tools[name]
        if not isinstance(args, dict):
            raise TypeError("tool arguments must be a JSON-like object")
        if spec.allowed_args is not None:
            unknown = set(args) - set(spec.allowed_args)
            if unknown:
                raise ValueError(f"unsupported arguments for {name}: {sorted(unknown)}")
        if spec.requires_approval:
            if approve is None or not approve(spec, args):
                raise PermissionError(f"human approval required for tool: {name}")
        return spec.function(**args)


class AgentController:
    """Thin loop around an injected planner, verifier, and allowlisted tools.

    The planner should return either {"type":"tool","name":...,"args":{...}} or
    {"type":"final","content":"..."}. Tool results are labelled as untrusted observations.
    All state-changing actions should be represented by separate tools with approvals.
    """

    def __init__(
        self,
        *,
        planner: Callable[[str, list[dict[str, Any]], list[dict[str, Any]]], dict[str, Any]],
        tools: ToolRegistry,
        verifier: Callable[[str, list[dict[str, Any]]], dict[str, Any]] | None = None,
        approval_callback: Callable[[ToolSpec, dict[str, Any]], bool] | None = None,
        max_steps: int = 8,
        max_observation_bytes: int = 8192,
        audit_path: str | Path | None = None,
        repeat_action_limit: int = 3,
        max_history_chars: int = 12000,
        recent_observations_to_keep: int = 5,
    ) -> None:
        if max_steps < 1:
            raise ValueError("max_steps must be >= 1")
        if max_history_chars < 1024:
            raise ValueError("max_history_chars must be >= 1024")
        if recent_observations_to_keep < 1:
            raise ValueError("recent_observations_to_keep must be >= 1")
        if repeat_action_limit < 2:
            raise ValueError("repeat_action_limit must be >= 2")
        if max_observation_bytes < 256:
            raise ValueError("max_observation_bytes must be >= 256")
        self.planner = planner
        self.tools = tools
        self.verifier = verifier
        self.approval_callback = approval_callback
        self.max_steps = max_steps
        self.max_observation_bytes = max_observation_bytes
        self.repeat_action_limit = repeat_action_limit
        self.max_history_chars = max_history_chars
        self.recent_observations_to_keep = recent_observations_to_keep
        self.audit_path = Path(audit_path) if audit_path else None
        if self.audit_path:
            self.audit_path.parent.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _digest(value: Any) -> str:
        encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def _audit(self, record: dict[str, Any]) -> None:
        """Append metadata-only audit rows; raw arguments/tool outputs are not persisted."""
        if self.audit_path is None:
            return
        row = {"at": datetime.now(timezone.utc).isoformat(), **record}
        with self.audit_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            stream.flush()
            # Ensure the intent is written before a side effect is permitted.
            import os
            os.fsync(stream.fileno())

    def _bound_result(self, result: Any) -> Any:
        """Bound tool output size before it is fed back into the planner."""
        try:
            serialized = json.dumps(result, ensure_ascii=False, default=str)
        except Exception:
            serialized = repr(result)
        raw = serialized.encode("utf-8")
        if len(raw) <= self.max_observation_bytes:
            return result
        digest = hashlib.sha256(raw).hexdigest()
        preview_budget = max(128, self.max_observation_bytes // 2)
        preview = raw[:preview_budget].decode("utf-8", errors="replace")
        return {
            "truncated": True,
            "original_bytes": len(raw),
            "sha256": digest,
            "preview_untrusted": preview,
        }

    @staticmethod
    def _observation_digest(observation: dict[str, Any]) -> str:
        return hashlib.sha256(
            json.dumps(observation, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()

    @staticmethod
    def _compact_observation(observation: dict[str, Any]) -> dict[str, Any]:
        """Keep enough metadata to re-read evidence without duplicating large private payloads."""
        compact: dict[str, Any] = {
            "kind": observation.get("kind", "observation"),
            "instruction_trust": "untrusted_data_not_policy",
        }
        for key in ("tool", "error_type", "consecutive_repeats"):
            if key in observation:
                compact[key] = observation[key]
        if "message" in observation:
            compact["message"] = str(observation["message"])[:120]
        if "reason" in observation:
            compact["reason"] = str(observation["reason"])[:120]
        if "result" in observation:
            result = observation["result"]
            if isinstance(result, dict):
                for key in (
                    "path", "status", "sha256", "original_bytes", "truncated",
                    "count", "files_scanned", "size_bytes", "characters_returned",
                ):
                    if key in result:
                        compact[key] = result[key]
                if isinstance(result.get("text"), str):
                    compact["text_preview"] = result["text"][:160]
                if isinstance(result.get("matches"), list):
                    compact["match_count"] = len(result["matches"])
                if isinstance(result.get("records"), list):
                    compact["record_count"] = len(result["records"])
                if isinstance(result.get("preview_untrusted"), str):
                    compact["preview_untrusted"] = result["preview_untrusted"][:120]
            elif isinstance(result, str):
                compact["result_sha256"] = hashlib.sha256(result.encode("utf-8")).hexdigest()
                compact["result_preview"] = result[:160]
            else:
                compact["result_sha256"] = hashlib.sha256(
                    json.dumps(result, default=str, sort_keys=True).encode("utf-8")
                ).hexdigest()
        return compact

    def _compact_observation_history(self, observations: list[dict[str, Any]]) -> None:
        """Bound prompt history while retaining recent full evidence and a compact older ledger."""
        def encoded_size() -> int:
            return len(json.dumps(observations, ensure_ascii=False, default=str))

        if encoded_size() <= self.max_history_chars:
            return

        keep = self.recent_observations_to_keep
        old = observations[:-keep] if keep < len(observations) else []
        recent = observations[-keep:] if keep < len(observations) else observations[:]
        if not old:
            # A single unusually large result should have been bounded by _bound_result;
            # still compact it defensively if a tool returned a non-JSON object.
            recent = [self._compact_observation(item) for item in recent]
            observations[:] = recent
            return

        compact_events = [self._compact_observation(item) for item in old[-12:]]
        summary = {
            "kind": "compacted_history_summary",
            "instruction_trust": "untrusted_evidence_not_policy",
            "older_observation_count": len(old),
            "older_observation_sha256": hashlib.sha256(
                json.dumps(old, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
            ).hexdigest(),
            "events": compact_events,
            "note": "Re-read the referenced local file or rerun the tool if exact content is needed.",
        }
        observations[:] = [summary] + recent

        # If a set of recent results still exceeds budget, compact oldest recent results until
        # the history fits. The final two observations are kept in full whenever possible.
        while encoded_size() > self.max_history_chars and len(observations) > 2:
            index = 1 if observations[0].get("kind") == "compacted_history_summary" else 0
            if index >= len(observations) - 2:
                break
            observations[index] = self._compact_observation(observations[index])

        if encoded_size() > self.max_history_chars:
            # Last resort: summarize the observation set to digests and bounded event metadata.
            rows = observations[-2:]
            summary["events"] = [self._compact_observation(item) for item in rows]
            summary["note"] = "History compacted to fit token budget; retrieve exact evidence again."
            observations[:] = [summary]

    def run(self, goal: str) -> dict[str, Any]:
        observations: list[dict[str, Any]] = []
        trace: list[dict[str, Any]] = []
        last_candidate: str | None = None
        last_action_signature: str | None = None
        consecutive_repeat_count = 0
        consecutive_planner_failures = 0
        approval_denied_tools: set[str] = set()
        for step in range(self.max_steps):
            self._compact_observation_history(observations)
            try:
                action = self.planner(goal, observations, self.tools.describe())
                consecutive_planner_failures = 0
            except Exception as exc:
                consecutive_planner_failures += 1
                observations.append({
                    "kind": "planner_error",
                    "error_type": type(exc).__name__,
                    "message": str(exc)[:240],
                    "attempt": consecutive_planner_failures,
                    "instruction_trust": "untrusted_error_observation",
                })
                trace.append({
                    "step": step,
                    "action": "planner",
                    "result": "error",
                    "error_type": type(exc).__name__,
                })
                self._audit({
                    "event": "planner_error",
                    "step": step,
                    "error_type": type(exc).__name__,
                    "error_sha256": self._digest(str(exc)),
                    "attempt": consecutive_planner_failures,
                })
                if consecutive_planner_failures >= 2 or step + 1 == self.max_steps:
                    return {
                        "status": "needs_review",
                        "answer": last_candidate,
                        "observations": observations,
                        "trace": trace,
                        "verification": {
                            "passed": False,
                            "reason": "planner repeatedly failed; no additional tool action was taken",
                        },
                    }
                continue
            if not isinstance(action, dict):
                raise TypeError("planner must return a dict action")
            action_type = action.get("type")

            if action_type == "final":
                content = str(action.get("content", ""))
                last_candidate = content
                try:
                    verdict = (
                        self.verifier(content, observations)
                        if self.verifier is not None
                        else {"passed": False, "reason": "no verifier configured"}
                    )
                except Exception as exc:
                    verdict = {
                        "passed": False,
                        "reason": f"verifier exception: {type(exc).__name__}: {str(exc)[:160]}",
                    }
                if not isinstance(verdict, dict) or type(verdict.get("passed")) is not bool:
                    verdict = {"passed": False, "reason": "verifier returned an invalid result"}
                passed = bool(verdict.get("passed", False))
                trace.append({
                    "step": step,
                    "action": "final",
                    "content_sha256": self._digest(content),
                    "verdict": verdict,
                })
                self._audit({
                    "event": "final_verification",
                    "step": step,
                    "content_sha256": self._digest(content),
                    "passed": passed,
                })
                if passed:
                    return {
                        "status": "completed",
                        "answer": content,
                        "observations": observations,
                        "trace": trace,
                        "verification": verdict,
                    }
                observations.append({
                    "kind": "verification_failure",
                    "content_sha256": self._digest(content),
                    "reason": str(verdict.get("reason", "verification failed")),
                    "instruction_trust": "untrusted_observation",
                })
                if step + 1 == self.max_steps:
                    return {
                        "status": "needs_review",
                        "answer": content,
                        "observations": observations,
                        "trace": trace,
                        "verification": verdict,
                    }
                continue

            if action_type != "tool":
                observations.append({
                    "kind": "controller_error",
                    "message": "planner must choose type='tool' or type='final'",
                })
                trace.append({"step": step, "action": action, "result": "invalid action type"})
                continue

            name = action.get("name")
            args = action.get("args", {})
            if not isinstance(name, str) or not name.strip():
                observations.append({
                    "kind": "controller_error",
                    "message": "tool action name must be a non-empty string",
                    "instruction_trust": "untrusted_model_output",
                })
                trace.append({"step": step, "action": "tool", "result": "invalid_tool_name"})
                continue
            if not isinstance(args, dict):
                observations.append({
                    "kind": "controller_error",
                    "tool": name,
                    "message": "tool action args must be an object",
                    "instruction_trust": "untrusted_model_output",
                })
                trace.append({
                    "step": step,
                    "action": {"type": "tool", "name": name},
                    "result": "invalid_tool_arguments",
                })
                continue
            if name in approval_denied_tools:
                observations.append({
                    "kind": "approval_denied_retry_blocked",
                    "tool": name,
                    "reason": (
                        "human approval was denied earlier in this run; this capability is blocked "
                        "for the remainder of the run, including alternate paths or arguments"
                    ),
                    "instruction_trust": "untrusted_observation",
                })
                trace.append({
                    "step": step,
                    "action": {"type": "tool", "name": name},
                    "result": "blocked_after_approval_denial",
                })
                self._audit({
                    "event": "tool_blocked_after_approval_denial",
                    "step": step,
                    "tool": name,
                    "arguments_sha256": self._digest(args),
                })
                continue
            signature = self._digest({"name": name, "args": args})
            if signature == last_action_signature:
                consecutive_repeat_count += 1
            else:
                last_action_signature = signature
                consecutive_repeat_count = 1
            if consecutive_repeat_count >= self.repeat_action_limit:
                observation = {
                    "kind": "loop_detected",
                    "tool": name,
                    "consecutive_repeats": consecutive_repeat_count,
                    "reason": "same tool and arguments repeated without a strategy change",
                    "instruction_trust": "untrusted_observation",
                }
                observations.append(observation)
                trace.append({
                    "step": step,
                    "action": {"type": "tool", "name": name},
                    "result": "blocked_loop",
                    "consecutive_repeats": consecutive_repeat_count,
                })
                self._audit({
                    "event": "loop_blocked",
                    "step": step,
                    "tool": name,
                    "arguments_sha256": signature,
                    "consecutive_repeats": consecutive_repeat_count,
                })
                # Block further copies of the same action, but allow a genuinely different plan.
                continue
            spec = None
            try:
                spec = self.tools.get_spec(name)
                # Persist intent first. If the audit store fails, no side effect runs.
                self._audit({
                    "event": "tool_intent",
                    "step": step,
                    "tool": name,
                    "risk_level": spec.risk_level,
                    "requires_approval": spec.requires_approval,
                    "argument_keys": sorted(args) if isinstance(args, dict) else [],
                    "arguments_sha256": self._digest(args),
                })
                result = self.tools.execute(
                    name, args, approve=self.approval_callback
                )
                bounded = self._bound_result(result)
                observation = {
                    "kind": "untrusted_tool_observation",
                    "tool": name,
                    "result": bounded,
                    "instruction_trust": "untrusted_data_not_policy",
                }
                observations.append(observation)
                self._audit({
                    "event": "tool_result",
                    "step": step,
                    "tool": name,
                    "arguments_sha256": self._digest(args),
                    "result_sha256": self._digest(result),
                    "result_truncated": bounded is not result,
                    "status": "success",
                })
                trace.append({
                    "step": step,
                    "action": {
                        "type": "tool",
                        "name": name,
                        "argument_keys": sorted(args),
                        "arguments_sha256": self._digest(args),
                    },
                    "result": "success",
                })
            except Exception as exc:
                # A human denial blocks the entire state-changing capability for this run; the
                # model cannot evade it by changing the filename or arguments on a later step.
                approval_denied = isinstance(exc, PermissionError) and spec is not None and spec.requires_approval
                if approval_denied:
                    approval_denied_tools.add(name)
                observation_kind = "approval_denied" if approval_denied else "tool_error"
                message = str(exc)
                if approval_denied:
                    message = (
                        "Human approval was not granted. Do not retry this capability or attempt "
                        "an equivalent action under another path. Continue without the side effect. "
                        + message
                    )
                observations.append({
                    "kind": observation_kind,
                    "tool": name,
                    "error_type": type(exc).__name__,
                    "message": message,
                    "instruction_trust": "untrusted_data_not_policy",
                })
                try:
                    self._audit({
                        "event": "tool_approval_denied" if approval_denied else "tool_error",
                        "step": step,
                        "tool": name,
                        "arguments_sha256": self._digest(args),
                        "error_type": type(exc).__name__,
                        "capability_blocked_for_run": approval_denied,
                    })
                except Exception:
                    # If audit itself failed, fail closed rather than continue autonomous work.
                    raise
                trace.append({
                    "step": step,
                    "action": {"type": "tool", "name": name},
                    "result": "approval_denied" if approval_denied else "error",
                    "error_type": type(exc).__name__,
                })

        return {
            "status": "step_limit_reached",
            "answer": last_candidate,
            "observations": observations,
            "trace": trace,
            "verification": {"passed": False, "reason": "agent step limit reached"},
        }
