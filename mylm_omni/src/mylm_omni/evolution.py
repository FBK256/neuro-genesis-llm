"""Evidence-based self-improvement gates; candidate code is never executed here."""
from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class ImprovementProposal:
    """A proposed change in a constrained, reviewable form."""

    hypothesis: str
    change_kind: str  # prompt, retrieval, memory_rule, tool_policy, model_adapter, model_weights
    change_description: str
    expected_gain: str
    risks: list[str]
    evidence_ids: list[str]
    primary_metric: str
    candidate_id: str = ""
    created_at: str = ""

    def __post_init__(self) -> None:
        if not self.candidate_id:
            self.candidate_id = str(uuid.uuid4())
        if not self.created_at:
            self.created_at = _now()
        if self.change_kind not in {
            "prompt", "retrieval", "memory_rule", "tool_policy", "model_adapter", "model_weights"
        }:
            raise ValueError("unsupported change_kind")
        if not self.hypothesis.strip() or not self.change_description.strip():
            raise ValueError("hypothesis and change description are required")


@dataclass
class EvalResult:
    """Metric snapshot produced by a fixed, versioned evaluator."""

    eval_set_version: str
    metrics: dict[str, float]
    safety_passed: bool
    safety_findings: list[str]
    run_id: str
    tokens_seen: int = 0
    wall_seconds: float = 0.0

    def __post_init__(self) -> None:
        if not self.eval_set_version.strip():
            raise ValueError("eval_set_version must not be empty")
        if not self.run_id.strip():
            raise ValueError("run_id must not be empty")
        if self.tokens_seen < 0 or self.wall_seconds < 0:
            raise ValueError("tokens_seen and wall_seconds must be non-negative")
        if not self.metrics:
            raise ValueError("EvalResult.metrics must contain at least one metric")
        for name, value in self.metrics.items():
            if not name.strip() or not isinstance(value, (int, float)):
                raise ValueError("metric names must be non-empty and values numeric")
            if not __import__("math").isfinite(float(value)):
                raise ValueError(f"metric {name} must be finite")


class ImprovementGate:
    """Run a baseline and isolated candidate experiment; promote only without regressions.

    The caller is responsible for sandboxing and applying the candidate. Evaluation must use
    a frozen set not editable by the agent. This class records decisions but does not run shell
    commands, execute generated code, or update model weights on its own.
    """

    def __init__(
        self,
        log_path: str | Path,
        *,
        max_regression: float = 0.01,
        min_improvement: float = 0.0,
        regression_metrics: dict[str, float] | None = None,
        metric_directions: dict[str, str] | None = None,
    ) -> None:
        if max_regression < 0 or min_improvement < 0:
            raise ValueError("regression and improvement limits must be non-negative")
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.max_regression = max_regression
        self.min_improvement = min_improvement
        self.regression_metrics = regression_metrics or {}
        self.metric_directions = metric_directions or {}
        if any(value < 0 for value in self.regression_metrics.values()):
            raise ValueError("metric-specific regression tolerances must be non-negative")
        if any(value not in {"higher", "lower", "neutral"} for value in self.metric_directions.values()):
            raise ValueError("metric directions must be higher, lower, or neutral")

    def evaluate(
        self,
        proposal: ImprovementProposal,
        *,
        baseline: EvalResult,
        candidate: EvalResult,
    ) -> dict:
        if baseline.eval_set_version != candidate.eval_set_version:
            raise ValueError("baseline and candidate must use the same frozen eval set version")
        if baseline.run_id == candidate.run_id:
            raise ValueError("baseline and candidate must be distinct evaluation runs")
        if baseline.metrics.keys() != candidate.metrics.keys():
            raise ValueError("baseline and candidate must report the same metric keys")
        metric = proposal.primary_metric
        if metric not in baseline.metrics:
            raise ValueError(f"primary metric missing: {metric}")
        regressions: dict[str, float] = {}
        lower_suffixes = ("_loss", "_latency", "_seconds", "_error", "_perplexity")
        for name in baseline.metrics:
            if name == metric:
                continue
            direction = self.metric_directions.get(
                name, "lower" if name.endswith(lower_suffixes) else "higher"
            )
            if direction == "neutral":
                continue
            tolerance = self.regression_metrics.get(name, self.max_regression)
            harm = (
                candidate.metrics[name] - baseline.metrics[name]
                if direction == "lower"
                else baseline.metrics[name] - candidate.metrics[name]
            )
            regressions[name] = max(0.0, harm - tolerance)

        primary_delta = candidate.metrics[metric] - baseline.metrics[metric]
        primary_direction = self.metric_directions.get(
            metric, "lower" if metric.endswith(("_loss", "_latency", "_seconds", "_error", "_perplexity")) else "higher"
        )
        if primary_direction == "neutral":
            raise ValueError("primary metric cannot have neutral direction")
        lower_primary = primary_direction == "lower"
        if lower_primary:
            primary_delta = baseline.metrics[metric] - candidate.metrics[metric]
        safety_passed = baseline.safety_passed and candidate.safety_passed
        accepted = (
            safety_passed
            and not any(value > 0 for value in regressions.values())
            and primary_delta > self.min_improvement
        )
        record = {
            "at": _now(),
            "proposal": asdict(proposal),
            "baseline": asdict(baseline),
            "candidate": asdict(candidate),
            "primary_metric": metric,
            "primary_improvement": primary_delta,
            "regression_excess": regressions,
            "decision": "canary_eligible" if accepted else "rejected",
            "reason": (
                "metric improved and guardrails passed"
                if accepted
                else "no sufficient gain, safety failure, or guardrail regression"
            ),
        }
        with self.log_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        return record

    @staticmethod
    def promote(
        record: dict,
        *,
        canary_passed: bool,
        promote_callback: Callable[[], None],
        rollback_callback: Callable[[], None] | None = None,
    ) -> dict:
        """Promote only after a canary. A failed canary requires an actual rollback callback."""
        if record.get("decision") != "canary_eligible":
            raise ValueError("proposal is not canary eligible")
        if not canary_passed:
            if rollback_callback is None:
                raise ValueError("canary failed; rollback_callback is required")
            rollback_callback()
            return {"decision": "rolled_back", "reason": "canary gate failed; rollback executed"}
        promote_callback()
        return {"decision": "promoted", "reason": "canary gate passed"}



@dataclass
class FailureEvent:
    """Compact failure record for the self-improvement loop.

    Raw prompts, documents, and tool outputs should remain in a separately access-controlled
    evidence store. This object carries summaries and stable evidence IDs, not a full transcript.
    """

    task_id: str
    failure_type: str
    expected_summary: str
    observed_summary: str
    evidence_ids: list[str]
    severity: str = "medium"
    failure_id: str = ""
    created_at: str = ""

    def __post_init__(self) -> None:
        if not self.failure_id:
            self.failure_id = str(uuid.uuid4())
        if not self.created_at:
            self.created_at = _now()
        if not self.task_id.strip() or not self.failure_type.strip():
            raise ValueError("task_id and failure_type must not be empty")
        if self.severity not in {"low", "medium", "high", "critical"}:
            raise ValueError("severity must be low, medium, high, or critical")
        if not self.evidence_ids:
            raise ValueError("failure events must include at least one evidence ID")
        if len(self.expected_summary) > 2000 or len(self.observed_summary) > 2000:
            raise ValueError("failure summaries must be <= 2000 characters")


class SelfImprovementEngine:
    """Orchestrate evidence → hypothesis → isolated candidate → fixed eval → canary.

    The callbacks are supplied by a trusted host application. In particular, candidate_builder
    must create an isolated candidate rather than mutate production state; candidate_evaluator
    must use a frozen test suite; promote_candidate should perform an atomic/versioned switch;
    rollback_candidate must restore the prior known-good state. This class never executes
    generated shell/Python code itself.

    If the candidate passes fixed evaluation but fails canary, rollback_callback is invoked
    before any procedure is written to memory. On a successful promotion, remember_procedure
    can save a source-backed procedural memory entry.
    """

    def __init__(
        self,
        *,
        gate: ImprovementGate,
        ledger_path: str | Path,
        proposal_generator: Callable[[FailureEvent], ImprovementProposal | None],
        baseline_evaluator: Callable[[], EvalResult],
        candidate_builder: Callable[[ImprovementProposal], Any],
        candidate_evaluator: Callable[[Any], EvalResult],
        canary_evaluator: Callable[[Any], bool],
        promote_candidate: Callable[[Any], None],
        rollback_candidate: Callable[[Any], None],
        remember_procedure: Callable[[FailureEvent, ImprovementProposal, EvalResult], None] | None = None,
    ) -> None:
        self.gate = gate
        self.ledger_path = Path(ledger_path)
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        self.proposal_generator = proposal_generator
        self.baseline_evaluator = baseline_evaluator
        self.candidate_builder = candidate_builder
        self.candidate_evaluator = candidate_evaluator
        self.canary_evaluator = canary_evaluator
        self.promote_candidate = promote_candidate
        self.rollback_candidate = rollback_candidate
        self.remember_procedure = remember_procedure

    def _write_event(self, event: dict[str, Any]) -> None:
        """Append a small audit event. Never duplicate raw prompt/tool content in the ledger."""
        payload = {"at": _now(), **event}
        with self.ledger_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str) + "\n")
            stream.flush()
            import os
            os.fsync(stream.fileno())

    def observe(self, failure: FailureEvent) -> dict[str, Any]:
        """Run one bounded improvement cycle and return a decision/audit summary."""
        self._write_event({
            "event": "failure_observed",
            "failure_id": failure.failure_id,
            "task_id": failure.task_id,
            "failure_type": failure.failure_type,
            "severity": failure.severity,
            "evidence_ids": failure.evidence_ids,
        })
        proposal: ImprovementProposal | None = None
        candidate_handle: Any = None
        promoted = False
        try:
            proposal = self.proposal_generator(failure)
            if proposal is None:
                self._write_event({
                    "event": "proposal_not_created",
                    "failure_id": failure.failure_id,
                    "reason": "proposal generator found no testable hypothesis",
                })
                return {"decision": "no_proposal", "failure_id": failure.failure_id}
            if not set(proposal.evidence_ids).intersection(failure.evidence_ids):
                raise ValueError("proposal must cite at least one evidence ID from the observed failure")

            self._write_event({
                "event": "proposal_created",
                "failure_id": failure.failure_id,
                "candidate_id": proposal.candidate_id,
                "change_kind": proposal.change_kind,
                "evidence_ids": proposal.evidence_ids,
                "hypothesis_sha256": __import__("hashlib").sha256(
                    proposal.hypothesis.encode("utf-8")
                ).hexdigest(),
            })
            candidate_handle = self.candidate_builder(proposal)
            if candidate_handle is None:
                raise RuntimeError("candidate_builder returned None")
            baseline = self.baseline_evaluator()
            candidate_result = self.candidate_evaluator(candidate_handle)
            decision = self.gate.evaluate(
                proposal, baseline=baseline, candidate=candidate_result
            )
            self._write_event({
                "event": "candidate_evaluated",
                "failure_id": failure.failure_id,
                "candidate_id": proposal.candidate_id,
                "candidate_run_id": candidate_result.run_id,
                "decision": decision["decision"],
                "primary_metric": decision["primary_metric"],
                "primary_improvement": decision["primary_improvement"],
            })
            if decision["decision"] != "canary_eligible":
                self.rollback_candidate(candidate_handle)
                self._write_event({
                    "event": "candidate_discarded",
                    "failure_id": failure.failure_id,
                    "candidate_id": proposal.candidate_id,
                    "reason": decision["reason"],
                })
                return {"decision": "rejected", "failure_id": failure.failure_id,
                        "candidate_id": proposal.candidate_id, "evaluation": decision}

            canary_passed = False
            try:
                canary_passed = bool(self.canary_evaluator(candidate_handle))
            except Exception as exc:
                self._write_event({
                    "event": "canary_exception",
                    "failure_id": failure.failure_id,
                    "candidate_id": proposal.candidate_id,
                    "error_type": type(exc).__name__,
                })
                canary_passed = False

            outcome = self.gate.promote(
                decision,
                canary_passed=canary_passed,
                promote_callback=lambda: self.promote_candidate(candidate_handle),
                rollback_callback=lambda: self.rollback_candidate(candidate_handle),
            )
            promoted = outcome["decision"] == "promoted"
            self._write_event({
                "event": "promotion_decision",
                "failure_id": failure.failure_id,
                "candidate_id": proposal.candidate_id,
                "decision": outcome["decision"],
                "reason": outcome["reason"],
            })
            if promoted and self.remember_procedure is not None:
                self.remember_procedure(failure, proposal, candidate_result)
                self._write_event({
                    "event": "validated_procedure_recorded",
                    "failure_id": failure.failure_id,
                    "candidate_id": proposal.candidate_id,
                    "source_run_id": candidate_result.run_id,
                    "evidence_ids": proposal.evidence_ids,
                })
            return {
                "decision": outcome["decision"],
                "failure_id": failure.failure_id,
                "candidate_id": proposal.candidate_id,
                "evaluation": decision,
                "promotion": outcome,
            }
        except Exception as exc:
            if candidate_handle is not None and not promoted:
                try:
                    self.rollback_candidate(candidate_handle)
                except Exception as rollback_exc:
                    self._write_event({
                        "event": "rollback_exception",
                        "failure_id": failure.failure_id,
                        "error_type": type(rollback_exc).__name__,
                    })
            self._write_event({
                "event": "improvement_cycle_exception",
                "failure_id": failure.failure_id,
                "candidate_id": proposal.candidate_id if proposal else None,
                "error_type": type(exc).__name__,
                "error_message_sha256": __import__("hashlib").sha256(
                    str(exc).encode("utf-8")
                ).hexdigest(),
            })
            return {
                "decision": "error",
                "failure_id": failure.failure_id,
                "candidate_id": proposal.candidate_id if proposal else None,
                "error_type": type(exc).__name__,
            }
