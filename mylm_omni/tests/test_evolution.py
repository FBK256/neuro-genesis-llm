import pytest

from mylm_omni.evolution import EvalResult, ImprovementGate, ImprovementProposal


def result(run_id, quality, safety=True, errors=0.0):
    return EvalResult(
        eval_set_version="frozen-v1",
        metrics={"quality": quality, "tool_error": errors},
        safety_passed=safety,
        safety_findings=[] if safety else ["policy regression"],
        run_id=run_id,
    )


def proposal():
    return ImprovementProposal(
        hypothesis="prompt reduces malformed tool arguments",
        change_kind="prompt",
        change_description="make required tool arguments explicit",
        expected_gain="higher tool success",
        risks=["more tokens"],
        evidence_ids=["failure-17"],
        primary_metric="quality",
    )


def test_candidate_is_eligible_only_if_quality_improves_without_regression(tmp_path):
    gate = ImprovementGate(tmp_path / "experiments.jsonl", regression_metrics={"tool_error": 0.01})
    record = gate.evaluate(
        proposal(), baseline=result("baseline", 0.70, errors=0.02),
        candidate=result("candidate", 0.75, errors=0.01),
    )
    assert record["decision"] == "canary_eligible"
    promoted = []
    outcome = gate.promote(record, canary_passed=True, promote_callback=lambda: promoted.append(True))
    assert outcome["decision"] == "promoted"
    assert promoted == [True]


def test_guardrail_regression_blocks_candidate(tmp_path):
    gate = ImprovementGate(tmp_path / "experiments.jsonl", regression_metrics={"tool_error": 0.01})
    record = gate.evaluate(
        proposal(), baseline=result("baseline", 0.70, errors=0.01),
        candidate=result("candidate", 0.90, errors=0.20),
    )
    assert record["decision"] == "rejected"
    with pytest.raises(ValueError):
        gate.promote(record, canary_passed=True, promote_callback=lambda: None)


def test_safety_failure_blocks_candidate_even_when_primary_improves(tmp_path):
    gate = ImprovementGate(tmp_path / "experiments.jsonl")
    record = gate.evaluate(
        proposal(), baseline=result("baseline", 0.70),
        candidate=result("candidate", 0.90, safety=False),
    )
    assert record["decision"] == "rejected"



def test_failed_canary_executes_rollback_callback(tmp_path):
    gate = ImprovementGate(tmp_path / "experiments.jsonl")
    record = gate.evaluate(
        proposal(),
        baseline=result("baseline", 0.70),
        candidate=result("candidate", 0.80),
    )
    calls = {"promote": 0, "rollback": 0}
    outcome = gate.promote(
        record,
        canary_passed=False,
        promote_callback=lambda: calls.__setitem__("promote", calls["promote"] + 1),
        rollback_callback=lambda: calls.__setitem__("rollback", calls["rollback"] + 1),
    )
    assert outcome["decision"] == "rolled_back"
    assert calls == {"promote": 0, "rollback": 1}


def test_eval_gate_rejects_non_finite_metrics_and_metric_set_mismatch(tmp_path):
    import math
    from mylm_omni.evolution import EvalResult

    with pytest.raises(ValueError, match="finite"):
        EvalResult(
            eval_set_version="frozen-v1",
            metrics={"quality": math.nan},
            safety_passed=True,
            safety_findings=[],
            run_id="bad-run",
        )

    gate = ImprovementGate(tmp_path / "gate-ledger.jsonl")
    with pytest.raises(ValueError, match="same metric keys"):
        gate.evaluate(
            proposal(),
            baseline=result("baseline", 0.7),
            candidate=EvalResult(
                eval_set_version="frozen-v1",
                metrics={"quality": 0.8, "new_metric": 1.0},
                safety_passed=True,
                safety_findings=[],
                run_id="candidate",
            ),
        )


def test_self_improvement_engine_requires_failure_evidence_and_rolls_back_rejected_candidate(tmp_path):
    from mylm_omni.evolution import FailureEvent, SelfImprovementEngine

    calls = {"built": 0, "rolled_back": 0, "promoted": 0}

    def builder(proposal):
        calls["built"] += 1
        return {"candidate_id": proposal.candidate_id}

    engine = SelfImprovementEngine(
        gate=ImprovementGate(tmp_path / "gate.jsonl", regression_metrics={"tool_error": 0.01}),
        ledger_path=tmp_path / "ledger.jsonl",
        proposal_generator=lambda failure: ImprovementProposal(
            hypothesis="change lowers tool errors",
            change_kind="prompt",
            change_description="clarify required fields",
            expected_gain="fewer malformed calls",
            risks=["slightly longer prompt"],
            evidence_ids=["case-17"],
            primary_metric="quality",
        ),
        baseline_evaluator=lambda: result("baseline", 0.70, errors=0.01),
        candidate_builder=builder,
        candidate_evaluator=lambda candidate: result("candidate", 0.72, errors=0.20),
        canary_evaluator=lambda candidate: True,
        promote_candidate=lambda candidate: calls.__setitem__("promoted", calls["promoted"] + 1),
        rollback_candidate=lambda candidate: calls.__setitem__("rolled_back", calls["rolled_back"] + 1),
    )
    failure = FailureEvent(
        task_id="task-17",
        failure_type="invalid_tool_args",
        expected_summary="tool argument schema valid",
        observed_summary="required argument missing",
        evidence_ids=["case-17"],
    )
    outcome = engine.observe(failure)
    assert outcome["decision"] == "rejected"
    assert calls == {"built": 1, "rolled_back": 1, "promoted": 0}
