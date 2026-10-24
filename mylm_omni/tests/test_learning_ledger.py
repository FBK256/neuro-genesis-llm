import json
import re

from mylm_omni.learning_ledger import LocalLearningLedger


class FakeTriageBackend:
    def __init__(self, malformed=False):
        self.malformed = malformed
        self.prompt = ""

    def complete(self, prompt, **kwargs):
        self.prompt = prompt
        if self.malformed:
            return "not-json"
        evidence_id = re.search(r"[0-9a-f-]{36}:observation:[0-9]+", prompt).group(0)
        return json.dumps({
            "hypothesis": "The planner repeats a tool after a failed result because it lacks a recovery hint.",
            "change_kind": "prompt",
            "change_description": "Add one explicit recovery instruction after a tool error.",
            "expected_gain": "Lower repeated tool-call rate on the frozen recovery set.",
            "risk": "The instruction may add unnecessary verbosity.",
            "test_plan": [
                "Compare baseline and candidate on the same frozen tool-error cases.",
                "Require a lower repeated-action rate with equal task success.",
                "Run prompt-injection and permission regression tests."
            ],
            "evidence_ids": [evidence_id]
        })


def test_outcome_ledger_stores_hashes_and_creates_unapplied_proposal(tmp_path):
    ledger_path = tmp_path / ".mylm-learning.jsonl"
    ledger = LocalLearningLedger(ledger_path)
    backend = FakeTriageBackend()
    result = ledger.record_run(
        goal="Private user task: do not persist this whole request",
        result={
            "status": "needs_review",
            "answer": "partial answer",
            "observations": [{
                "kind": "tool_error",
                "tool": "read_workspace_text",
                "error_type": "UnicodeDecodeError",
                "message": "input was not UTF-8",
            }],
            "trace": [{"step": 0, "action": "tool"}],
            "verification": {"passed": False, "reason": "tool operation did not succeed"},
        },
        backend=backend,
    )
    rows = [json.loads(line) for line in ledger_path.read_text(encoding="utf-8").splitlines()]
    assert result["proposal"]["event"] == "improvement_proposal"
    assert result["proposal"]["proposal"]["state"] == "awaiting_isolated_evaluation"
    assert result["proposal"]["proposal"]["requires_human_review"] is True
    assert rows[0]["event"] == "task_outcome"
    assert rows[1]["event"] == "improvement_proposal"
    serialized = ledger_path.read_text(encoding="utf-8")
    assert "Private user task" not in serialized
    assert "partial answer" not in serialized
    assert result["outcome"]["goal_sha256"]


def test_successful_run_is_recorded_without_running_failure_triage(tmp_path):
    ledger = LocalLearningLedger(tmp_path / "ledger.jsonl")
    backend = FakeTriageBackend()
    result = ledger.record_run(
        goal="Summarize public text",
        result={
            "status": "completed",
            "answer": "summary",
            "observations": [],
            "trace": [{"step": 0}],
            "verification": {"passed": True},
        },
        backend=backend,
    )
    assert result["proposal"] is None
    assert backend.prompt == ""
    row = json.loads((tmp_path / "ledger.jsonl").read_text(encoding="utf-8"))
    assert row["status"] == "completed"
    assert row["answer_sha256"]


def test_malformed_triage_does_not_hide_primary_task_failure(tmp_path):
    ledger = LocalLearningLedger(tmp_path / "ledger.jsonl")
    result = ledger.record_run(
        goal="Investigate a failure",
        result={
            "status": "needs_review",
            "observations": [{"kind": "verification_failure", "reason": "uncertain result"}],
            "trace": [],
            "verification": {"passed": False},
        },
        backend=FakeTriageBackend(malformed=True),
    )
    rows = [json.loads(line) for line in (tmp_path / "ledger.jsonl").read_text(encoding="utf-8").splitlines()]
    assert result["outcome"]["status"] == "needs_review"
    assert result["proposal"] is None
    assert rows[-1]["event"] == "improvement_triage_error"
