import pytest

from mylm_omni.agent import AgentController, ToolRegistry, ToolSpec


def test_tool_registry_rejects_unregistered_capabilities():
    registry = ToolRegistry()
    with pytest.raises(PermissionError, match="not registered"):
        registry.execute("shell", {"command": "echo unsafe"})


def test_sensitive_tool_requires_explicit_approval():
    registry = ToolRegistry()
    registry.register(ToolSpec(
        name="write_file", description="write a user file",
        function=lambda path, content: {"path": path, "bytes": len(content)},
        requires_approval=True, allowed_args=frozenset({"path", "content"}),
    ))
    with pytest.raises(PermissionError, match="approval"):
        registry.execute("write_file", {"path": "x.txt", "content": "hello"})
    result = registry.execute(
        "write_file", {"path": "x.txt", "content": "hello"},
        approve=lambda spec, args: spec.name == "write_file" and args["path"] == "x.txt",
    )
    assert result["bytes"] == 5


def test_agent_runs_plan_act_observe_verify_loop():
    registry = ToolRegistry()
    registry.register(ToolSpec("add", "sum two numbers", lambda a, b: a + b,
                               allowed_args=frozenset({"a", "b"})))

    def planner(goal, observations, tools):
        if not observations:
            return {"type": "tool", "name": "add", "args": {"a": 2, "b": 3}}
        return {"type": "final", "content": f"Result: {observations[0]['result']}"}

    agent = AgentController(
        planner=planner, tools=registry,
        verifier=lambda answer, observations: {
            "passed": answer == "Result: 5" and observations[0]["result"] == 5
        },
    )
    result = agent.run("Calculate 2 + 3")
    assert result["status"] == "completed"
    assert result["answer"] == "Result: 5"
    assert result["verification"]["passed"]



def test_high_risk_tool_automatically_requires_approval():
    spec = ToolSpec(
        name="publish", description="publish publicly",
        function=lambda content: {"published": True},
        risk_level="critical",
    )
    assert spec.requires_approval


def test_failed_verification_causes_replanning_not_false_completion():
    registry = ToolRegistry()
    attempts = {"count": 0}

    def planner(goal, observations, tools):
        attempts["count"] += 1
        if attempts["count"] == 1:
            return {"type": "final", "content": "unsupported claim"}
        return {"type": "final", "content": "verified correction"}

    def verifier(answer, observations):
        return {"passed": answer == "verified correction", "reason": "fixed evidence gap"}

    agent = AgentController(planner=planner, tools=registry, verifier=verifier, max_steps=3)
    result = agent.run("Answer only when verified")
    assert result["status"] == "completed"
    assert result["answer"] == "verified correction"
    assert attempts["count"] == 2
    assert any(row["kind"] == "verification_failure" for row in result["observations"])


def test_oversized_tool_results_are_bounded_and_audit_is_metadata_only(tmp_path):
    registry = ToolRegistry()
    registry.register(ToolSpec(
        name="lookup", description="lookup test data",
        function=lambda: {"text": "needle" * 2000},
    ))
    audit = tmp_path / "agent-audit.jsonl"
    calls = {"count": 0}

    def planner(goal, observations, tools):
        calls["count"] += 1
        if calls["count"] == 1:
            return {"type": "tool", "name": "lookup", "args": {}}
        return {"type": "final", "content": "done"}

    agent = AgentController(
        planner=planner,
        tools=registry,
        verifier=lambda answer, observations: {"passed": True},
        audit_path=audit,
        max_observation_bytes=256,
    )
    result = agent.run("lookup")
    observation = result["observations"][0]
    assert observation["result"]["truncated"] is True
    assert observation["instruction_trust"] == "untrusted_data_not_policy"
    audit_text = audit.read_text(encoding="utf-8")
    assert "needle" not in audit_text
    assert '"event": "tool_intent"' in audit_text


def test_no_verifier_never_marks_final_answer_verified():
    agent = AgentController(
        planner=lambda goal, observations, tools: {"type": "final", "content": "unverified"},
        tools=ToolRegistry(),
        max_steps=1,
    )
    result = agent.run("test")
    assert result["status"] == "needs_review"
    assert result["verification"]["passed"] is False



def test_repeated_planner_exceptions_stop_before_any_tool_side_effect():
    attempts = {"planner": 0, "side_effect": 0}

    def planner(goal, observations, tools):
        attempts["planner"] += 1
        raise ValueError("input context exceeds local model budget")

    registry = ToolRegistry()
    registry.register(ToolSpec(
        name="danger",
        description="side effect",
        function=lambda: attempts.__setitem__("side_effect", attempts["side_effect"] + 1),
    ))
    result = AgentController(
        planner=planner,
        tools=registry,
        max_steps=6,
        verifier=lambda answer, obs: {"passed": True},
    ).run("try a task")
    assert result["status"] == "needs_review"
    assert attempts == {"planner": 2, "side_effect": 0}
    assert result["verification"]["passed"] is False


def test_verifier_exception_never_counts_as_success():
    calls = {"count": 0}

    def planner(goal, observations, tools):
        calls["count"] += 1
        return {"type": "final", "content": "claim"}

    def broken_verifier(answer, observations):
        raise RuntimeError("verifier unavailable")

    result = AgentController(
        planner=planner, tools=ToolRegistry(), verifier=broken_verifier, max_steps=2
    ).run("verify this")
    assert result["status"] == "needs_review"
    assert result["verification"]["passed"] is False
    assert any("verifier exception" in row["reason"] for row in result["observations"]
               if row["kind"] == "verification_failure")



def test_denied_write_cannot_be_retried_under_an_alternate_path():
    registry = ToolRegistry()
    state = {"approve_calls": 0, "write_calls": 0, "read_calls": 0}
    registry.register(ToolSpec(
        name="create_workspace_file",
        description="create a new file",
        function=lambda path, content: state.__setitem__("write_calls", state["write_calls"] + 1),
        requires_approval=True,
        risk_level="high",
        allowed_args=frozenset({"path", "content"}),
    ))
    registry.register(ToolSpec(
        name="read_workspace_text",
        description="read workspace",
        function=lambda path: state.__setitem__("read_calls", state["read_calls"] + 1) or {"text": "ok"},
        allowed_args=frozenset({"path"}),
    ))
    actions = [
        {"type": "tool", "name": "create_workspace_file",
         "args": {"path": "approved.md", "content": "first attempt"}},
        # Try to evade denial by changing the path and content.
        {"type": "tool", "name": "create_workspace_file",
         "args": {"path": "alternate.md", "content": "second attempt"}},
        {"type": "tool", "name": "read_workspace_text", "args": {"path": "notes.md"}},
        {"type": "final", "content": "I continued without writing."},
    ]

    def planner(goal, observations, tools):
        return actions.pop(0)

    def deny_approval(spec, args):
        state["approve_calls"] += 1
        return False

    result = AgentController(
        planner=planner,
        tools=registry,
        verifier=lambda answer, observations: {"passed": answer.startswith("I continued")},
        approval_callback=deny_approval,
        max_steps=5,
    ).run("create a report")
    assert result["status"] == "completed"
    assert state == {"approve_calls": 1, "write_calls": 0, "read_calls": 1}
    assert any(row["kind"] == "approval_denied" for row in result["observations"])
    assert any(row["kind"] == "approval_denied_retry_blocked" for row in result["observations"])



def test_malformed_tool_name_or_arguments_are_rejected_without_crashing():
    registry = ToolRegistry()
    calls = {"count": 0}
    registry.register(ToolSpec(
        name="safe", description="safe no-op",
        function=lambda: calls.__setitem__("count", calls["count"] + 1),
    ))
    actions = [
        {"type": "tool", "name": {"not": "hashable"}, "args": {}},
        {"type": "tool", "name": "safe", "args": ["not", "an", "object"]},
        {"type": "final", "content": "Stopped safely."},
    ]

    def planner(goal, observations, tools):
        return actions.pop(0)

    result = AgentController(
        planner=planner,
        tools=registry,
        verifier=lambda answer, observations: {"passed": answer == "Stopped safely."},
        max_steps=3,
    ).run("exercise schema guard")
    assert result["status"] == "completed"
    assert calls["count"] == 0
    assert sum(row["kind"] == "controller_error" for row in result["observations"]) == 2
