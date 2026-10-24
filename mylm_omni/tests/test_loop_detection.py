from mylm_omni.agent import AgentController, ToolRegistry, ToolSpec


def test_repeat_action_guard_blocks_identical_tool_calls():
    registry = ToolRegistry()
    calls = {"read": 0, "other": 0}
    registry.register(ToolSpec(
        name="read", description="read fixed data",
        function=lambda: calls.__setitem__("read", calls["read"] + 1) or {"data": "x"},
    ))
    registry.register(ToolSpec(
        name="other", description="return another result",
        function=lambda: calls.__setitem__("other", calls["other"] + 1) or {"data": "y"},
    ))
    actions = [
        {"type": "tool", "name": "read", "args": {}},
        {"type": "tool", "name": "read", "args": {}},
        {"type": "tool", "name": "read", "args": {}},
        {"type": "tool", "name": "read", "args": {}},
        {"type": "tool", "name": "other", "args": {}},
        {"type": "final", "content": "done"},
    ]

    def planner(goal, observations, tools):
        return actions.pop(0)

    agent = AgentController(
        planner=planner,
        tools=registry,
        verifier=lambda answer, obs: {"passed": answer == "done"},
        max_steps=6,
        repeat_action_limit=3,
    )
    result = agent.run("test repeat guard")
    assert result["status"] == "completed"
    assert calls == {"read": 2, "other": 1}
    assert any(row["kind"] == "loop_detected" for row in result["observations"])
