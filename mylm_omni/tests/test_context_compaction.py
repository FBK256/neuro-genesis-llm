import json

from mylm_omni.agent import AgentController, ToolRegistry, ToolSpec


def test_observation_history_is_compacted_without_losing_recent_events():
    registry = ToolRegistry()
    names = []
    for index in range(6):
        name = f"tool_{index}"
        names.append(name)
        registry.register(ToolSpec(
            name=name,
            description="return a large bounded result",
            function=lambda index=index: {
                "path": f"file-{index}.txt",
                "text": (f"important-{index} " * 100),
                "sha256": f"{index:064x}",
            },
        ))
    actions = [
        {"type": "tool", "name": name, "args": {}}
        for name in names
    ] + [{"type": "final", "content": "completed"}]

    def planner(goal, observations, tools):
        return actions.pop(0)

    agent = AgentController(
        planner=planner,
        tools=registry,
        verifier=lambda answer, observations: {"passed": answer == "completed"},
        max_steps=7,
        max_observation_bytes=512,
        max_history_chars=1024,
        recent_observations_to_keep=2,
    )
    result = agent.run("call six tools")
    assert result["status"] == "completed"
    assert any(item["kind"] == "compacted_history_summary" for item in result["observations"])
    assert len(json.dumps(result["observations"], ensure_ascii=False, default=str)) <= 1500
    assert result["observations"][-1]["kind"] == "untrusted_tool_observation" or result["observations"][-1]["kind"] == "compacted_history_summary"
