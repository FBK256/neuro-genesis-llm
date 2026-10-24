from mylm_omni.budget import choose_task_budget


def test_simple_question_gets_reflex_budget():
    budget = choose_task_budget(
        "What is the capital of France?",
        has_workspace_tools=True,
        max_allowed_steps=12,
        max_allowed_tokens=768,
    )
    assert budget.mode == "reflex"
    assert budget.max_steps == 1
    assert budget.max_new_tokens == 160


def test_multistep_file_work_gets_bounded_deep_budget():
    budget = choose_task_budget(
        "Inspect the workspace files, analyze the results, create a summary, and verify it.",
        has_workspace_tools=True,
        mode="auto",
        max_allowed_steps=8,
        max_allowed_tokens=400,
    )
    assert budget.mode == "deep"
    assert budget.max_steps == 8
    assert budget.max_new_tokens == 400


def test_image_input_requires_at_least_deliberative_mode():
    budget = choose_task_budget("Describe this image.", has_image=True)
    assert budget.mode == "deliberative"
    assert budget.max_steps > 1


def test_explicit_reflex_cannot_exceed_hard_limits():
    budget = choose_task_budget(
        "Analyze this complex project.",
        mode="reflex",
        max_allowed_steps=5,
        max_allowed_tokens=64,
    )
    assert budget.mode == "reflex"
    assert budget.max_steps == 1
    assert budget.max_new_tokens == 64
