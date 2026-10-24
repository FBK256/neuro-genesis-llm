import json
from pathlib import Path

import pytest

from scripts.build_agent_sft_dataset import build_dataset


def valid_episode():
    return {
        "episode_id": "episode-001",
        "split": "train",
        "goal": "Find and summarize the project notes.",
        "sanitized": True,
        "license": "internal-approved",
        "source": "curated-run-17",
        "tool_specs": [
            {"name": "list_files", "description": "List files.", "allowed_args": ["path"]},
            {"name": "read_file", "description": "Read file.", "allowed_args": ["path"]},
        ],
        "verified_success": True,
        "safety_passed": True,
        "steps": [
            {
                "action": {"type": "tool", "name": "list_files", "args": {"path": "."}},
                "observation": {
                    "kind": "tool_result",
                    "summary": "Found report.md.",
                    "count": 1,
                },
                "observation_verified": True,
            },
            {
                "action": {"type": "final", "content": "The project has one report file."},
            },
        ],
        "verified_repairs": [
            {
                "completed_steps_before_failure": 0,
                "prior_observation": {
                    "kind": "tool_error",
                    "reason": "Requested path did not exist.",
                    "tool": "read_file",
                },
                "corrective_action": {
                    "type": "tool",
                    "name": "list_files",
                    "args": {"path": "."},
                },
                "repair_verified": True,
                "safety_passed": True,
                "summary": "Listing the parent directory identified a valid path.",
            }
        ],
    }


def write_jsonl(path: Path, records):
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records),
        encoding="utf-8",
    )


def test_builds_success_and_verified_repair_examples_without_future_leakage(tmp_path):
    source = tmp_path / "trajectories.jsonl"
    output = tmp_path / "train.jsonl"
    write_jsonl(source, [valid_episode()])

    manifest = build_dataset(source, output, max_context_chars=8000, allow_repairs=True)
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]

    assert len(rows) == 3
    assert manifest["eligible_episodes"] == 1
    assert manifest["counts"]["verified_success_examples"] == 2
    assert manifest["counts"]["verified_repair_examples"] == 1
    repair = next(row for row in rows if row["metadata"]["example_type"] == "verified_repair")
    prompt_input = json.loads(repair["input"])
    assert [x["kind"] for x in prompt_input["observations"]] == ["tool_error"]
    assert "Found report.md" not in repair["input"]
    assert json.loads(repair["output"])["name"] == "list_files"
    assert output.with_suffix(".jsonl.manifest.json").is_file()


def test_skips_test_split_and_unsanitized_records(tmp_path):
    source = tmp_path / "trajectories.jsonl"
    output = tmp_path / "train.jsonl"
    valid = valid_episode()
    test_row = dict(valid, episode_id="test-only", split="test")
    unsanitized = dict(valid, episode_id="not-sanitized", sanitized=False)
    write_jsonl(source, [valid, test_row, unsanitized])

    manifest = build_dataset(source, output, max_context_chars=8000, allow_repairs=False)
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 2
    assert all(row["metadata"]["episode_id"] == "episode-001" for row in rows)
    assert manifest["counts"]["skipped_non_train_split"] == 1
    assert manifest["counts"]["skipped_not_explicitly_sanitized"] == 1


def test_refuses_unregistered_tools_and_unverified_actions(tmp_path):
    source = tmp_path / "trajectories.jsonl"
    output = tmp_path / "train.jsonl"
    bad = valid_episode()
    bad["steps"][0]["action"] = {
        "type": "tool", "name": "shell", "args": {"command": "unsafe"}
    }
    write_jsonl(source, [bad])
    with pytest.raises(ValueError, match="unregistered tool"):
        build_dataset(source, output, max_context_chars=8000, allow_repairs=False)

    bad = valid_episode()
    bad["steps"][0]["observation_verified"] = False
    write_jsonl(source, [bad])
    with pytest.raises(ValueError, match="independently verified"):
        build_dataset(source, output, max_context_chars=8000, allow_repairs=False)


def test_repair_requires_explicit_prior_context_count(tmp_path):
    source = tmp_path / "trajectories.jsonl"
    output = tmp_path / "train.jsonl"
    bad = valid_episode()
    del bad["verified_repairs"][0]["completed_steps_before_failure"]
    write_jsonl(source, [bad])
    with pytest.raises(ValueError, match="completed_steps_before_failure"):
        build_dataset(source, output, max_context_chars=8000, allow_repairs=True)


def test_refuses_missing_license_or_empty_eligible_dataset(tmp_path):
    source = tmp_path / "trajectories.jsonl"
    output = tmp_path / "train.jsonl"
    bad = valid_episode()
    bad["license"] = ""
    write_jsonl(source, [bad])
    with pytest.raises(ValueError, match="no eligible verified training examples"):
        build_dataset(source, output, max_context_chars=8000, allow_repairs=False)



def test_rejects_obvious_credential_material_before_export(tmp_path):
    source = tmp_path / "trajectories.jsonl"
    output = tmp_path / "train.jsonl"
    row = valid_episode()
    row["steps"][0]["observation"]["summary"] = (
        "API key: sk-proj-abcdefghijklmnopqrstuvwxyz1234567890ABCDE"
    )
    write_jsonl(source, [row])
    with pytest.raises(ValueError, match="credential material detected"):
        build_dataset(source, output, max_context_chars=8000, allow_repairs=False)
    assert not output.exists()
