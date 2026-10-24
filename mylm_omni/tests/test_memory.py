import json

import pytest

from mylm_omni.memory import MemoryStore


def test_conflict_is_not_silently_overwritten(tmp_path):
    store = MemoryStore(
        tmp_path / "memory.sqlite",
        approval_verifier=lambda token, reference: token == f"approved:{reference}",
    )
    original = store.add(
        memory_type="semantic", key="tax rate", content="tax rate is 10 percent",
        source="official source version A", confidence=0.8,
    )
    candidate = store.add(
        memory_type="semantic", key="tax rate", content="tax rate is 12 percent",
        source="official source version B", confidence=0.9,
    )
    assert original["status"] == "active"
    assert candidate["status"] == "review"
    assert candidate["conflicts"] == [original["id"]]
    results = store.retrieve("tax rate")
    assert [item["id"] for item in results] == [original["id"]]

    with pytest.raises(PermissionError, match="human approval token"):
        store.resolve_conflict(
            candidate["id"], decision="supersede", supersedes_id=original["id"],
            actor="human", reason="official rule updated",
        )
    decision = store.resolve_conflict(
        candidate["id"], decision="supersede", supersedes_id=original["id"],
        actor="human", reason="official rule updated",
        approval_token=f"approved:memory-conflict:{candidate['id']}:supersede:{original['id']}",
    )
    assert decision["status"] == "active"
    assert [item["id"] for item in store.retrieve("tax rate")] == [candidate["id"]]


def test_delete_requires_approval_and_candidate_is_not_retrieved(tmp_path):
    store = MemoryStore(
        tmp_path / "memory.sqlite",
        approval_verifier=lambda token, proposal_id: token == f"approved:{proposal_id}",
    )
    item = store.add(
        memory_type="preference", key="format", content="prefer markdown",
        source="user correction", confidence=1.0,
    )
    proposal = store.propose_deletion(item["id"], reason="possibly obsolete")
    assert store.retrieve("markdown") == []
    resolution = store.resolve_deletion(proposal, approved=False, actor="human")
    assert resolution["record_status"] == "active"
    assert store.retrieve("markdown")[0]["id"] == item["id"]

    proposal = store.propose_deletion(item["id"], reason="user-approved removal")
    with pytest.raises(PermissionError, match="approval token"):
        store.resolve_deletion(proposal, approved=True, actor="human")
    with pytest.raises(PermissionError, match="invalid or expired"):
        store.resolve_deletion(proposal, approved=True, actor="human", approval_token="forged")
    resolution = store.resolve_deletion(
        proposal, approved=True, actor="human", approval_token=f"approved:{proposal}"
    )
    assert resolution["record_status"] == "deleted"
    with store._session() as db:
        row = db.execute("SELECT content, source, status FROM memories WHERE id=?", (item["id"],)).fetchone()
    assert row["content"] == "[ERASED]"
    assert row["source"] is None
    assert row["status"] == "deleted"
    audit_text = json.dumps(store.audit(item["id"]), ensure_ascii=False)
    assert "prefer markdown" not in audit_text


def test_semantic_memory_requires_source(tmp_path):
    store = MemoryStore(tmp_path / "memory.sqlite")
    with pytest.raises(ValueError, match="require a source"):
        store.add(memory_type="semantic", key="x", content="y", source=None)



def test_direct_memory_supersede_requires_verified_token(tmp_path):
    store = MemoryStore(
        tmp_path / "memory.sqlite",
        approval_verifier=lambda token, reference: token == f"approved:{reference}",
    )
    old = store.add(
        memory_type="preference", key="format", content="format A",
        source="confirmed evidence",
    )
    with pytest.raises(PermissionError, match="requires a human approval token"):
        store.add(
            memory_type="preference", key="format", content="format B",
            source="new evidence", supersedes=old["id"],
        )
    updated = store.add(
        memory_type="preference", key="format", content="format B",
        source="new evidence",
        supersedes=old["id"],
        approval_token=f"approved:memory-supersede:preference:format:{old['id']}",
    )
    assert updated["status"] == "active"
    assert store.retrieve("format")[0]["id"] == updated["id"]



def test_pending_deletion_review_can_restore_or_erase_only_with_scoped_token(tmp_path):
    from mylm_omni.memory_approval import OneTimeApprovalAuthority

    authority = OneTimeApprovalAuthority()
    store = MemoryStore(
        tmp_path / "memory.sqlite",
        approval_verifier=lambda token, proposal_id: authority.verify(
            token, f"memory-deletion:{proposal_id}"
        ),
    )
    active = store.add(
        memory_type="procedural",
        key="obsolete deployment",
        content="use the v1 deployment command",
        source="incident report 2024-11",
    )
    proposal_id = store.propose_deletion(
        active["id"], reason="superseded by the validated deployment v3", actor="local-agent"
    )
    pending = store.list_pending_deletions()
    assert len(pending) == 1
    assert pending[0]["proposal_id"] == proposal_id
    assert pending[0]["content"] == "use the v1 deployment command"
    assert store.retrieve("deployment v1") == []

    with pytest.raises(PermissionError, match="approval token"):
        store.resolve_deletion(
            proposal_id, approved=True, actor="human-local-console"
        )

    token = authority.issue(f"memory-deletion:{proposal_id}")
    resolved = store.resolve_deletion(
        proposal_id,
        approved=True,
        actor="human-local-console",
        reason="reviewed and approved",
        approval_token=token,
    )
    assert resolved["state"] == "approved"
    assert resolved["record_status"] == "deleted"
    erased = store.get_memory(active["id"])
    assert erased["content"] == "[ERASED]"
    assert erased["source"] is None
    assert store.retrieve("deployment v1") == []


def test_rejected_deletion_restores_previous_memory_status(tmp_path):
    store = MemoryStore(tmp_path / "memory.sqlite")
    item = store.add(
        memory_type="preference",
        key="response style",
        content="prefer concise answers",
        source="human-confirmed preference",
    )
    proposal = store.propose_deletion(item["id"], reason="possibly stale", actor="agent")
    result = store.resolve_deletion(
        proposal, approved=False, actor="human-local-console", reason="still useful"
    )
    assert result["state"] == "rejected"
    assert result["record_status"] == "active"
    assert store.retrieve("response style")[0]["id"] == item["id"]
