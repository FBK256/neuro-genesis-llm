import pytest

from mylm_omni.memory_approval import OneTimeApprovalAuthority


def test_one_time_token_is_scoped_to_one_exact_operation():
    authority = OneTimeApprovalAuthority()
    reference = "memory-deletion:proposal-123"
    token = authority.issue(reference)
    assert not authority.verify(token, "memory-deletion:proposal-other")
    # The failed mismatched use consumes the token.
    assert not authority.verify(token, reference)


def test_one_time_token_can_be_used_once_for_the_correct_operation():
    authority = OneTimeApprovalAuthority()
    reference = "memory-deletion:proposal-456"
    token = authority.issue(reference)
    assert authority.verify(token, reference)
    assert not authority.verify(token, reference)


def test_empty_approval_reference_is_rejected():
    authority = OneTimeApprovalAuthority()
    with pytest.raises(ValueError, match="must not be empty"):
        authority.issue(" ")
