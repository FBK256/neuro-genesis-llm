"""Ephemeral one-time approval tokens for trusted local UI workflows."""
from __future__ import annotations

import secrets


class OneTimeApprovalAuthority:
    """Mint scoped, in-memory tokens after an explicit human confirmation.

    Tokens are deliberately not persisted. A remote or production UI must authenticate the human
    separately before calling issue(); this helper only binds consent to one operation reference.
    """

    def __init__(self) -> None:
        self._issued: dict[str, str] = {}

    def issue(self, operation_reference: str) -> str:
        if not operation_reference.strip():
            raise ValueError("operation_reference must not be empty")
        token = secrets.token_urlsafe(32)
        self._issued[token] = operation_reference
        return token

    def verify(self, token: str, operation_reference: str) -> bool:
        """Consume every token attempt so a token cannot be replayed or repurposed."""
        expected = self._issued.pop(token, None)
        return (
            expected is not None
            and bool(operation_reference)
            and secrets.compare_digest(expected, operation_reference)
        )
