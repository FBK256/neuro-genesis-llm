"""Durable local memory with source tracking, conflict review, and deletion proposals.

SQLite stores versioned episodic/semantic/procedural memories. Optional embeddings are supplied
by a local embedder and stored alongside lexical metadata; without one, retrieval stays an
explicit lexical baseline instead of silently faking vector search.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import sqlite3
from contextlib import contextmanager
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _terms(text: str) -> set[str]:
    """Extract simple Unicode word tokens for a transparent lexical baseline."""
    return {token.casefold() for token in re.findall(r"[^\W_]+", text, flags=re.UNICODE)}


class MemoryStore:
    """SQLite-backed memory store.

    Memory types can include episodic, semantic, preference, procedural, reasoning, and
    archive. A contradiction is never silently overwritten: the new record enters review.
    Permanent deletion is a separate approval flow; candidate records are not retrieved.
    """

    RETRIEVABLE_STATUSES = ("active",)

    def __init__(
        self,
        db_path: str | Path,
        require_semantic_source: bool = True,
        approval_verifier: Callable[[str, str], bool] | None = None,
        embedder: Callable[[str], list[float]] | None = None,
        min_semantic_similarity: float = 0.20,
    ) -> None:
        if not -1.0 <= min_semantic_similarity <= 1.0:
            raise ValueError("min_semantic_similarity must be in [-1, 1]")
        self.db_path = str(db_path)
        self.require_semantic_source = require_semantic_source
        self.embedder = embedder
        self.min_semantic_similarity = min_semantic_similarity
        # The application UI should inject a verifier for an approval token it issued after
        # a real human confirmation. Never infer authorization from an LLM-supplied actor name.
        self.approval_verifier = approval_verifier
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 30000")
        connection.execute("PRAGMA secure_delete = ON")
        return connection

    @contextmanager
    def _session(self):
        """Commit/rollback a transaction and always close its SQLite connection."""
        connection = self._connect()
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self._session() as db:
            db.executescript(
                """
                CREATE TABLE IF NOT EXISTS memories (
                    id TEXT PRIMARY KEY,
                    memory_type TEXT NOT NULL,
                    memory_key TEXT NOT NULL,
                    content TEXT NOT NULL,
                    source TEXT,
                    confidence REAL NOT NULL,
                    valid_from TEXT,
                    valid_until TEXT,
                    status TEXT NOT NULL,
                    version INTEGER NOT NULL,
                    supersedes TEXT,
                    metadata_json TEXT NOT NULL,
                    embedding_json TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_memory_key_status
                    ON memories(memory_type, memory_key, status);
                CREATE INDEX IF NOT EXISTS idx_memory_status
                    ON memories(status, memory_type);
                CREATE TABLE IF NOT EXISTS memory_audit (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    at TEXT NOT NULL,
                    record_id TEXT NOT NULL,
                    operation TEXT NOT NULL,
                    actor TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    content_sha256 TEXT,
                    status_before TEXT,
                    status_after TEXT
                );
                CREATE TABLE IF NOT EXISTS deletion_proposals (
                    id TEXT PRIMARY KEY,
                    record_id TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    old_status TEXT NOT NULL,
                    state TEXT NOT NULL,
                    requested_by TEXT NOT NULL,
                    resolved_by TEXT,
                    created_at TEXT NOT NULL,
                    resolved_at TEXT
                );
                """
            )
            # Migrate older local databases without losing existing memory records.
            memory_columns = {
                row["name"] for row in db.execute("PRAGMA table_info(memories)").fetchall()
            }
            if "embedding_json" not in memory_columns:
                db.execute("ALTER TABLE memories ADD COLUMN embedding_json TEXT")

    def _embed(self, text: str) -> list[float] | None:
        """Compute and validate an optional embedding using an injected local model."""
        if self.embedder is None:
            return None
        vector = self.embedder(text)
        if not isinstance(vector, (list, tuple)) or not vector:
            raise ValueError("embedder must return a non-empty list/tuple of numbers")
        result = [float(value) for value in vector]
        if not all(math.isfinite(value) for value in result):
            raise ValueError("embedding contains NaN or infinity")
        norm = math.sqrt(sum(value * value for value in result))
        if norm <= 1e-12:
            raise ValueError("embedding vector norm is zero")
        # Normalize at write time, so cosine similarity becomes a dot product.
        return [value / norm for value in result]

    @staticmethod
    def _cosine_normalized(left: list[float], right: list[float]) -> float:
        if len(left) != len(right):
            raise ValueError("query embedding dimension differs from stored memory embedding")
        return sum(a * b for a, b in zip(left, right))

    def _audit(
        self,
        db: sqlite3.Connection,
        record_id: str,
        operation: str,
        actor: str,
        reason: str,
        content: str | None,
        status_before: str | None,
        status_after: str | None,
    ) -> None:
        # Audit only records a hash and status metadata, not a second copy of private content.
        db.execute(
            """INSERT INTO memory_audit
               (at, record_id, operation, actor, reason, content_sha256, status_before, status_after)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                _now(), record_id, operation, actor, reason,
                _sha(content) if content else None, status_before, status_after,
            ),
        )

    def add(
        self,
        *,
        memory_type: str,
        key: str,
        content: str,
        source: str | None,
        confidence: float = 0.8,
        valid_from: str | None = None,
        valid_until: str | None = None,
        metadata: dict[str, Any] | None = None,
        supersedes: str | None = None,
        approval_token: str | None = None,
        actor: str = "agent",
        reason: str = "new memory candidate",
    ) -> dict[str, Any]:
        """Add a memory. Conflict replacement requires a verified human approval token."""
        if supersedes:
            approval_ref = f"memory-supersede:{memory_type}:{key}:{supersedes}"
            if not approval_token or self.approval_verifier is None:
                raise PermissionError("superseding an existing memory requires a human approval token")
            if not self.approval_verifier(approval_token, approval_ref):
                raise PermissionError("memory supersede approval token is invalid or expired")
        if not memory_type.strip() or not key.strip() or not content.strip():
            raise ValueError("memory_type, key, and content must be non-empty")
        if not 0 <= confidence <= 1:
            raise ValueError("confidence must be between 0 and 1")
        if self.require_semantic_source and memory_type == "semantic" and not source:
            raise ValueError("semantic memories require a source reference")

        record_id = str(uuid.uuid4())
        now = _now()
        embedding = self._embed(key + "\n" + content)
        embedding_json = json.dumps(embedding) if embedding is not None else None
        status = "active"
        conflict_ids: list[str] = []
        with self._session() as db:
            prior = db.execute(
                """SELECT id, content, version, status FROM memories
                   WHERE memory_type=? AND memory_key=? AND status IN ('active', 'review')
                   ORDER BY version DESC""",
                (memory_type, key),
            ).fetchall()

            for old in prior:
                if old["content"] != content and old["id"] != supersedes:
                    conflict_ids.append(old["id"])

            if conflict_ids and not supersedes:
                status = "review"

            version = 1 + max((int(row["version"]) for row in prior), default=0)
            if supersedes:
                old = db.execute(
                    "SELECT id, content, status FROM memories WHERE id=?", (supersedes,)
                ).fetchone()
                if old is None:
                    raise ValueError(f"supersedes id does not exist: {supersedes}")
                if old["status"] not in ("active", "review"):
                    raise ValueError(f"cannot supersede memory in status {old['status']}")
                # A supersedes pointer must refer to the same logical slot, not an unrelated fact.
                target_slot = db.execute(
                    "SELECT memory_type, memory_key FROM memories WHERE id=?", (supersedes,)
                ).fetchone()
                if (target_slot["memory_type"], target_slot["memory_key"]) != (memory_type, key):
                    raise ValueError("supersedes record must have the same memory_type and key")
                db.execute(
                    "UPDATE memories SET status='superseded', updated_at=? WHERE id=?",
                    (now, supersedes),
                )
                self._audit(
                    db, supersedes, "supersede", actor,
                    f"superseded by {record_id}: {reason}", old["content"], old["status"], "superseded",
                )
                status = "review" if conflict_ids else "active"
            db.execute(
                """INSERT INTO memories
                   (id, memory_type, memory_key, content, source, confidence, valid_from, valid_until,
                    status, version, supersedes, metadata_json, embedding_json, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    record_id, memory_type, key, content, source, confidence, valid_from, valid_until,
                    status, version, supersedes, json.dumps(metadata or {}, ensure_ascii=False),
                    embedding_json, now, now,
                ),
            )
            self._audit(db, record_id, "add", actor, reason, content, None, status)
            if conflict_ids:
                self._audit(
                    db, record_id, "conflict_detected", actor,
                    "conflicts with record ids: " + ",".join(conflict_ids),
                    content, None, status,
                )
        return {"id": record_id, "status": status, "version": version, "conflicts": conflict_ids}

    def resolve_conflict(
        self,
        candidate_id: str,
        *,
        decision: str,
        actor: str,
        reason: str,
        approval_token: str | None = None,
        supersedes_id: str | None = None,
    ) -> dict[str, Any]:
        """Resolve a review candidate only after the UI verifies a token for this exact decision."""
        if decision not in {"accept", "reject", "supersede"}:
            raise ValueError("decision must be accept, reject, or supersede")
        approval_ref = f"memory-conflict:{candidate_id}:{decision}:{supersedes_id or ''}"
        if not approval_token or self.approval_verifier is None:
            raise PermissionError("resolving memory conflicts requires a human approval token")
        if not self.approval_verifier(approval_token, approval_ref):
            raise PermissionError("memory conflict approval token is invalid or expired")
        with self._session() as db:
            candidate = db.execute("SELECT * FROM memories WHERE id=?", (candidate_id,)).fetchone()
            if candidate is None:
                raise KeyError(candidate_id)
            if candidate["status"] != "review":
                raise ValueError("candidate must be in review status")
            before = candidate["status"]
            after = {"accept": "active", "reject": "archived", "supersede": "active"}[decision]
            conflicting_active = db.execute(
                """SELECT id, content FROM memories
                   WHERE memory_type=? AND memory_key=? AND status='active' AND id<>?""",
                (candidate["memory_type"], candidate["memory_key"], candidate_id),
            ).fetchall()
            conflicting_active = [
                row for row in conflicting_active if row["content"] != candidate["content"]
            ]
            if decision == "accept" and conflicting_active:
                raise ValueError(
                    "an active conflicting memory exists; use decision='supersede' and select its id"
                )
            if decision == "supersede":
                if not supersedes_id:
                    raise ValueError("supersede decision requires supersedes_id")
                old = db.execute(
                    "SELECT * FROM memories WHERE id=? AND status='active'", (supersedes_id,)
                ).fetchone()
                if old is None:
                    raise ValueError("supersedes_id must be an active memory")
                if (old["memory_type"], old["memory_key"]) != (
                    candidate["memory_type"], candidate["memory_key"]
                ):
                    raise ValueError("cannot supersede a different memory type/key")
                extra_conflicts = [
                    row for row in conflicting_active if row["id"] != supersedes_id
                ]
                if extra_conflicts:
                    raise ValueError(
                        "multiple active conflicting memories exist; resolve each conflict separately"
                    )
                db.execute(
                    "UPDATE memories SET status='superseded', updated_at=? WHERE id=?",
                    (_now(), supersedes_id),
                )
                self._audit(
                    db, supersedes_id, "supersede", actor, reason,
                    old["content"], "active", "superseded",
                )
                db.execute(
                    "UPDATE memories SET supersedes=? WHERE id=?", (supersedes_id, candidate_id)
                )
            db.execute(
                "UPDATE memories SET status=?, updated_at=? WHERE id=?",
                (after, _now(), candidate_id),
            )
            self._audit(db, candidate_id, "resolve_conflict", actor, reason,
                        candidate["content"], before, after)
        return {"id": candidate_id, "status": after, "decision": decision}

    def retrieve(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        """Return active records using hybrid lexical + optional local-vector retrieval.

        Vector search is enabled only when a local embedder was injected. Without it, this
        method returns only lexical matches. Confidence is a small tie-breaker; it cannot
        make an unrelated record match a query.
        """
        if limit < 1:
            return []
        query_terms = _terms(query)
        query_embedding = self._embed(query) if self.embedder is not None else None
        now = _now()
        with self._session() as db:
            rows = db.execute(
                """SELECT * FROM memories WHERE status='active'
                   AND (valid_from IS NULL OR valid_from <= ?)
                   AND (valid_until IS NULL OR valid_until >= ?)
                   ORDER BY updated_at DESC""",
                (now, now),
            ).fetchall()
        ranked: list[tuple[float, dict[str, Any]]] = []
        for row in rows:
            item = dict(row)
            item["metadata"] = json.loads(item.pop("metadata_json"))
            stored_embedding_json = item.pop("embedding_json", None)
            stored_embedding = json.loads(stored_embedding_json) if stored_embedding_json else None
            terms = _terms(item["memory_key"] + " " + item["content"])
            overlap = len(query_terms & terms) / max(1, len(query_terms)) if query_terms else 0.0
            similarity = None
            if query_embedding is not None and stored_embedding is not None:
                similarity = self._cosine_normalized(query_embedding, stored_embedding)

            lexical_match = overlap > 0
            semantic_match = (
                similarity is not None and similarity >= self.min_semantic_similarity
            )
            if query_terms and not lexical_match and not semantic_match:
                continue
            if query_embedding is None:
                score = overlap + 0.05 * float(item["confidence"])
            else:
                semantic_score = max(0.0, similarity or 0.0)
                # Weights emphasize semantic similarity while preserving exact-key matches.
                score = 0.60 * semantic_score + 0.35 * overlap + 0.05 * float(item["confidence"])
            item["lexical_overlap"] = overlap
            item["semantic_similarity"] = similarity
            item["retrieval_score"] = score
            ranked.append((score, item))
        ranked.sort(key=lambda pair: (pair[0], pair[1]["updated_at"]), reverse=True)
        return [item for _, item in ranked[:limit]]

    def list_pending_deletions(self, limit: int = 50) -> list[dict[str, Any]]:
        """Return local deletion proposals for a human review UI; do not expose them to normal retrieval."""
        if limit < 1 or limit > 500:
            raise ValueError("limit must be between 1 and 500")
        with self._session() as db:
            rows = db.execute(
                """SELECT p.id AS proposal_id, p.record_id, p.reason AS proposal_reason,
                          p.old_status, p.state, p.requested_by, p.created_at,
                          m.memory_type, m.memory_key, m.content, m.source,
                          m.confidence, m.valid_from, m.valid_until, m.version
                   FROM deletion_proposals p
                   JOIN memories m ON m.id = p.record_id
                   WHERE p.state='pending' AND m.status='deletion_candidate'
                   ORDER BY p.created_at ASC LIMIT ?""",
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]

    def get_memory(self, record_id: str) -> dict[str, Any]:
        """Fetch one memory record by ID for a trusted local review UI."""
        with self._session() as db:
            row = db.execute("SELECT * FROM memories WHERE id=?", (record_id,)).fetchone()
        if row is None:
            raise KeyError(record_id)
        item = dict(row)
        item["metadata"] = json.loads(item.pop("metadata_json"))
        item.pop("embedding_json", None)
        return item

    def propose_deletion(
        self, record_id: str, *, reason: str, actor: str = "agent"
    ) -> str:
        """Stop retrieval immediately, but require an explicit approval for erasure."""
        proposal_id = str(uuid.uuid4())
        with self._session() as db:
            row = db.execute("SELECT * FROM memories WHERE id=?", (record_id,)).fetchone()
            if row is None:
                raise KeyError(record_id)
            if row["status"] not in ("active", "review", "archived", "superseded"):
                raise ValueError(f"cannot propose deletion from status {row['status']}")
            now = _now()
            db.execute(
                """INSERT INTO deletion_proposals
                   (id, record_id, reason, old_status, state, requested_by, created_at)
                   VALUES (?, ?, ?, ?, 'pending', ?, ?)""",
                (proposal_id, record_id, reason, row["status"], actor, now),
            )
            db.execute(
                "UPDATE memories SET status='deletion_candidate', updated_at=? WHERE id=?",
                (now, record_id),
            )
            self._audit(
                db, record_id, "deletion_proposed", actor, reason, row["content"],
                row["status"], "deletion_candidate",
            )
        return proposal_id

    def resolve_deletion(
        self,
        proposal_id: str,
        *,
        approved: bool,
        actor: str,
        approval_token: str | None = None,
        reason: str | None = None,
    ) -> dict[str, Any]:
        """Approve logical erasure only with a UI-verifiable human approval token.

        The UI must create the approval token after an authenticated person confirms the exact
        proposal. A model-supplied string such as actor="human" is not proof of approval.
        Without an injected verifier, permanent deletion is disabled by default.
        """
        if approved:
            if not approval_token or self.approval_verifier is None:
                raise PermissionError(
                    "permanent deletion requires an approval token from the authenticated UI"
                )
            if not self.approval_verifier(approval_token, proposal_id):
                raise PermissionError("human deletion approval token is invalid or expired")
        with self._session() as db:
            proposal = db.execute(
                "SELECT * FROM deletion_proposals WHERE id=?", (proposal_id,)
            ).fetchone()
            if proposal is None:
                raise KeyError(proposal_id)
            if proposal["state"] != "pending":
                raise ValueError(f"proposal already resolved: {proposal['state']}")
            row = db.execute(
                "SELECT * FROM memories WHERE id=?", (proposal["record_id"],)
            ).fetchone()
            if row is None:
                raise KeyError(proposal["record_id"])
            now = _now()
            if approved:
                # Do not duplicate content in the audit. secure_delete is enabled, but storage
                # snapshots/backups and SSD wear-leveling are outside SQLite's control.
                db.execute(
                    """UPDATE memories SET content='[ERASED]', source=NULL, confidence=0,
                       valid_from=NULL, valid_until=NULL, status='deleted', metadata_json='{}',
                       embedding_json=NULL, updated_at=? WHERE id=?""",
                    (now, row["id"]),
                )
                state, after, default_reason = "approved", "deleted", "human-approved logical erasure"
                content_for_hash = row["content"]
            else:
                db.execute(
                    "UPDATE memories SET status=?, updated_at=? WHERE id=?",
                    (proposal["old_status"], now, row["id"]),
                )
                state, after, default_reason = "rejected", proposal["old_status"], "deletion rejected"
                content_for_hash = row["content"]
            db.execute(
                """UPDATE deletion_proposals SET state=?, resolved_by=?, resolved_at=?
                   WHERE id=?""",
                (state, actor, now, proposal_id),
            )
            audit_reason = (
                reason.strip()[:1000]
                if isinstance(reason, str) and reason.strip()
                else default_reason
            )
            self._audit(
                db, row["id"], "deletion_resolved", actor, audit_reason,
                content_for_hash, row["status"], after,
            )
        return {"proposal_id": proposal_id, "state": state, "record_status": after}

    def audit(self, record_id: str | None = None) -> list[dict[str, Any]]:
        """Return audit metadata. Private content is never stored in audit rows."""
        with self._session() as db:
            if record_id:
                rows = db.execute(
                    "SELECT * FROM memory_audit WHERE record_id=? ORDER BY id", (record_id,)
                ).fetchall()
            else:
                rows = db.execute("SELECT * FROM memory_audit ORDER BY id").fetchall()
        return [dict(row) for row in rows]
