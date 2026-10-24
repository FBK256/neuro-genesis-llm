"""Review deletion candidates in a directly controlled local console.

Only records already marked deletion_candidate are shown. Typing DELETE plus the short proposal ID
issues a one-time in-memory approval token for that exact proposal. Typing KEEP restores the record
to its prior status. This script does not create or auto-approve deletion proposals.

Security boundary: this is intended for a trusted, interactive local terminal. Do not expose it via
a remote unauthenticated shell or web endpoint. A production GUI should replace the console prompt
with authenticated user identity and a UI-issued, one-time approval token.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from mylm_omni.memory import MemoryStore
from mylm_omni.memory_approval import OneTimeApprovalAuthority


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path("./workspace"))
    parser.add_argument("--limit", type=int, default=50)
    args = parser.parse_args()
    if not 1 <= args.limit <= 500:
        parser.error("--limit must be in [1, 500]")

    workspace = args.workspace.expanduser().resolve()
    if not workspace.is_dir():
        raise NotADirectoryError(workspace)
    database_path = workspace / ".mylm-omni-memory.sqlite"
    if not database_path.is_file():
        raise FileNotFoundError(f"memory database not found: {database_path}")

    authority = OneTimeApprovalAuthority()
    store = MemoryStore(
        database_path,
        approval_verifier=lambda token, proposal_id: authority.verify(
            token, f"memory-deletion:{proposal_id}"
        ),
    )
    proposals = store.list_pending_deletions(limit=args.limit)
    if not proposals:
        print("No pending memory deletions.")
        return 0

    print(f"Pending memory deletion candidates: {len(proposals)}")
    print("They are excluded from normal retrieval but have not been erased.")
    resolved = 0
    for item in proposals:
        print("\n" + "=" * 78)
        print(f"Proposal ID : {item['proposal_id']}")
        print(f"Record ID   : {item['record_id']}")
        print(f"Type / key  : {item['memory_type']} / {item['memory_key']}")
        print(f"Version     : {item['version']}")
        print(f"Confidence  : {item['confidence']}")
        print(f"Source      : {item['source'] or '(no source)'}")
        print(f"Valid from  : {item['valid_from'] or '(not set)'}")
        print(f"Valid until : {item['valid_until'] or '(not set)'}")
        print(f"Requested by: {item['requested_by']}")
        print(f"AI reason   : {item['proposal_reason']}")
        print("Memory content (review locally before making a decision):")
        print(str(item["content"])[:3000])
        if len(str(item["content"])) > 3000:
            print("[content preview truncated]")
        short_id = item["proposal_id"][:8]
        try:
            decision = input(
                f"Type DELETE {short_id} to permanently erase, KEEP {short_id} to restore, "
                "or ENTER to leave pending: "
            ).strip()
        except (EOFError, KeyboardInterrupt):
            print("\nReview interrupted; remaining candidates stay pending.")
            return 130

        if decision == f"DELETE {short_id}":
            try:
                confirmation = input(
                    f"This is irreversible at the application level. Re-type DELETE {short_id}: "
                ).strip()
            except (EOFError, KeyboardInterrupt):
                print("\nReview interrupted; this candidate stays pending.")
                return 130
            if confirmation != f"DELETE {short_id}":
                print("Confirmation did not match; candidate remains pending.")
                continue
            # The approval token is issued only after the exact proposal has been displayed and
            # explicitly confirmed, and it can be consumed only once for this proposal ID.
            token = authority.issue(f"memory-deletion:{item['proposal_id']}")
            result = store.resolve_deletion(
                item["proposal_id"],
                approved=True,
                actor="human-local-console",
                reason="Explicit confirmation in review_memory_deletions.py",
                approval_token=token,
            )
            print(f"Erasure approved: {result['record_status']} (proposal {short_id})")
            resolved += 1
        elif decision == f"KEEP {short_id}":
            result = store.resolve_deletion(
                item["proposal_id"],
                approved=False,
                actor="human-local-console",
                reason="Human review rejected the deletion proposal",
            )
            print(f"Memory restored to status: {result['record_status']}")
            resolved += 1
        else:
            print("Left pending; no permanent deletion was performed.")

    print(f"Review finished. Decisions applied: {resolved}; candidates originally listed: {len(proposals)}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, KeyError, PermissionError) as exc:
        print(f"Review error: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)
