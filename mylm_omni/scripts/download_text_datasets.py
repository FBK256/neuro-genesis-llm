"""Download explicitly named Hugging Face text datasets with license/provenance checks."""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ALLOWED_DEFAULT = {"apache-2.0", "mit", "cc0-1.0", "cc-by-4.0", "cc-by-3.0", "public-domain"}


def _license(info: Any) -> str | None:
    card_data = getattr(info, "card_data", None)
    value = getattr(card_data, "license", None)
    if value is None and isinstance(card_data, dict):
        value = card_data.get("license")
    if isinstance(value, list):
        return str(value[0]).strip().lower() if value else None
    return str(value).strip().lower() if value else None


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")[:100] or "dataset"


def _download_one(
    dataset_spec: str,
    output_dir: Path,
    text_fields: list[str],
    allow_licenses: set[str],
    allow_unknown_license: bool,
) -> dict:
    try:
        dataset_id, split = dataset_spec.rsplit(":", 1)
    except ValueError as exc:
        raise ValueError(f"dataset must be written as DATASET_ID:SPLIT: {dataset_spec}") from exc
    if not dataset_id or not split:
        raise ValueError(f"invalid dataset spec: {dataset_spec}")

    from datasets import load_dataset
    from huggingface_hub import HfApi

    api = HfApi()
    info = api.dataset_info(dataset_id)
    license_id = _license(info)
    if license_id is None and not allow_unknown_license:
        raise PermissionError(f"{dataset_id}: license metadata missing; refusing download")
    if license_id is not None and license_id not in allow_licenses:
        raise PermissionError(
            f"{dataset_id}: license {license_id!r} not in the explicitly allowed set "
            f"{sorted(allow_licenses)}; review the card and pass --allow-license if appropriate"
        )

    dataset = load_dataset(dataset_id, split=split)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{_safe_name(dataset_id)}-{_safe_name(split)}.jsonl"
    rows_written = 0
    with output_path.open("w", encoding="utf-8") as out:
        for row_number, row in enumerate(dataset):
            pieces = []
            for field in text_fields:
                value = row.get(field)
                if isinstance(value, str) and value.strip():
                    pieces.append(value.strip())
            if not pieces:
                continue
            out.write(json.dumps({
                "text": "\n\n".join(pieces),
                "source_dataset": dataset_id,
                "source_split": split,
                "source_row": row_number,
                "license": license_id,
                "dataset_revision": getattr(info, "sha", None),
            }, ensure_ascii=False) + "\n")
            rows_written += 1

    provenance = {
        "dataset_id": dataset_id,
        "split": split,
        "license": license_id,
        "revision": getattr(info, "sha", None),
        "text_fields": text_fields,
        "rows_written": rows_written,
        "output": str(output_path),
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
        "transform": "selected text fields joined with blank lines; non-text fields excluded",
    }
    with (output_dir / "provenance.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(provenance, ensure_ascii=False) + "\n")
    return provenance


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", action="append", required=True, metavar="ID:SPLIT",
        help="Hugging Face dataset ID and split; repeat to download more than one",
    )
    parser.add_argument("--output-dir", default="data/text/raw")
    parser.add_argument("--text-fields", nargs="+", default=["text"])
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--allow-license", action="append", default=[],
                        help="additional exact license identifier reviewed by you")
    parser.add_argument("--allow-unknown-license", action="store_true",
                        help="override missing license metadata; use only after independent review")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be >= 1")
    licenses = ALLOWED_DEFAULT | {item.casefold() for item in args.allow_license}
    output = Path(args.output_dir)
    # Concurrency is bounded because a few large datasets can exhaust RAM/disk bandwidth.
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(args.workers, 4)) as pool:
        futures = {
            pool.submit(
                _download_one, spec, output, args.text_fields, licenses, args.allow_unknown_license
            ): spec
            for spec in args.dataset
        }
        for future in concurrent.futures.as_completed(futures):
            spec = futures[future]
            try:
                print(json.dumps(future.result(), ensure_ascii=False))
            except Exception as exc:
                print(json.dumps({"dataset": spec, "status": "failed", "error": str(exc)},
                                 ensure_ascii=False))
                raise


if __name__ == "__main__":
    main()
