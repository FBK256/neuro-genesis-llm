"""Download an explicitly chosen Hugging Face image-text split to a local image manifest.

The script preserves original image bytes in content-addressed files, checks that images can
be decoded, and records license/revision for each sample. It does not scrape arbitrary URLs.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ALLOWED_DEFAULT = {"apache-2.0", "mit", "cc0-1.0", "cc-by-4.0", "cc-by-3.0", "public-domain"}


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")[:100] or "dataset"


def _license(info: Any) -> str | None:
    card_data = getattr(info, "card_data", None)
    value = getattr(card_data, "license", None)
    if value is None and isinstance(card_data, dict):
        value = card_data.get("license")
    if isinstance(value, list):
        return str(value[0]).strip().lower() if value else None
    return str(value).strip().lower() if value else None


def _digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def _materialize_image(value: Any, image_dir: Path) -> tuple[str, str]:
    from PIL import Image

    if not isinstance(value, dict):
        raise ValueError("image column must return a path/bytes mapping")
    source_path = value.get("path")
    raw_bytes = value.get("bytes")
    temp_path: Path | None = None
    if source_path and Path(source_path).is_file():
        temp_path = Path(source_path)
    elif raw_bytes:
        temp_path = image_dir / f".incoming-{hashlib.sha256(raw_bytes).hexdigest()}"
        temp_path.write_bytes(raw_bytes)
    else:
        raise FileNotFoundError("image row has neither local cached path nor embedded bytes")
    try:
        with Image.open(temp_path) as image:
            image.verify()
            image_format = (image.format or "PNG").lower()
        digest = _digest(temp_path)
        extension = {"jpeg": "jpg", "tiff": "tif"}.get(image_format, image_format)
        destination = image_dir / f"{digest}.{extension}"
        if not destination.exists():
            shutil.copy2(temp_path, destination)
        return str(destination.resolve()), digest
    finally:
        if raw_bytes and temp_path is not None:
            temp_path.unlink(missing_ok=True)


def _download_one(
    spec: str,
    output_dir: Path,
    image_field: str,
    text_fields: list[str],
    allowed_licenses: set[str],
    allow_unknown_license: bool,
) -> dict:
    try:
        dataset_id, split = spec.rsplit(":", 1)
    except ValueError as exc:
        raise ValueError(f"dataset must be ID:SPLIT: {spec}") from exc
    from datasets import Image, load_dataset
    from huggingface_hub import HfApi

    info = HfApi().dataset_info(dataset_id)
    license_id = _license(info)
    if license_id is None and not allow_unknown_license:
        raise PermissionError(f"{dataset_id}: missing license metadata; refusing download")
    if license_id and license_id not in allowed_licenses:
        raise PermissionError(
            f"{dataset_id}: license {license_id!r} not approved; review and pass --allow-license"
        )
    dataset = load_dataset(dataset_id, split=split)
    if image_field not in dataset.column_names:
        raise ValueError(f"{dataset_id} has no image column {image_field!r}")
    dataset = dataset.cast_column(image_field, Image(decode=False))

    output_dir.mkdir(parents=True, exist_ok=True)
    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / f"{_safe_name(dataset_id)}-{_safe_name(split)}.jsonl"
    written = 0
    skipped = 0
    with manifest_path.open("w", encoding="utf-8") as out:
        for row_index, row in enumerate(dataset):
            try:
                image_path, digest = _materialize_image(row.get(image_field), image_dir)
            except Exception as exc:
                skipped += 1
                print(f"SKIP {dataset_id}[{row_index}]: image decode/materialize failed: {exc}")
                continue
            captions = [
                str(row[field]).strip() for field in text_fields
                if field in row and isinstance(row[field], str) and row[field].strip()
            ]
            if not captions:
                skipped += 1
                continue
            out.write(json.dumps({
                "image_path": image_path,
                "image_sha256": digest,
                "text": "\n".join(captions),
                "source_dataset": dataset_id,
                "source_split": split,
                "source_row": row_index,
                "dataset_revision": getattr(info, "sha", None),
                "license": license_id,
            }, ensure_ascii=False) + "\n")
            written += 1

    provenance = {
        "dataset_id": dataset_id, "split": split, "license": license_id,
        "revision": getattr(info, "sha", None), "image_field": image_field,
        "text_fields": text_fields, "rows_written": written, "rows_skipped": skipped,
        "manifest": str(manifest_path),
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
    }
    with (output_dir / "provenance.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(provenance, ensure_ascii=False) + "\n")
    return provenance


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", action="append", required=True, metavar="ID:SPLIT")
    parser.add_argument("--output-dir", default="data/multimodal/raw")
    parser.add_argument("--image-field", default="image")
    parser.add_argument("--text-fields", nargs="+", default=["text", "caption"])
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--allow-license", action="append", default=[])
    parser.add_argument("--allow-unknown-license", action="store_true")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be >= 1")
    licenses = ALLOWED_DEFAULT | {value.casefold() for value in args.allow_license}
    output_dir = Path(args.output_dir)
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(args.workers, 3)) as pool:
        futures = {
            pool.submit(
                _download_one, spec, output_dir, args.image_field, args.text_fields,
                licenses, args.allow_unknown_license,
            ): spec for spec in args.dataset
        }
        for future in concurrent.futures.as_completed(futures):
            try:
                print(json.dumps(future.result(), ensure_ascii=False))
            except Exception as exc:
                print(json.dumps({"dataset": futures[future], "status": "failed", "error": str(exc)},
                                 ensure_ascii=False))
                raise


if __name__ == "__main__":
    main()
