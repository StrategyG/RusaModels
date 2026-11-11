"""Verify the integrity and internal consistency of a released JSONL dataset."""
from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import re
import sys
from pathlib import Path
from typing import Any


class DatasetVerificationError(ValueError):
    """Raised when a dataset release fails integrity or consistency checks."""


_EXPECTED_DATA_FILES = {
    "train.jsonl",
    "validation.jsonl",
    "test.jsonl",
    "provenance.jsonl",
}
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    )


def _fingerprint(record: dict[str, Any]) -> str:
    content = {key: value for key, value in record.items() if key not in {"id", "group_id"}}
    return hashlib.sha256(_canonical_json(content).encode("utf-8")).hexdigest()


def _load_json(raw: str, source: str) -> Any:
    try:
        return json.loads(raw, parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError(f"non-standard JSON constant: {value}")
        ))
    except (json.JSONDecodeError, ValueError) as exc:
        raise DatasetVerificationError(f"{source}: invalid JSON: {exc}") from exc


def _hash_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def _safe_release_file(root: Path, name: str) -> Path:
    if name not in _EXPECTED_DATA_FILES:
        raise DatasetVerificationError(f"manifest references unsupported file: {name!r}")
    path = root / name
    if path.is_symlink() or path.resolve().parent != root or not path.is_file():
        raise DatasetVerificationError(f"dataset release file is missing or unsafe: {name}")
    return path


def verify_dataset_release(release_path: str | Path) -> dict[str, Any]:
    """Verify manifest, checksums, split records, provenance, and split isolation."""
    root = Path(release_path).expanduser().resolve()
    if not root.is_dir():
        raise DatasetVerificationError(f"dataset release directory does not exist: {root}")
    manifest_path = root / "manifest.json"
    manifest_digest_path = root / "manifest.sha256"
    if manifest_path.is_symlink() or not manifest_path.is_file() or manifest_path.resolve().parent != root:
        raise DatasetVerificationError("manifest.json is missing or unsafe")
    if manifest_digest_path.is_symlink() or not manifest_digest_path.is_file() or manifest_digest_path.resolve().parent != root:
        raise DatasetVerificationError("manifest.sha256 is missing or unsafe")
    manifest_bytes = manifest_path.read_bytes()
    expected_manifest_digest = manifest_digest_path.read_text(encoding="ascii").strip()
    actual_manifest_digest = hashlib.sha256(manifest_bytes).hexdigest()
    if not _SHA256.fullmatch(expected_manifest_digest) or not hmac.compare_digest(
        expected_manifest_digest, actual_manifest_digest
    ):
        raise DatasetVerificationError("manifest SHA-256 verification failed")
    manifest = _load_json(manifest_bytes.decode("utf-8"), "manifest.json")
    if not isinstance(manifest, dict) or manifest.get("schema_version") != "1.0.0":
        raise DatasetVerificationError("unsupported or malformed dataset manifest")
    if manifest.get("dataset_id") != root.parent.name or manifest.get("dataset_version") != root.name:
        raise DatasetVerificationError("dataset identity does not match release directory")
    files = manifest.get("files")
    if not isinstance(files, dict) or set(files) != _EXPECTED_DATA_FILES:
        raise DatasetVerificationError("manifest must describe all four dataset release files")
    expected_entries = _EXPECTED_DATA_FILES | {"manifest.json", "manifest.sha256"}
    if {entry.name for entry in root.iterdir()} != expected_entries:
        raise DatasetVerificationError("dataset release contains missing or unmanifested files")

    for name in sorted(_EXPECTED_DATA_FILES):
        path = _safe_release_file(root, name)
        metadata = files[name]
        if not isinstance(metadata, dict) or not _SHA256.fullmatch(str(metadata.get("sha256", ""))):
            raise DatasetVerificationError(f"{name}: missing or invalid SHA-256 in manifest")
        actual_digest, actual_size = _hash_file(path)
        if not hmac.compare_digest(metadata["sha256"], actual_digest):
            raise DatasetVerificationError(f"{name}: SHA-256 verification failed")
        if metadata.get("size_bytes") != actual_size:
            raise DatasetVerificationError(f"{name}: size does not match manifest")

    source_rows = manifest.get("sources")
    if not isinstance(source_rows, list) or not source_rows:
        raise DatasetVerificationError("manifest must include source provenance")
    sources: dict[str, dict[str, Any]] = {}
    for source in source_rows:
        if not isinstance(source, dict):
            raise DatasetVerificationError("manifest contains malformed source metadata")
        source_id = source.get("source_id")
        if not isinstance(source_id, str) or not source_id or source_id in sources:
            raise DatasetVerificationError("manifest contains an invalid or duplicate source_id")
        if not _SHA256.fullmatch(str(source.get("sha256", ""))):
            raise DatasetVerificationError(f"source {source_id}: invalid source SHA-256")
        if not isinstance(source.get("license_id"), str) or not source["license_id"].strip():
            raise DatasetVerificationError(f"source {source_id}: license_id is missing")
        if type(source.get("records_read")) is not int or source["records_read"] < 0:
            raise DatasetVerificationError(f"source {source_id}: invalid records_read")
        sources[source_id] = source

    records_meta = manifest.get("records")
    if not isinstance(records_meta, dict):
        raise DatasetVerificationError("manifest records metadata is missing")
    split_counts = {"train": 0, "validation": 0, "test": 0}
    record_splits: dict[str, str] = {}
    record_fingerprints: dict[str, str] = {}
    fingerprints_seen: set[str] = set()
    for split in ("train", "validation", "test"):
        path = _safe_release_file(root, f"{split}.jsonl")
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    raise DatasetVerificationError(f"{path.name}:{line_number}: blank lines are not allowed")
                record = _load_json(line, f"{path.name}:{line_number}")
                if not isinstance(record, dict):
                    raise DatasetVerificationError(f"{path.name}:{line_number}: record must be an object")
                record_id = record.get("id")
                if not isinstance(record_id, str) or not record_id.strip():
                    raise DatasetVerificationError(f"{path.name}:{line_number}: record id is missing")
                if record_id in record_splits:
                    raise DatasetVerificationError(f"record id occurs more than once: {record_id}")
                fingerprint = _fingerprint(record)
                if fingerprint in fingerprints_seen:
                    raise DatasetVerificationError("duplicate content crosses or repeats dataset splits")
                record_splits[record_id] = split
                record_fingerprints[record_id] = fingerprint
                fingerprints_seen.add(fingerprint)
                split_counts[split] += 1
        if files[f"{split}.jsonl"].get("records") != split_counts[split]:
            raise DatasetVerificationError(f"{split}.jsonl record count does not match manifest")

    provenance_path = _safe_release_file(root, "provenance.jsonl")
    provenance_count = 0
    source_counts = {source_id: 0 for source_id in sources}
    source_lines: set[tuple[str, int]] = set()
    group_splits: dict[str, str] = {}
    records_with_provenance: set[str] = set()
    with provenance_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise DatasetVerificationError(
                    f"provenance.jsonl:{line_number}: blank lines are not allowed"
                )
            item = _load_json(line, f"provenance.jsonl:{line_number}")
            if not isinstance(item, dict):
                raise DatasetVerificationError(
                    f"provenance.jsonl:{line_number}: entry must be an object"
                )
            record_id = item.get("record_id")
            source_id = item.get("source_id")
            source_line = item.get("source_line")
            if (
                not isinstance(record_id, str)
                or not isinstance(source_id, str)
                or record_id not in record_splits
                or source_id not in sources
            ):
                raise DatasetVerificationError(
                    f"provenance.jsonl:{line_number}: unknown record or source"
                )
            if type(source_line) is not int or source_line < 1:
                raise DatasetVerificationError(
                    f"provenance.jsonl:{line_number}: source_line must be positive"
                )
            source_line_key = (source_id, source_line)
            if source_line_key in source_lines:
                raise DatasetVerificationError("duplicate source line in provenance")
            source_lines.add(source_line_key)
            source = sources[source_id]
            if item.get("source_sha256") != source["sha256"]:
                raise DatasetVerificationError("source checksum mismatch in provenance")
            if item.get("license_id") != source["license_id"]:
                raise DatasetVerificationError("source license mismatch in provenance")
            if item.get("record_fingerprint") != record_fingerprints[record_id]:
                raise DatasetVerificationError("record fingerprint mismatch in provenance")
            split = record_splits[record_id]
            if item.get("split") != split:
                raise DatasetVerificationError("provenance split does not match record split")
            group_id = item.get("group_id")
            if not isinstance(group_id, str) or not group_id:
                raise DatasetVerificationError("provenance group_id is missing")
            previous_split = group_splits.setdefault(group_id, split)
            if previous_split != split:
                raise DatasetVerificationError("a dataset group crosses train/validation/test splits")
            source_counts[source_id] += 1
            records_with_provenance.add(record_id)
            provenance_count += 1

    unique_records = len(record_splits)
    raw_records = sum(source["records_read"] for source in sources.values())
    if source_counts != {source_id: source["records_read"] for source_id, source in sources.items()}:
        raise DatasetVerificationError("per-source provenance counts do not match manifest")
    if records_with_provenance != set(record_splits):
        raise DatasetVerificationError("one or more released records have no provenance")
    if provenance_count != raw_records:
        raise DatasetVerificationError("provenance count does not match raw record count")
    if records_meta.get("raw_records") != raw_records:
        raise DatasetVerificationError("raw record count does not match source metadata")
    if records_meta.get("unique_records") != unique_records:
        raise DatasetVerificationError("unique record count does not match split files")
    if records_meta.get("duplicates_removed") != raw_records - unique_records:
        raise DatasetVerificationError("duplicate count is inconsistent")
    if records_meta.get("split_counts") != split_counts:
        raise DatasetVerificationError("split counts do not match split files")
    if files["provenance.jsonl"].get("records") != provenance_count:
        raise DatasetVerificationError("provenance record count does not match manifest")

    return {
        "dataset_id": manifest["dataset_id"],
        "dataset_version": manifest["dataset_version"],
        "unique_records": unique_records,
        "raw_records": raw_records,
        "split_counts": split_counts,
        "sources": len(sources),
        "verified_files": len(_EXPECTED_DATA_FILES) + 2,
        "manifest_sha256": actual_manifest_digest,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", required=True, type=Path, help="Dataset release directory")
    args = parser.parse_args(argv)
    try:
        result = verify_dataset_release(args.release)
    except (DatasetVerificationError, OSError, UnicodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
