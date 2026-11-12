"""Verify dataset release integrity and provenance with disk-backed staging."""
from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import re
import sqlite3
import sys
import tempfile
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
    return hashlib.sha256(_canonical_json(content).encode()).hexdigest()


def _object_pairs_without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def _load_json(raw: str, source: str) -> Any:
    try:
        return json.loads(
            raw,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-standard JSON constant: {value}")
            ),
            object_pairs_hook=_object_pairs_without_duplicates,
        )
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


def _initialize_verification_db(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        CREATE TABLE records (
            record_id TEXT PRIMARY KEY,
            split TEXT NOT NULL,
            fingerprint TEXT NOT NULL UNIQUE
        );
        CREATE TABLE group_splits (
            group_id TEXT PRIMARY KEY,
            split TEXT NOT NULL
        );
        CREATE TABLE record_groups (
            record_id TEXT PRIMARY KEY,
            group_id TEXT NOT NULL
        );
        CREATE TABLE source_lines (
            source_id TEXT NOT NULL,
            source_line INTEGER NOT NULL,
            PRIMARY KEY (source_id, source_line)
        );
        CREATE TABLE provenance_records (
            record_id TEXT PRIMARY KEY
        );
        """
    )


def verify_dataset_release(release_path: str | Path) -> dict[str, Any]:
    """Verify manifest, checksums, split records, provenance, and split isolation.

    Record IDs, content fingerprints, source lines, and group assignments are
    indexed in a temporary SQLite database so verification does not require
    keeping the complete dataset index in Python memory.
    """
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
        if not isinstance(source.get("name"), str) or not source["name"]:
            raise DatasetVerificationError(f"source {source_id}: source name is missing")
        if not isinstance(source.get("license_id"), str) or not source["license_id"].strip():
            raise DatasetVerificationError(f"source {source_id}: license_id is missing")
        if type(source.get("records_read")) is not int or source["records_read"] < 0:
            raise DatasetVerificationError(f"source {source_id}: invalid records_read")
        sources[source_id] = source

    expected_licenses = sorted({source["license_id"] for source in sources.values()})
    if manifest.get("licenses") != expected_licenses:
        raise DatasetVerificationError("manifest license list does not match source declarations")
    records_meta = manifest.get("records")
    if not isinstance(records_meta, dict):
        raise DatasetVerificationError("manifest records metadata is missing")

    split_counts = {"train": 0, "validation": 0, "test": 0}
    provenance_count = 0
    raw_records = sum(source["records_read"] for source in sources.values())
    temporary_directory = tempfile.TemporaryDirectory(prefix="rusamodels-verify-")
    try:
        connection = sqlite3.connect(Path(temporary_directory.name) / "verify.sqlite3")
        try:
            _initialize_verification_db(connection)
            for split in ("train", "validation", "test"):
                path = _safe_release_file(root, f"{split}.jsonl")
                with path.open("r", encoding="utf-8") as handle:
                    for line_number, line in enumerate(handle, start=1):
                        if not line.strip():
                            raise DatasetVerificationError(
                                f"{path.name}:{line_number}: blank lines are not allowed"
                            )
                        record = _load_json(line, f"{path.name}:{line_number}")
                        if not isinstance(record, dict):
                            raise DatasetVerificationError(
                                f"{path.name}:{line_number}: record must be an object"
                            )
                        record_id = record.get("id")
                        if not isinstance(record_id, str) or not record_id.strip():
                            raise DatasetVerificationError(
                                f"{path.name}:{line_number}: record id is missing"
                            )
                        try:
                            fingerprint = _fingerprint(record)
                        except (TypeError, ValueError) as exc:
                            raise DatasetVerificationError(
                                f"{path.name}:{line_number}: record cannot be fingerprinted"
                            ) from exc
                        try:
                            connection.execute(
                                "INSERT INTO records(record_id, split, fingerprint) VALUES (?, ?, ?)",
                                (record_id, split, fingerprint),
                            )
                        except sqlite3.IntegrityError as exc:
                            duplicate_id = connection.execute(
                                "SELECT 1 FROM records WHERE record_id = ?", (record_id,)
                            ).fetchone()
                            if duplicate_id:
                                raise DatasetVerificationError(
                                    f"record id occurs more than once: {record_id}"
                                ) from exc
                            raise DatasetVerificationError(
                                "duplicate content crosses or repeats dataset splits"
                            ) from exc
                        split_counts[split] += 1
                if split_counts[split] == 0:
                    raise DatasetVerificationError(f"{split} split is empty")
                if files[f"{split}.jsonl"].get("records") != split_counts[split]:
                    raise DatasetVerificationError(
                        f"{split}.jsonl record count does not match manifest"
                    )
            connection.commit()

            provenance_path = _safe_release_file(root, "provenance.jsonl")
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
                        or record_id == ""
                        or source_id not in sources
                    ):
                        raise DatasetVerificationError(
                            f"provenance.jsonl:{line_number}: unknown record or source"
                        )
                    if type(source_line) is not int or source_line < 1:
                        raise DatasetVerificationError(
                            f"provenance.jsonl:{line_number}: source_line must be positive"
                        )
                    source = sources[source_id]
                    if item.get("source_name") != source["name"]:
                        raise DatasetVerificationError("source name mismatch in provenance")
                    if item.get("source_sha256") != source["sha256"]:
                        raise DatasetVerificationError("source checksum mismatch in provenance")
                    if item.get("license_id") != source["license_id"]:
                        raise DatasetVerificationError("source license mismatch in provenance")
                    record = connection.execute(
                        "SELECT split, fingerprint FROM records WHERE record_id = ?",
                        (record_id,),
                    ).fetchone()
                    if record is None:
                        raise DatasetVerificationError(
                            f"provenance references unknown record: {record_id}"
                        )
                    if item.get("record_fingerprint") != record[1]:
                        raise DatasetVerificationError("record fingerprint mismatch in provenance")
                    if item.get("split") != record[0]:
                        raise DatasetVerificationError("provenance split does not match record split")
                    group_id = item.get("group_id")
                    if not isinstance(group_id, str) or not group_id:
                        raise DatasetVerificationError("provenance group_id is missing")
                    existing_group = connection.execute(
                        "SELECT split FROM group_splits WHERE group_id = ?", (group_id,)
                    ).fetchone()
                    if existing_group and existing_group[0] != record[0]:
                        raise DatasetVerificationError(
                            "a dataset group crosses train/validation/test splits"
                        )
                    connection.execute(
                        "INSERT OR IGNORE INTO group_splits(group_id, split) VALUES (?, ?)",
                        (group_id, record[0]),
                    )
                    existing_record_group = connection.execute(
                        "SELECT group_id FROM record_groups WHERE record_id = ?",
                        (record_id,),
                    ).fetchone()
                    if existing_record_group and existing_record_group[0] != group_id:
                        raise DatasetVerificationError(
                            "one deduplicated record has conflicting group provenance"
                        )
                    connection.execute(
                        "INSERT OR IGNORE INTO record_groups(record_id, group_id) VALUES (?, ?)",
                        (record_id, group_id),
                    )
                    try:
                        connection.execute(
                            "INSERT INTO source_lines(source_id, source_line) VALUES (?, ?)",
                            (source_id, source_line),
                        )
                    except sqlite3.IntegrityError as exc:
                        raise DatasetVerificationError(
                            "duplicate source line in provenance"
                        ) from exc
                    connection.execute(
                        "INSERT OR IGNORE INTO provenance_records(record_id) VALUES (?)",
                        (record_id,),
                    )
                    provenance_count += 1
            connection.commit()

            unique_records = connection.execute(
                "SELECT COUNT(*) FROM records"
            ).fetchone()[0]
            records_with_provenance = connection.execute(
                "SELECT COUNT(*) FROM provenance_records"
            ).fetchone()[0]
            if records_with_provenance != unique_records:
                raise DatasetVerificationError(
                    "one or more released records have no provenance"
                )
            if connection.execute(
                "SELECT COUNT(*) FROM record_groups"
            ).fetchone()[0] != unique_records:
                raise DatasetVerificationError(
                    "one or more released records have inconsistent group provenance"
                )
            source_line_counts = dict(connection.execute(
                "SELECT source_id, COUNT(*) FROM source_lines GROUP BY source_id"
            ).fetchall())
            for source_id, source in sources.items():
                if source_line_counts.get(source_id, 0) != source["records_read"]:
                    raise DatasetVerificationError(
                        f"source {source_id}: provenance count does not match records_read"
                    )
        finally:
            connection.close()
    except sqlite3.Error as exc:
        raise DatasetVerificationError(f"dataset verification database error: {exc}") from exc
    finally:
        temporary_directory.cleanup()

    if provenance_count != raw_records:
        raise DatasetVerificationError("provenance count does not match raw record count")
    unique_records = sum(split_counts.values())
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
    except (DatasetVerificationError, OSError, UnicodeError, sqlite3.Error) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
