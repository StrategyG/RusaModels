"""Reproducible, provenance-aware JSONL dataset release pipeline.

The v1 pipeline is deliberately dependency-light and disk-backed. It performs
strict JSONL ingestion, Unicode normalization, configurable redaction, exact
deduplication, deterministic group-aware splitting, checksums, and atomic
release publication. It does not claim semantic deduplication or automatic
license/PII classification.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sqlite3
import sys
import tempfile
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable


PIPELINE_VERSION = "0.1.0"
_DATASET_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
_SOURCE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
_VERSION = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
_EMAIL = re.compile(r"(?<![\w.+-])[\w.!#$%&'*+/=?^_|~-]+@[\w-]+(?:\.[\w-]+)+")
_SECRET = re.compile(
    r"(?i)\b(?:github_pat_[A-Za-z0-9_]{20,}|gh[pousr]_[A-Za-z0-9]{20,}|"
    r"sk-[A-Za-z0-9_-]{20,}|AKIA[0-9A-Z]{16})\b"
)
_BEARER = re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/-]{16,}=*")
_PHONE = re.compile(r"(?<!\w)(?:\+?\d[\d\s().-]{6,}\d)(?!\w)")


class DatasetPipelineError(ValueError):
    """Raised when a dataset source or release violates pipeline policy."""


@dataclass(frozen=True)
class DatasetSource:
    source_id: str
    path: Path
    license_id: str
    source_uri: str | None = None


@dataclass(frozen=True)
class RedactionPolicy:
    emails: bool = True
    secrets: bool = True
    phone_numbers: bool = False


@dataclass(frozen=True)
class DatasetReleaseConfig:
    dataset_id: str
    version: str
    sources: tuple[DatasetSource, ...]
    output_root: Path
    seed: int = 42
    validation_fraction: float = 0.10
    test_fraction: float = 0.10
    max_record_bytes: int = 2 * 1024 * 1024
    redaction: RedactionPolicy = field(default_factory=RedactionPolicy)


def _canonical_json(value: Any) -> str:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise DatasetPipelineError(f"record is not canonical JSON: {exc}") from exc


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant is prohibited: {value}")


def _object_pairs_without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def _normalize(value: Any) -> Any:
    if isinstance(value, str):
        normalized = unicodedata.normalize("NFC", value).replace("\r\n", "\n").replace("\r", "\n")
        if "\x00" in normalized:
            raise DatasetPipelineError("record contains a NUL character")
        return normalized
    if isinstance(value, list):
        return [_normalize(item) for item in value]
    if isinstance(value, dict):
        normalized: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise DatasetPipelineError("JSON object keys must be strings")
            normalized_key = unicodedata.normalize("NFC", key)
            if normalized_key in normalized:
                raise DatasetPipelineError("Unicode normalization produced duplicate object keys")
            normalized[normalized_key] = _normalize(item)
        return normalized
    if value is None or type(value) in {bool, int, float}:
        if isinstance(value, float) and not (float("-inf") < value < float("inf")):
            raise DatasetPipelineError("non-finite numeric values are prohibited")
        return value
    raise DatasetPipelineError(f"unsupported JSON value type: {type(value).__name__}")


def _redact_text(value: str, policy: RedactionPolicy, counts: Counter[str]) -> str:
    if policy.emails:
        value, count = _EMAIL.subn("[REDACTED_EMAIL]", value)
        counts["emails"] += count
    if policy.secrets:
        value, secret_count = _SECRET.subn("[REDACTED_SECRET]", value)
        value, bearer_count = _BEARER.subn("[REDACTED_SECRET]", value)
        counts["secrets"] += secret_count + bearer_count
    if policy.phone_numbers:
        def replace_phone(match: re.Match[str]) -> str:
            digits = sum(character.isdigit() for character in match.group(0))
            if 8 <= digits <= 15:
                counts["phone_numbers"] += 1
                return "[REDACTED_PHONE]"
            return match.group(0)
        value = _PHONE.sub(replace_phone, value)
    return value


def _redact(value: Any, policy: RedactionPolicy, counts: Counter[str]) -> Any:
    if isinstance(value, str):
        return _redact_text(value, policy, counts)
    if isinstance(value, list):
        return [_redact(item, policy, counts) for item in value]
    if isinstance(value, dict):
        return {key: _redact(item, policy, counts) for key, item in value.items()}
    return value


def _has_text_content(value: Any) -> bool:
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, list):
        return any(_has_text_content(item) for item in value)
    if isinstance(value, dict):
        return any(
            _has_text_content(item)
            for key, item in value.items()
            if key not in {"id", "group_id"}
        )
    return False


def _record_fingerprint(record: dict[str, Any]) -> str:
    content = {key: value for key, value in record.items() if key not in {"id", "group_id"}}
    return hashlib.sha256(_canonical_json(content).encode("utf-8")).hexdigest()


def _validate_config(config: DatasetReleaseConfig) -> None:
    if not isinstance(config.dataset_id, str) or not _DATASET_ID.fullmatch(config.dataset_id) or config.dataset_id in {".", ".."}:
        raise DatasetPipelineError("dataset_id must be a safe 1-64 character identifier")
    if not isinstance(config.version, str) or not _VERSION.fullmatch(config.version) or config.version in {".", ".."}:
        raise DatasetPipelineError("version must be a safe 1-64 character identifier")
    if not config.sources:
        raise DatasetPipelineError("at least one dataset source is required")
    if not isinstance(config.output_root, (str, Path)):
        raise DatasetPipelineError("output_root must be a string or Path")
    if type(config.seed) is not int:
        raise DatasetPipelineError("seed must be an integer")
    if type(config.validation_fraction) not in {int, float} or type(config.test_fraction) not in {int, float}:
        raise DatasetPipelineError("split fractions must be numbers")
    if not isinstance(config.redaction, RedactionPolicy) or any(
        type(value) is not bool for value in (
            config.redaction.emails, config.redaction.secrets, config.redaction.phone_numbers
        )
    ):
        raise DatasetPipelineError("redaction policy values must be booleans")
    if not (0 < config.validation_fraction < 1):
        raise DatasetPipelineError("validation_fraction must be between 0 and 1")
    if not (0 < config.test_fraction < 1):
        raise DatasetPipelineError("test_fraction must be between 0 and 1")
    if config.validation_fraction + config.test_fraction >= 1:
        raise DatasetPipelineError("validation_fraction + test_fraction must be less than 1")
    if type(config.max_record_bytes) is not int or config.max_record_bytes < 1:
        raise DatasetPipelineError("max_record_bytes must be a positive integer")
    source_ids: set[str] = set()
    for source in config.sources:
        if not isinstance(source, DatasetSource):
            raise DatasetPipelineError("sources must contain DatasetSource values")
        if not isinstance(source.path, (str, Path)):
            raise DatasetPipelineError(f"source {source.source_id!r} path must be a string or Path")
        if not isinstance(source.source_id, str) or not _SOURCE_ID.fullmatch(source.source_id):
            raise DatasetPipelineError(f"invalid source_id: {source.source_id!r}")
        if source.source_id in source_ids:
            raise DatasetPipelineError(f"duplicate source_id: {source.source_id}")
        source_ids.add(source.source_id)
        if not isinstance(source.license_id, str) or not source.license_id.strip():
            raise DatasetPipelineError(f"source {source.source_id} requires a declared license_id")
        if source.source_uri is not None:
            if not isinstance(source.source_uri, str) or len(source.source_uri) > 2048:
                raise DatasetPipelineError(f"source_uri for {source.source_id} is invalid or too long")
            from urllib.parse import urlsplit
            try:
                parsed = urlsplit(source.source_uri)
            except ValueError as exc:
                raise DatasetPipelineError(
                    f"source_uri for {source.source_id} is malformed"
                ) from exc
            if parsed.username or parsed.password or parsed.query or parsed.fragment:
                raise DatasetPipelineError(f"source_uri for {source.source_id} must not contain credentials, query, or fragment")


def _prepare_record(
    raw: Any,
    policy: RedactionPolicy,
    redaction_counts: Counter[str],
) -> tuple[str, str, str, str]:
    if not isinstance(raw, dict):
        raise DatasetPipelineError("each JSONL record must be a JSON object")
    record = _normalize(raw)
    record_id = record.get("id")
    if record_id is not None and (
        not isinstance(record_id, str) or not record_id.strip() or len(record_id) > 256
    ):
        raise DatasetPipelineError("record id must be a non-empty string of at most 256 characters")
    group_id = record.get("group_id")
    if group_id is not None and (
        not isinstance(group_id, str) or not group_id.strip() or len(group_id) > 256
    ):
        raise DatasetPipelineError("group_id must be a non-empty string of at most 256 characters")

    content = {key: value for key, value in record.items() if key not in {"id", "group_id"}}
    if not _has_text_content(content):
        raise DatasetPipelineError("record must contain at least one non-empty text value")
    content = _redact(content, policy, redaction_counts)
    fingerprint = hashlib.sha256(_canonical_json(content).encode("utf-8")).hexdigest()
    if record_id is None:
        record_id = f"record_{fingerprint[:24]}"
    payload = dict(content)
    payload["id"] = record_id
    group_key = group_id or f"content:{fingerprint}"
    return record_id, group_key, fingerprint, _canonical_json(payload)


def _initialize_staging(connection: sqlite3.Connection) -> None:
    connection.execute("PRAGMA foreign_keys = ON")
    connection.executescript(
        """
        CREATE TABLE sources (
            source_id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            source_uri TEXT,
            license_id TEXT NOT NULL,
            sha256 TEXT,
            size_bytes INTEGER NOT NULL DEFAULT 0,
            records_read INTEGER NOT NULL DEFAULT 0
        );
        CREATE TABLE records (
            record_id TEXT PRIMARY KEY,
            fingerprint TEXT NOT NULL UNIQUE,
            group_key TEXT NOT NULL,
            payload TEXT NOT NULL,
            sort_key TEXT NOT NULL,
            split TEXT
        );
        CREATE TABLE groups (
            group_key TEXT PRIMARY KEY,
            sort_key TEXT NOT NULL,
            split TEXT
        );
        CREATE TABLE provenance (
            record_id TEXT NOT NULL REFERENCES records(record_id),
            source_id TEXT NOT NULL REFERENCES sources(source_id),
            source_line INTEGER NOT NULL,
            fingerprint TEXT NOT NULL,
            PRIMARY KEY (source_id, source_line)
        );
        CREATE INDEX ix_records_split_sort ON records(split, sort_key);
        CREATE INDEX ix_provenance_record ON provenance(record_id);
        """
    )


def _ingest_source(
    connection: sqlite3.Connection,
    source: DatasetSource,
    config: DatasetReleaseConfig,
    redaction_counts: Counter[str],
) -> tuple[str, int, int]:
    path = Path(source.path).expanduser()
    if not path.is_file():
        raise DatasetPipelineError(f"dataset source is not a readable file: {path}")
    connection.execute(
        "INSERT INTO sources(source_id, name, source_uri, license_id) VALUES (?, ?, ?, ?)",
        (source.source_id, path.name, source.source_uri, source.license_id.strip()),
    )
    digest = hashlib.sha256()
    size_bytes = 0
    records_read = 0
    with path.open("rb") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            digest.update(raw_line)
            size_bytes += len(raw_line)
            if len(raw_line) > config.max_record_bytes:
                raise DatasetPipelineError(
                    f"{source.source_id}:{line_number}: record exceeds "
                    f"max_record_bytes={config.max_record_bytes}"
                )
            if not raw_line.strip():
                continue
            try:
                line = raw_line.decode("utf-8")
                raw_record = json.loads(
                    line,
                    parse_constant=_reject_json_constant,
                    object_pairs_hook=_object_pairs_without_duplicates,
                )
                record_id, group_key, fingerprint, payload = _prepare_record(
                    raw_record, config.redaction, redaction_counts
                )
            except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
                raise DatasetPipelineError(
                    f"{source.source_id}:{line_number}: {exc}"
                ) from exc
            existing_fingerprint = connection.execute(
                "SELECT record_id, group_key FROM records WHERE fingerprint = ?", (fingerprint,)
            ).fetchone()
            existing_id = connection.execute(
                "SELECT fingerprint FROM records WHERE record_id = ?", (record_id,)
            ).fetchone()
            if existing_id and existing_id[0] != fingerprint:
                raise DatasetPipelineError(
                    f"{source.source_id}:{line_number}: record id {record_id!r} "
                    "is reused for different content"
                )
            if existing_fingerprint:
                if existing_fingerprint[1] != group_key:
                    raise DatasetPipelineError(
                        f"{source.source_id}:{line_number}: exact duplicate content appears "
                        "under different group_id values; reconcile the group metadata before release"
                    )
                canonical_id = existing_fingerprint[0]
            else:
                sort_key = hashlib.sha256(
                    f"{config.seed}\0record\0{record_id}".encode("utf-8")
                ).hexdigest()
                connection.execute(
                    "INSERT INTO records(record_id, fingerprint, group_key, payload, sort_key) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (record_id, fingerprint, group_key, payload, sort_key),
                )
                group_sort_key = hashlib.sha256(
                    f"{config.seed}\0group\0{group_key}".encode("utf-8")
                ).hexdigest()
                connection.execute(
                    "INSERT OR IGNORE INTO groups(group_key, sort_key) VALUES (?, ?)",
                    (group_key, group_sort_key),
                )
                canonical_id = record_id
            connection.execute(
                "INSERT INTO provenance(record_id, source_id, source_line, fingerprint) "
                "VALUES (?, ?, ?, ?)",
                (canonical_id, source.source_id, line_number, fingerprint),
            )
            records_read += 1
    connection.execute(
        "UPDATE sources SET sha256 = ?, size_bytes = ?, records_read = ? WHERE source_id = ?",
        (digest.hexdigest(), size_bytes, records_read, source.source_id),
    )
    return digest.hexdigest(), size_bytes, records_read


def _assign_splits(connection: sqlite3.Connection, config: DatasetReleaseConfig) -> dict[str, int]:
    group_count = connection.execute("SELECT COUNT(*) FROM groups").fetchone()[0]
    if group_count < 3:
        raise DatasetPipelineError(
            "at least three distinct groups are required to create train, validation, and test splits"
        )
    test_count = max(1, round(group_count * config.test_fraction))
    validation_count = max(1, round(group_count * config.validation_fraction))
    if test_count + validation_count >= group_count:
        raise DatasetPipelineError(
            "split fractions leave no training groups; provide more groups or smaller fractions"
        )
    connection.execute(
        """
        WITH ranked_groups AS (
            SELECT group_key,
                   ROW_NUMBER() OVER (ORDER BY sort_key, group_key) AS position
            FROM groups
        )
        UPDATE groups
        SET split = (
            SELECT CASE
                WHEN position <= ? THEN 'test'
                WHEN position <= ? THEN 'validation'
                ELSE 'train'
            END
            FROM ranked_groups
            WHERE ranked_groups.group_key = groups.group_key
        )
        """,
        (test_count, test_count + validation_count),
    )
    connection.execute(
        "UPDATE records SET split = (SELECT split FROM groups WHERE groups.group_key = records.group_key)"
    )
    return {
        split: connection.execute(
            "SELECT COUNT(*) FROM records WHERE split = ?", (split,)
        ).fetchone()[0]
        for split in ("train", "validation", "test")
    }


def _write_split(
    connection: sqlite3.Connection,
    output_path: Path,
    split: str,
) -> dict[str, Any]:
    digest = hashlib.sha256()
    size_bytes = 0
    count = 0
    with output_path.open("wb") as handle:
        rows = connection.execute(
            "SELECT payload FROM records WHERE split = ? ORDER BY sort_key, record_id",
            (split,),
        )
        for (payload,) in rows:
            encoded = payload.encode("utf-8") + b"\n"
            handle.write(encoded)
            digest.update(encoded)
            size_bytes += len(encoded)
            count += 1
    return {"sha256": digest.hexdigest(), "size_bytes": size_bytes, "records": count}


def _write_provenance(connection: sqlite3.Connection, output_path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    size_bytes = 0
    count = 0
    with output_path.open("wb") as handle:
        rows = connection.execute(
            """
            SELECT p.record_id, p.source_id, s.name, s.license_id, s.sha256,
                   p.source_line, p.fingerprint, r.split, r.group_key
            FROM provenance p
            JOIN sources s ON s.source_id = p.source_id
            JOIN records r ON r.record_id = p.record_id
            ORDER BY p.source_id, p.source_line
            """
        )
        for row in rows:
            payload = {
                "record_id": row[0],
                "source_id": row[1],
                "source_name": row[2],
                "license_id": row[3],
                "source_sha256": row[4],
                "source_line": row[5],
                "record_fingerprint": row[6],
                "split": row[7],
                "group_id": row[8],
            }
            encoded = (_canonical_json(payload) + "\n").encode("utf-8")
            handle.write(encoded)
            digest.update(encoded)
            size_bytes += len(encoded)
            count += 1
    return {"sha256": digest.hexdigest(), "size_bytes": size_bytes, "records": count}


def build_dataset_release(config: DatasetReleaseConfig) -> Path:
    """Build and atomically publish an immutable dataset release."""
    _validate_config(config)
    root = Path(config.output_root).expanduser().resolve()
    dataset_dir = root / config.dataset_id
    final_dir = dataset_dir / config.version
    if root not in final_dir.resolve().parents:
        raise DatasetPipelineError("dataset release path escapes output_root")
    dataset_dir.mkdir(parents=True, exist_ok=True)
    lock_path = dataset_dir / f".{config.version}.lock"
    try:
        lock_fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise DatasetPipelineError(
            f"release build is already running or a stale lock exists: {lock_path.name}"
        ) from exc
    os.close(lock_fd)
    staging_dir: Path | None = None
    published = False
    connection: sqlite3.Connection | None = None
    try:
        if final_dir.exists():
            raise DatasetPipelineError(
                f"dataset release is immutable and already exists: {final_dir}"
            )
        staging_dir = Path(tempfile.mkdtemp(prefix=".dataset-release-", dir=dataset_dir))
        staging_db = staging_dir / "staging.sqlite3"
        connection = sqlite3.connect(staging_db)
        _initialize_staging(connection)
        redaction_counts: Counter[str] = Counter()
        source_summaries = []
        raw_records = 0
        for source in config.sources:
            sha256, size_bytes, records_read = _ingest_source(
                connection, source, config, redaction_counts
            )
            raw_records += records_read
            source_summaries.append({
                "source_id": source.source_id,
                "name": Path(source.path).name,
                "license_id": source.license_id.strip(),
                "source_uri": source.source_uri,
                "sha256": sha256,
                "size_bytes": size_bytes,
                "records_read": records_read,
            })
            connection.commit()
        split_counts = _assign_splits(connection, config)
        connection.commit()
        files = {
            f"{split}.jsonl": _write_split(
                connection, staging_dir / f"{split}.jsonl", split
            )
            for split in ("train", "validation", "test")
        }
        files["provenance.jsonl"] = _write_provenance(
            connection, staging_dir / "provenance.jsonl"
        )
        unique_records = connection.execute("SELECT COUNT(*) FROM records").fetchone()[0]
        connection.close()
        connection = None
        staging_db.unlink(missing_ok=True)
        manifest = {
            "schema_version": "1.0.0",
            "pipeline_version": PIPELINE_VERSION,
            "dataset_id": config.dataset_id,
            "dataset_version": config.version,
            "created_at": datetime.now(UTC).isoformat(),
            "sources": source_summaries,
            "licenses": sorted({source["license_id"] for source in source_summaries}),
            "records": {
                "raw_records": raw_records,
                "unique_records": unique_records,
                "duplicates_removed": raw_records - unique_records,
                "split_counts": split_counts,
            },
            "splitting": {
                "strategy": "deterministic_hash_ordered_groups",
                "group_field": "group_id",
                "seed": config.seed,
                "validation_fraction": config.validation_fraction,
                "test_fraction": config.test_fraction,
            },
            "normalization": {
                "unicode": "NFC",
                "newlines": "LF",
                "deduplication": "exact_canonical_json_excluding_id_and_group_id",
            },
            "redaction": {
                "policy": {
                    "emails": config.redaction.emails,
                    "secrets": config.redaction.secrets,
                    "phone_numbers": config.redaction.phone_numbers,
                },
                "counts": dict(sorted(redaction_counts.items())),
            },
            "files": files,
        }
        manifest_bytes = (json.dumps(
            manifest, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False
        ) + "\n").encode("utf-8")
        (staging_dir / "manifest.json").write_bytes(manifest_bytes)
        (staging_dir / "manifest.sha256").write_text(
            hashlib.sha256(manifest_bytes).hexdigest() + "\n", encoding="ascii"
        )
        os.rename(staging_dir, final_dir)
        published = True
        return final_dir
    except DatasetPipelineError:
        raise
    except (OSError, sqlite3.Error, UnicodeError, ValueError) as exc:
        raise DatasetPipelineError(f"dataset release failed: {exc}") from exc
    finally:
        if connection is not None:
            connection.close()
        if staging_dir is not None and not published:
            shutil.rmtree(staging_dir, ignore_errors=True)
        lock_path.unlink(missing_ok=True)


def load_release_config(config_path: str | Path) -> DatasetReleaseConfig:
    path = Path(config_path).expanduser().resolve()
    try:
        raw = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=_reject_json_constant,
            object_pairs_hook=_object_pairs_without_duplicates,
        )
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise DatasetPipelineError(f"cannot load dataset config {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise DatasetPipelineError("dataset config must be a JSON object")
    allowed = {
        "dataset_id", "version", "output_root", "sources", "seed",
        "validation_fraction", "test_fraction", "max_record_bytes", "redaction",
    }
    if set(raw) - allowed:
        raise DatasetPipelineError(f"dataset config has unsupported keys: {sorted(set(raw) - allowed)}")
    sources_raw = raw.get("sources")
    if not isinstance(sources_raw, list):
        raise DatasetPipelineError("dataset config sources must be an array")
    sources = []
    for item in sources_raw:
        if not isinstance(item, dict) or set(item) - {"source_id", "path", "license_id", "source_uri"}:
            raise DatasetPipelineError("each source requires source_id, path, license_id and optional source_uri")
        if not {"source_id", "path", "license_id"}.issubset(item):
            raise DatasetPipelineError("each source requires source_id, path, and license_id")
        if not isinstance(item["source_id"], str) or not isinstance(item["path"], str):
            raise DatasetPipelineError("source_id and path must be strings")
        if not isinstance(item["license_id"], str):
            raise DatasetPipelineError("source license_id must be a string")
        if item.get("source_uri") is not None and not isinstance(item["source_uri"], str):
            raise DatasetPipelineError("source_uri must be a string when provided")
        source_path = Path(item["path"]).expanduser()
        if not source_path.is_absolute():
            source_path = (path.parent / source_path).resolve()
        sources.append(DatasetSource(
            source_id=item["source_id"],
            path=source_path,
            license_id=item["license_id"],
            source_uri=item.get("source_uri"),
        ))
    redaction_raw = raw.get("redaction", {})
    if not isinstance(redaction_raw, dict) or set(redaction_raw) - {
        "emails", "secrets", "phone_numbers"
    }:
        raise DatasetPipelineError("redaction must contain only emails, secrets, and phone_numbers")
    for key, value in redaction_raw.items():
        if type(value) is not bool:
            raise DatasetPipelineError(f"redaction.{key} must be a boolean")
    output_root_value = raw.get("output_root", "artifacts/datasets")
    if not isinstance(output_root_value, str):
        raise DatasetPipelineError("output_root must be a string")
    output_root = Path(output_root_value).expanduser()
    if not output_root.is_absolute():
        output_root = (path.parent / output_root).resolve()
    return DatasetReleaseConfig(
        dataset_id=raw.get("dataset_id"),
        version=raw.get("version"),
        sources=tuple(sources),
        output_root=output_root,
        seed=raw.get("seed", 42),
        validation_fraction=raw.get("validation_fraction", 0.10),
        test_fraction=raw.get("test_fraction", 0.10),
        max_record_bytes=raw.get("max_record_bytes", 2 * 1024 * 1024),
        redaction=RedactionPolicy(
            emails=redaction_raw.get("emails", True),
            secrets=redaction_raw.get("secrets", True),
            phone_numbers=redaction_raw.get("phone_numbers", False),
        ),
    )


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path, help="Path to a JSON dataset release config")
    args = parser.parse_args(argv)
    try:
        release_path = build_dataset_release(load_release_config(args.config))
    except (ValueError, OSError, TypeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    manifest = json.loads((release_path / "manifest.json").read_text(encoding="utf-8"))
    print(json.dumps({
        "release_path": str(release_path),
        "dataset_id": manifest["dataset_id"],
        "dataset_version": manifest["dataset_version"],
        "records": manifest["records"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
