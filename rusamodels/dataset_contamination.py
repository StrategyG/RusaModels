"""Check dataset splits against an immutable, separately maintained evaluation set."""
from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
import re
import sqlite3
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

from rusamodels.dataset_quality import (
    DatasetQualityError,
    simhash64_text,
    simhash_bands,
)
from rusamodels.dataset_text_export import (
    DatasetTextExportError,
    record_to_text,
)
from rusamodels.dataset_verification import (
    DatasetVerificationError,
    verify_dataset_release,
)


class DatasetContaminationError(ValueError):
    """Raised when a frozen evaluation set or contamination scan is invalid."""


_MAX_HAMMING_DISTANCE = 7
_BANDS = 8
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_DATASET_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant is prohibited: {value}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
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
            parse_constant=_reject_constant,
            object_pairs_hook=_unique_object,
        )
    except (json.JSONDecodeError, ValueError) as exc:
        raise DatasetContaminationError(f"{source}: invalid JSON: {exc}") from exc


def _hash_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def _fingerprint_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _pseudonymize_frozen_id(value: str) -> str:
    digest = hashlib.sha256(value.encode()).hexdigest()[:24]
    return f"frozen_{digest}"


def _validate_frozen_manifest(
    evaluation_path: Path,
    manifest_path: Path,
) -> tuple[dict[str, Any], str, str]:
    if evaluation_path.is_symlink() or not evaluation_path.is_file():
        raise DatasetContaminationError("frozen evaluation file is missing or unsafe")
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise DatasetContaminationError("frozen evaluation manifest is missing or unsafe")
    manifest_bytes = manifest_path.read_bytes()
    manifest_digest = hashlib.sha256(manifest_bytes).hexdigest()
    manifest = _load_json(manifest_bytes.decode("utf-8"), manifest_path.name)
    expected_keys = {
        "schema_version",
        "purpose",
        "dataset_id",
        "dataset_version",
        "locked",
        "file",
        "sha256",
        "size_bytes",
        "records",
        "created_at",
    }
    if not isinstance(manifest, dict) or set(manifest) != expected_keys:
        raise DatasetContaminationError("frozen evaluation manifest has unsupported or missing fields")
    if manifest["schema_version"] != "1.0.0" or manifest["purpose"] != "frozen_evaluation":
        raise DatasetContaminationError("manifest is not a supported frozen evaluation manifest")
    if manifest["locked"] is not True:
        raise DatasetContaminationError("frozen evaluation manifest must declare locked=true")
    if (
        not isinstance(manifest["dataset_id"], str)
        or not _DATASET_ID.fullmatch(manifest["dataset_id"])
        or not isinstance(manifest["dataset_version"], str)
        or not _DATASET_ID.fullmatch(manifest["dataset_version"])
    ):
        raise DatasetContaminationError("frozen evaluation dataset identity is invalid")
    if manifest["file"] != evaluation_path.name or Path(manifest["file"]).name != manifest["file"]:
        raise DatasetContaminationError("manifest file name does not match the frozen evaluation file")
    if not isinstance(manifest["sha256"], str) or not _SHA256.fullmatch(manifest["sha256"]):
        raise DatasetContaminationError("frozen evaluation SHA-256 is invalid")
    if type(manifest["size_bytes"]) is not int or manifest["size_bytes"] < 1:
        raise DatasetContaminationError("frozen evaluation size_bytes is invalid")
    if type(manifest["records"]) is not int or manifest["records"] < 1:
        raise DatasetContaminationError("frozen evaluation records must be a positive integer")
    try:
        created_at = datetime.fromisoformat(manifest["created_at"].replace("Z", "+00:00"))
    except (AttributeError, TypeError, ValueError) as exc:
        raise DatasetContaminationError("frozen evaluation created_at must be ISO-8601") from exc
    if created_at.tzinfo is None or created_at.utcoffset() is None:
        raise DatasetContaminationError("frozen evaluation created_at must include a timezone")
    actual_sha256, actual_size = _hash_file(evaluation_path)
    if not hmac.compare_digest(actual_sha256, manifest["sha256"]):
        raise DatasetContaminationError("frozen evaluation file SHA-256 does not match its manifest")
    if actual_size != manifest["size_bytes"]:
        raise DatasetContaminationError("frozen evaluation file size does not match its manifest")
    return manifest, manifest_digest, actual_sha256


def _initialize(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        CREATE TABLE frozen_records (
            record_id TEXT PRIMARY KEY,
            fingerprint TEXT NOT NULL UNIQUE,
            simhash_hex TEXT NOT NULL
        );
        CREATE TABLE buckets (
            band_index INTEGER NOT NULL,
            band_value INTEGER NOT NULL,
            record_id TEXT NOT NULL,
            PRIMARY KEY (band_index, band_value, record_id)
        );
        CREATE INDEX ix_frozen_buckets_lookup ON buckets(band_index, band_value);
        """
    )


def _write_new_file(path: Path, payload: bytes) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def check_frozen_evaluation_contamination(
    release_path: str | Path,
    frozen_evaluation_path: str | Path,
    frozen_manifest_path: str | Path,
    report_path: str | Path,
    *,
    max_hamming_distance: int = 7,
    max_candidates_per_record: int = 10000,
) -> dict[str, Any]:
    """Compare all released splits with a separately frozen evaluation JSONL.

    Exact text matches and SimHash near-duplicates are reported without
    copying evaluation text into the report. The report is written outside
    the immutable release directory.
    """
    if type(max_hamming_distance) is not int or not 0 <= max_hamming_distance <= _MAX_HAMMING_DISTANCE:
        raise DatasetContaminationError(
            f"max_hamming_distance must be between 0 and {_MAX_HAMMING_DISTANCE}"
        )
    if type(max_candidates_per_record) is not int or max_candidates_per_record < 1:
        raise DatasetContaminationError("max_candidates_per_record must be a positive integer")

    root = Path(release_path).expanduser().resolve()
    evaluation_input = Path(frozen_evaluation_path).expanduser()
    manifest_input = Path(frozen_manifest_path).expanduser()
    output_input = Path(report_path).expanduser()
    if evaluation_input.is_symlink() or manifest_input.is_symlink() or output_input.is_symlink():
        raise DatasetContaminationError("symlink paths are not allowed for frozen inputs or reports")
    evaluation = evaluation_input.parent.resolve() / evaluation_input.name
    frozen_manifest_path = manifest_input.parent.resolve() / manifest_input.name
    output = output_input.parent.resolve() / output_input.name
    summary_path = output.with_suffix(output.suffix + ".summary.json")
    for path, label in (
        (evaluation, "frozen evaluation file"),
        (frozen_manifest_path, "frozen evaluation manifest"),
        (output, "contamination report"),
        (summary_path, "contamination summary"),
    ):
        if root == path or root in path.parents:
            raise DatasetContaminationError(f"{label} must be outside the immutable dataset release")
    if output.exists() or summary_path.exists():
        raise DatasetContaminationError("contamination report or summary already exists")
    output.parent.mkdir(parents=True, exist_ok=True)

    try:
        verification = verify_dataset_release(root)
    except (DatasetVerificationError, OSError, UnicodeError, sqlite3.Error) as exc:
        raise DatasetContaminationError(f"dataset release verification failed: {exc}") from exc
    frozen_manifest, frozen_manifest_sha256, frozen_sha256 = _validate_frozen_manifest(
        evaluation, frozen_manifest_path
    )

    manifest_sha_before, _ = _hash_file(root / "manifest.json")
    release_manifest = _load_json((root / "manifest.json").read_text(encoding="utf-8"), "manifest.json")
    split_digests = {
        split: release_manifest["files"][f"{split}.jsonl"]["sha256"]
        for split in ("train", "validation", "test")
    }

    temporary_directory = tempfile.TemporaryDirectory(prefix="rusamodels-contamination-")
    report_temporary: Path | None = None
    report_published = False
    summary_published = False
    try:
        connection = sqlite3.connect(Path(temporary_directory.name) / "frozen.sqlite3")
        try:
            _initialize(connection)
            frozen_records = 0
            with evaluation.open("r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    if not line.strip():
                        raise DatasetContaminationError(
                            f"{evaluation.name}:{line_number}: blank lines are not allowed"
                        )
                    record = _load_json(line, f"{evaluation.name}:{line_number}")
                    if not isinstance(record, dict):
                        raise DatasetContaminationError(
                            f"{evaluation.name}:{line_number}: record must be an object"
                        )
                    record_id = record.get("id", record.get("record_id"))
                    if not isinstance(record_id, str) or not record_id.strip() or len(record_id) > 256:
                        raise DatasetContaminationError(
                            f"{evaluation.name}:{line_number}: record id is missing or invalid"
                        )
                    try:
                        text = record_to_text(record)
                        fingerprint = _fingerprint_text(text)
                        simhash = simhash64_text(text)
                        bands = simhash_bands(simhash)
                    except (DatasetTextExportError, DatasetQualityError) as exc:
                        raise DatasetContaminationError(
                            f"{evaluation.name}:{line_number}: {exc}"
                        ) from exc
                    try:
                        connection.execute(
                            "INSERT INTO frozen_records(record_id, fingerprint, simhash_hex) VALUES (?, ?, ?)",
                            (record_id, fingerprint, f"{simhash:016x}"),
                        )
                    except sqlite3.IntegrityError as exc:
                        raise DatasetContaminationError(
                            f"{evaluation.name}:{line_number}: duplicate frozen record ID or exact content"
                        ) from exc
                    connection.executemany(
                        "INSERT INTO buckets(band_index, band_value, record_id) VALUES (?, ?, ?)",
                        [
                            (index, band_value, record_id)
                            for index, band_value in enumerate(bands)
                        ],
                    )
                    frozen_records += 1
            if frozen_records != frozen_manifest["records"]:
                raise DatasetContaminationError(
                    "frozen evaluation record count does not match its manifest"
                )
            connection.commit()

            clauses = " OR ".join("(band_index = ? AND band_value = ?)" for _ in range(_BANDS))
            candidate_comparisons = 0
            records_scanned = 0
            exact_matches = 0
            near_matches = 0
            split_counts = {"train": 0, "validation": 0, "test": 0}
            report_descriptor, report_name = tempfile.mkstemp(
                prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
            )
            report_temporary = Path(report_name)
            report_digest = hashlib.sha256()
            with os.fdopen(report_descriptor, "wb") as report_handle:
                for split in ("train", "validation", "test"):
                    split_path = root / f"{split}.jsonl"
                    with split_path.open("r", encoding="utf-8") as handle:
                        for line_number, line in enumerate(handle, start=1):
                            if not line.strip():
                                raise DatasetContaminationError(
                                    f"{split_path.name}:{line_number}: blank lines are not allowed"
                                )
                            record = _load_json(line, f"{split_path.name}:{line_number}")
                            if not isinstance(record, dict):
                                raise DatasetContaminationError(
                                    f"{split_path.name}:{line_number}: record must be an object"
                                )
                            record_id = record.get("id")
                            if not isinstance(record_id, str):
                                raise DatasetContaminationError(
                                    f"{split_path.name}:{line_number}: record ID is missing"
                                )
                            try:
                                text = record_to_text(record)
                                fingerprint = _fingerprint_text(text)
                                simhash = simhash64_text(text)
                                bands = simhash_bands(simhash)
                            except (DatasetTextExportError, DatasetQualityError) as exc:
                                raise DatasetContaminationError(
                                    f"{split_path.name}:{line_number}: {exc}"
                                ) from exc
                            parameters = tuple(
                                value
                                for index, band_value in enumerate(bands)
                                for value in (index, band_value)
                            )
                            candidates = [
                                row[0]
                                for row in connection.execute(
                                    f"SELECT DISTINCT record_id FROM buckets WHERE {clauses}",
                                    parameters,
                                )
                            ]
                            if len(candidates) > max_candidates_per_record:
                                raise DatasetContaminationError(
                                    f"{split_path.name}:{line_number}: candidate set exceeds "
                                    f"max_candidates_per_record={max_candidates_per_record}"
                                )
                            for candidate_id in candidates:
                                candidate = connection.execute(
                                    "SELECT fingerprint, simhash_hex FROM frozen_records WHERE record_id = ?",
                                    (candidate_id,),
                                ).fetchone()
                                if candidate is None:
                                    raise DatasetContaminationError("frozen LSH index is inconsistent")
                                candidate_comparisons += 1
                                candidate_fingerprint, candidate_hex = candidate
                                candidate_simhash = int(candidate_hex, 16)
                                if fingerprint == candidate_fingerprint:
                                    match_type = "exact"
                                    distance = 0
                                    exact_matches += 1
                                else:
                                    distance = (simhash ^ candidate_simhash).bit_count()
                                    if distance > max_hamming_distance:
                                        continue
                                    match_type = "near_duplicate"
                                    near_matches += 1
                                item = {
                                    "dataset_record_id": record_id,
                                    "dataset_split": split,
                                    "frozen_eval_record_id": _pseudonymize_frozen_id(candidate_id),
                                    "match_type": match_type,
                                    "hamming_distance": distance,
                                    "text_sha256": fingerprint,
                                }
                                encoded = (json.dumps(item, sort_keys=True) + "\n").encode()
                                report_handle.write(encoded)
                                report_digest.update(encoded)
                            records_scanned += 1
                            split_counts[split] += 1
                report_handle.flush()
                os.fsync(report_handle.fileno())
            os.link(report_temporary, output)
            report_published = True
            report_temporary.unlink(missing_ok=True)
            report_temporary = None

            for split in ("train", "validation", "test"):
                digest_after, size_after = _hash_file(root / f"{split}.jsonl")
                expected = release_manifest["files"][f"{split}.jsonl"]
                if digest_after != split_digests[split] or size_after != expected["size_bytes"]:
                    raise DatasetContaminationError(
                        f"{split}.jsonl changed while contamination analysis was running"
                    )
            manifest_sha_after, _ = _hash_file(root / "manifest.json")
            if manifest_sha_after != manifest_sha_before or manifest_sha_after != verification["manifest_sha256"]:
                raise DatasetContaminationError("release manifest changed during contamination analysis")
            frozen_sha_after, frozen_size_after = _hash_file(evaluation)
            frozen_manifest_sha_after, _ = _hash_file(frozen_manifest_path)
            if (
                frozen_sha_after != frozen_sha256
                or frozen_size_after != frozen_manifest["size_bytes"]
                or frozen_manifest_sha_after != frozen_manifest_sha256
            ):
                raise DatasetContaminationError("frozen evaluation set changed during contamination analysis")

            report_sha256 = report_digest.hexdigest()
            summary = {
                "schema_version": "1.0.0",
                "report_type": "frozen_evaluation_contamination",
                "dataset_id": verification["dataset_id"],
                "dataset_version": verification["dataset_version"],
                "release_manifest_sha256": verification["manifest_sha256"],
                "frozen_evaluation_id": frozen_manifest["dataset_id"],
                "frozen_evaluation_version": frozen_manifest["dataset_version"],
                "frozen_evaluation_manifest_sha256": frozen_manifest_sha256,
                "frozen_evaluation_sha256": frozen_sha256,
                "frozen_evaluation_records": frozen_records,
                "records_scanned": records_scanned,
                "split_counts": split_counts,
                "candidate_comparisons": candidate_comparisons,
                "exact_matches": exact_matches,
                "near_duplicate_matches": near_matches,
                "total_matches": exact_matches + near_matches,
                "max_hamming_distance": max_hamming_distance,
                "algorithm": "text_export_sha256_plus_simhash64_word_trigram_lsh_8x8",
                "report_file": output.name,
                "report_sha256": report_sha256,
                "contamination_free": exact_matches == 0 and near_matches == 0,
            }
            summary_bytes = (
                json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
            ).encode()
            _write_new_file(summary_path, summary_bytes)
            summary_published = True
        finally:
            connection.close()
    except (OSError, sqlite3.Error, UnicodeError) as exc:
        raise DatasetContaminationError(f"contamination analysis failed: {exc}") from exc
    finally:
        temporary_directory.cleanup()
        if report_temporary is not None:
            report_temporary.unlink(missing_ok=True)
        if report_published and not summary_published:
            output.unlink(missing_ok=True)

    return {
        "dataset_id": verification["dataset_id"],
        "dataset_version": verification["dataset_version"],
        "records_scanned": records_scanned,
        "frozen_evaluation_records": frozen_records,
        "exact_matches": exact_matches,
        "near_duplicate_matches": near_matches,
        "total_matches": exact_matches + near_matches,
        "contamination_free": exact_matches == 0 and near_matches == 0,
        "report_path": str(output),
        "summary_path": str(summary_path),
        "report_sha256": report_sha256,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", required=True, type=Path)
    parser.add_argument("--frozen-evaluation", required=True, type=Path)
    parser.add_argument("--frozen-manifest", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--max-hamming-distance", type=int, default=7)
    parser.add_argument("--max-candidates-per-record", type=int, default=10000)
    args = parser.parse_args(argv)
    try:
        result = check_frozen_evaluation_contamination(
            args.release,
            args.frozen_evaluation,
            args.frozen_manifest,
            args.report,
            max_hamming_distance=args.max_hamming_distance,
            max_candidates_per_record=args.max_candidates_per_record,
        )
    except (DatasetContaminationError, OSError, UnicodeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["contamination_free"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
