"""Validate human review evidence and gate dataset release promotion."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sqlite3
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

from rusamodels.dataset_verification import (
    DatasetVerificationError,
    verify_dataset_release,
)


class DatasetReviewError(ValueError):
    """Raised when a review ledger is malformed or fails release policy."""


_REQUIRED_REVIEW_TYPES = (
    "technical",
    "rights",
    "security",
    "independent_qa",
    "publication_approval",
)
_DECISIONS = {"approved", "rejected", "needs_revision"}
_RISK_ORDER = {"low": 0, "medium": 1, "high": 2, "critical": 3}
_EVIDENCE_STATUSES = {
    "source_assertion",
    "implementation_observation",
    "test_assertion",
    "proposal",
    "hypothesis",
    "independently_verified_operational_evidence",
    "synthetic",
}
_RECORD_ID = re.compile(r"^record_[0-9a-f]{24}$")
_REVIEW_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_REVIEWER_ID = re.compile(r"^reviewer_[A-Za-z0-9._-]{1,60}$")
_LANGUAGE = re.compile(r"^(?:[a-z]{2,3}|mixed|und)$")
_TASK_TYPE = re.compile(r"^[a-z][a-z0-9_.-]{0,63}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_REVIEW_KEYS = {
    "review_id",
    "release_manifest_sha256",
    "record_id",
    "review_type",
    "reviewer_id",
    "decision",
    "quality_score",
    "risk_class",
    "rights_status",
    "evidence_status",
    "language",
    "task_type",
    "publication_authorized",
    "reviewed_at",
}


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant is prohibited: {value}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def _parse_review(raw: str, source: str) -> dict[str, Any]:
    try:
        value = json.loads(
            raw,
            parse_constant=_reject_constant,
            object_pairs_hook=_unique_object,
        )
    except (json.JSONDecodeError, ValueError) as exc:
        raise DatasetReviewError(f"{source}: invalid JSON: {exc}") from exc
    if not isinstance(value, dict) or set(value) != _REVIEW_KEYS:
        raise DatasetReviewError(
            f"{source}: review event must contain exactly {sorted(_REVIEW_KEYS)}"
        )

    if not isinstance(value["review_id"], str) or not _REVIEW_ID.fullmatch(value["review_id"]):
        raise DatasetReviewError(f"{source}: review_id is invalid")
    if not isinstance(value["release_manifest_sha256"], str) or not _SHA256.fullmatch(
        value["release_manifest_sha256"]
    ):
        raise DatasetReviewError(f"{source}: release_manifest_sha256 must be a lowercase SHA-256 digest")
    if not isinstance(value["record_id"], str) or not _RECORD_ID.fullmatch(value["record_id"]):
        raise DatasetReviewError(f"{source}: record_id must be a pseudonymized dataset record ID")
    if not isinstance(value["review_type"], str) or value["review_type"] not in _REQUIRED_REVIEW_TYPES:
        raise DatasetReviewError(f"{source}: unsupported review_type")
    if not isinstance(value["reviewer_id"], str) or not _REVIEWER_ID.fullmatch(value["reviewer_id"]):
        raise DatasetReviewError(f"{source}: reviewer_id must be a pseudonymous reviewer identifier")
    if not isinstance(value["decision"], str) or value["decision"] not in _DECISIONS:
        raise DatasetReviewError(f"{source}: unsupported review decision")
    if value["quality_score"] is not None and (
        type(value["quality_score"]) is not int or not 1 <= value["quality_score"] <= 5
    ):
        raise DatasetReviewError(f"{source}: quality_score must be null or an integer from 1 to 5")
    if value["risk_class"] is not None and (
        not isinstance(value["risk_class"], str) or value["risk_class"] not in _RISK_ORDER
    ):
        raise DatasetReviewError(f"{source}: risk_class is invalid")
    if value["rights_status"] is not None and (
        not isinstance(value["rights_status"], str)
        or value["rights_status"] not in {"approved", "conditional", "rejected"}
    ):
        raise DatasetReviewError(f"{source}: rights_status is invalid")
    if value["evidence_status"] is not None and (
        not isinstance(value["evidence_status"], str)
        or value["evidence_status"] not in _EVIDENCE_STATUSES
    ):
        raise DatasetReviewError(f"{source}: evidence_status is invalid")
    if value["language"] is not None and (
        not isinstance(value["language"], str) or not _LANGUAGE.fullmatch(value["language"])
    ):
        raise DatasetReviewError(f"{source}: language must be an ISO 639-1/3 code, mixed, or und")
    if value["task_type"] is not None and (
        not isinstance(value["task_type"], str) or not _TASK_TYPE.fullmatch(value["task_type"])
    ):
        raise DatasetReviewError(f"{source}: task_type is invalid")
    if type(value["publication_authorized"]) is not bool:
        raise DatasetReviewError(f"{source}: publication_authorized must be a boolean")

    try:
        timestamp = datetime.fromisoformat(value["reviewed_at"].replace("Z", "+00:00"))
    except (AttributeError, TypeError, ValueError) as exc:
        raise DatasetReviewError(f"{source}: reviewed_at must be an ISO-8601 timestamp") from exc
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise DatasetReviewError(f"{source}: reviewed_at must include a timezone")
    value["_reviewed_epoch"] = timestamp.timestamp()

    review_type = value["review_type"]
    if review_type == "technical":
        if (
            value["quality_score"] is None
            or value["evidence_status"] is None
            or value["language"] is None
            or value["task_type"] is None
        ):
            raise DatasetReviewError(f"{source}: technical review requires quality, evidence, language, and task type")
    elif review_type == "rights":
        if value["rights_status"] is None:
            raise DatasetReviewError(f"{source}: rights review requires rights_status")
    elif review_type == "security":
        if value["risk_class"] is None:
            raise DatasetReviewError(f"{source}: security review requires risk_class")
    elif review_type == "independent_qa":
        if (
            value["quality_score"] is None
            or value["risk_class"] is None
            or value["evidence_status"] is None
        ):
            raise DatasetReviewError(f"{source}: independent QA requires quality, risk, and evidence")
    elif review_type == "publication_approval":
        if not value["publication_authorized"]:
            raise DatasetReviewError(f"{source}: publication approval must authorize publication")

    if review_type != "publication_approval" and value["publication_authorized"]:
        raise DatasetReviewError(f"{source}: only publication_approval may authorize publication")
    if review_type != "rights" and value["rights_status"] is not None:
        raise DatasetReviewError(f"{source}: rights_status is only valid for rights reviews")
    if review_type != "security" and review_type != "independent_qa" and value["risk_class"] is not None:
        raise DatasetReviewError(f"{source}: risk_class is only valid for security and independent QA reviews")
    if review_type not in {"technical", "independent_qa"} and value["quality_score"] is not None:
        raise DatasetReviewError(f"{source}: quality_score is only valid for technical and independent QA reviews")
    return value


def _initialize(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        CREATE TABLE release_records (
            record_id TEXT PRIMARY KEY
        );
        CREATE TABLE review_ids (
            review_id TEXT PRIMARY KEY
        );
        CREATE TABLE latest_reviews (
            record_id TEXT NOT NULL,
            review_type TEXT NOT NULL,
            reviewer_id TEXT NOT NULL,
            decision TEXT NOT NULL,
            quality_score INTEGER,
            risk_class TEXT,
            rights_status TEXT,
            publication_authorized INTEGER NOT NULL,
            reviewed_at TEXT NOT NULL,
            reviewed_epoch REAL NOT NULL,
            PRIMARY KEY (record_id, review_type)
        );
        """
    )


def _write_report(path: Path, report: dict[str, Any]) -> str:
    payload = (json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode()
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
    return hashlib.sha256(payload).hexdigest()


def evaluate_dataset_reviews(
    release_path: str | Path,
    review_ledger_path: str | Path,
    report_path: str | Path,
    *,
    minimum_quality_score: int = 4,
    maximum_risk_class: str = "medium",
) -> dict[str, Any]:
    """Evaluate append-only human review events against every released record.

    Every record requires technical, rights, security, independent QA, and
    publication approval. The independent QA reviewer must differ from the
    technical reviewer; the publication approver must differ from both.
    """
    if type(minimum_quality_score) is not int or not 1 <= minimum_quality_score <= 5:
        raise DatasetReviewError("minimum_quality_score must be an integer from 1 to 5")
    if not isinstance(maximum_risk_class, str) or maximum_risk_class not in _RISK_ORDER:
        raise DatasetReviewError("maximum_risk_class must be low, medium, high, or critical")

    root = Path(release_path).expanduser().resolve()
    ledger_input = Path(review_ledger_path).expanduser()
    output_input = Path(report_path).expanduser()
    if ledger_input.is_symlink() or output_input.is_symlink():
        raise DatasetReviewError("symlink paths are not allowed for review ledgers or reports")
    ledger = ledger_input.parent.resolve() / ledger_input.name
    output = output_input.parent.resolve() / output_input.name
    if root == output or root in output.parents:
        raise DatasetReviewError("review report must be outside the immutable dataset release")
    if not ledger.is_file():
        raise DatasetReviewError(f"review ledger does not exist: {ledger}")
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise DatasetReviewError(f"review report already exists: {output}")

    try:
        verification = verify_dataset_release(root)
    except (DatasetVerificationError, OSError, UnicodeError, sqlite3.Error) as exc:
        raise DatasetReviewError(f"dataset release verification failed: {exc}") from exc

    temporary_directory = tempfile.TemporaryDirectory(prefix="rusamodels-review-")
    ledger_digest = hashlib.sha256()
    ledger_events = 0
    blocker_counts: dict[str, int] = {}
    approved_records = 0
    reviewed_records = 0
    try:
        connection = sqlite3.connect(Path(temporary_directory.name) / "review.sqlite3")
        try:
            _initialize(connection)
            for split in ("train", "validation", "test"):
                with (root / f"{split}.jsonl").open("r", encoding="utf-8") as handle:
                    for line_number, line in enumerate(handle, start=1):
                        try:
                            record = json.loads(
                                line,
                                parse_constant=_reject_constant,
                                object_pairs_hook=_unique_object,
                            )
                        except (json.JSONDecodeError, ValueError) as exc:
                            raise DatasetReviewError(
                                f"{split}.jsonl:{line_number}: invalid JSON"
                            ) from exc
                        record_id = record.get("id") if isinstance(record, dict) else None
                        if not isinstance(record_id, str):
                            raise DatasetReviewError(
                                f"{split}.jsonl:{line_number}: record ID is missing"
                            )
                        connection.execute(
                            "INSERT INTO release_records(record_id) VALUES (?)", (record_id,)
                        )
            connection.commit()

            with ledger.open("rb") as handle:
                for line_number, raw_line in enumerate(handle, start=1):
                    ledger_digest.update(raw_line)
                    if not raw_line.strip():
                        raise DatasetReviewError(
                            f"{ledger.name}:{line_number}: blank review lines are not allowed"
                        )
                    try:
                        decoded = raw_line.decode("utf-8")
                    except UnicodeDecodeError as exc:
                        raise DatasetReviewError(
                            f"{ledger.name}:{line_number}: review ledger must be UTF-8"
                        ) from exc
                    event = _parse_review(decoded, f"{ledger.name}:{line_number}")
                    if event["release_manifest_sha256"] != verification["manifest_sha256"]:
                        raise DatasetReviewError(
                            f"{ledger.name}:{line_number}: review event release manifest SHA-256 "
                            "does not match the selected dataset release"
                        )
                    record_exists = connection.execute(
                        "SELECT 1 FROM release_records WHERE record_id = ?",
                        (event["record_id"],),
                    ).fetchone()
                    if record_exists is None:
                        raise DatasetReviewError(
                            f"{ledger.name}:{line_number}: review references an unknown record_id"
                        )
                    try:
                        connection.execute(
                            "INSERT INTO review_ids(review_id) VALUES (?)",
                            (event["review_id"],),
                        )
                    except sqlite3.IntegrityError as exc:
                        raise DatasetReviewError(
                            f"{ledger.name}:{line_number}: duplicate review_id"
                        ) from exc
                    existing = connection.execute(
                        """
                        SELECT reviewed_epoch FROM latest_reviews
                        WHERE record_id = ? AND review_type = ?
                        """,
                        (event["record_id"], event["review_type"]),
                    ).fetchone()
                    if existing and event["_reviewed_epoch"] == existing[0]:
                        raise DatasetReviewError(
                            f"{ledger.name}:{line_number}: ambiguous review events share a timestamp"
                        )
                    if existing is None or event["_reviewed_epoch"] > existing[0]:
                        connection.execute(
                            """
                            INSERT INTO latest_reviews(
                                record_id, review_type, reviewer_id, decision,
                                quality_score, risk_class, rights_status,
                                publication_authorized, reviewed_at, reviewed_epoch
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                            ON CONFLICT(record_id, review_type) DO UPDATE SET
                                reviewer_id = excluded.reviewer_id,
                                decision = excluded.decision,
                                quality_score = excluded.quality_score,
                                risk_class = excluded.risk_class,
                                rights_status = excluded.rights_status,
                                publication_authorized = excluded.publication_authorized,
                                reviewed_at = excluded.reviewed_at,
                                reviewed_epoch = excluded.reviewed_epoch
                            """,
                            (
                                event["record_id"],
                                event["review_type"],
                                event["reviewer_id"],
                                event["decision"],
                                event["quality_score"],
                                event["risk_class"],
                                event["rights_status"],
                                int(event["publication_authorized"]),
                                event["reviewed_at"],
                                event["_reviewed_epoch"],
                            ),
                        )
                    ledger_events += 1
            connection.commit()

            total_records = connection.execute(
                "SELECT COUNT(*) FROM release_records"
            ).fetchone()[0]
            cursor = connection.execute(
                "SELECT record_id FROM release_records ORDER BY record_id"
            )
            for (record_id,) in cursor:
                reviews = {
                    row[0]: {
                        "reviewer_id": row[1],
                        "decision": row[2],
                        "quality_score": row[3],
                        "risk_class": row[4],
                        "rights_status": row[5],
                        "publication_authorized": bool(row[6]),
                        "reviewed_epoch": row[7],
                    }
                    for row in connection.execute(
                        """
                        SELECT review_type, reviewer_id, decision, quality_score,
                               risk_class, rights_status, publication_authorized,
                               reviewed_epoch
                        FROM latest_reviews
                        WHERE record_id = ?
                        """,
                        (record_id,),
                    )
                }
                blockers: list[str] = []
                for review_type in _REQUIRED_REVIEW_TYPES:
                    if review_type not in reviews:
                        blockers.append(f"missing_{review_type}")
                        continue
                    review = reviews[review_type]
                    if review["decision"] != "approved":
                        blockers.append(f"{review_type}_{review['decision']}")
                if all(review_type in reviews for review_type in _REQUIRED_REVIEW_TYPES):
                    technical = reviews["technical"]
                    rights = reviews["rights"]
                    security = reviews["security"]
                    independent_qa = reviews["independent_qa"]
                    publication = reviews["publication_approval"]
                    if technical["quality_score"] < minimum_quality_score:
                        blockers.append("technical_quality_below_threshold")
                    if independent_qa["quality_score"] < minimum_quality_score:
                        blockers.append("independent_qa_quality_below_threshold")
                    for review_type, review in (
                        ("security", security),
                        ("independent_qa", independent_qa),
                    ):
                        if review["risk_class"] not in _RISK_ORDER or (
                            _RISK_ORDER[review["risk_class"]] > _RISK_ORDER[maximum_risk_class]
                        ):
                            blockers.append(f"{review_type}_risk_above_threshold")
                    if rights["rights_status"] != "approved":
                        blockers.append("rights_not_approved")
                    if not publication["publication_authorized"]:
                        blockers.append("publication_not_authorized")
                    latest_review_epoch = max(
                        review["reviewed_epoch"]
                        for review_type, review in reviews.items()
                        if review_type != "publication_approval"
                    )
                    if publication["reviewed_epoch"] < latest_review_epoch:
                        blockers.append("publication_approval_stale")
                    if technical["reviewer_id"] == independent_qa["reviewer_id"]:
                        blockers.append("independent_qa_reviewer_conflict")
                    if publication["reviewer_id"] in {
                        technical["reviewer_id"],
                        independent_qa["reviewer_id"],
                        rights["reviewer_id"],
                        security["reviewer_id"],
                    }:
                        blockers.append("publication_approver_conflict")
                if blockers:
                    for blocker in set(blockers):
                        blocker_counts[blocker] = blocker_counts.get(blocker, 0) + 1
                else:
                    approved_records += 1
                if all(review_type in reviews for review_type in _REQUIRED_REVIEW_TYPES):
                    reviewed_records += 1

            report = {
                "schema_version": "1.0.0",
                "report_type": "dataset_human_review_gate",
                "dataset_id": verification["dataset_id"],
                "dataset_version": verification["dataset_version"],
                "release_manifest_sha256": verification["manifest_sha256"],
                "review_ledger_sha256": ledger_digest.hexdigest(),
                "review_ledger_file": ledger.name,
                "ledger_events": ledger_events,
                "total_records": total_records,
                "reviewed_records": reviewed_records,
                "approved_records": approved_records,
                "blocked_records": total_records - approved_records,
                "required_review_types": list(_REQUIRED_REVIEW_TYPES),
                "policy": {
                    "minimum_quality_score": minimum_quality_score,
                    "maximum_risk_class": maximum_risk_class,
                    "independent_qa_reviewer_required": True,
                    "separate_publication_approver_required": True,
                },
                "blocker_counts": dict(sorted(blocker_counts.items())),
                "approved_for_release": approved_records == total_records,
            }
        finally:
            connection.close()
    except sqlite3.Error as exc:
        raise DatasetReviewError(f"review evaluation database error: {exc}") from exc
    finally:
        temporary_directory.cleanup()

    report["report_payload_sha256"] = hashlib.sha256(
        json.dumps(report, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    report_file_sha256 = _write_report(output, report)
    return {
        "dataset_id": report["dataset_id"],
        "dataset_version": report["dataset_version"],
        "total_records": report["total_records"],
        "approved_records": report["approved_records"],
        "blocked_records": report["blocked_records"],
        "approved_for_release": report["approved_for_release"],
        "report_path": str(output),
        "report_sha256": report_file_sha256,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", required=True, type=Path)
    parser.add_argument("--ledger", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--minimum-quality-score", type=int, default=4)
    parser.add_argument("--maximum-risk-class", choices=tuple(_RISK_ORDER), default="medium")
    args = parser.parse_args(argv)
    try:
        result = evaluate_dataset_reviews(
            args.release,
            args.ledger,
            args.report,
            minimum_quality_score=args.minimum_quality_score,
            maximum_risk_class=args.maximum_risk_class,
        )
    except (DatasetReviewError, OSError, UnicodeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["approved_for_release"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
