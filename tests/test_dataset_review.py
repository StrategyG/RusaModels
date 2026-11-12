import json
from pathlib import Path

import pytest

from rusamodels.data_pipeline import (
    DatasetReleaseConfig,
    DatasetSource,
    build_dataset_release,
)
from rusamodels.dataset_review import (
    DatasetReviewError,
    evaluate_dataset_reviews,
)


def _write_jsonl(path: Path, records: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )
    return path


def _release(tmp_path: Path) -> Path:
    source = _write_jsonl(
        tmp_path / "source.jsonl",
        [
            {
                "id": f"example-{index}",
                "group_id": f"document-{index}",
                "text": f"Human-reviewed technical example number {index}",
            }
            for index in range(1, 8)
        ],
    )
    return build_dataset_release(
        DatasetReleaseConfig(
            dataset_id="review-gate-test",
            version="1.0.0",
            sources=(DatasetSource("approved-source", source, "CC0-1.0"),),
            output_root=tmp_path / "releases",
            seed=19,
            validation_fraction=0.2,
            test_fraction=0.2,
        )
    )


def _record_ids(release: Path) -> list[str]:
    ids = []
    for split in ("train", "validation", "test"):
        ids.extend(
            json.loads(line)["id"]
            for line in (release / f"{split}.jsonl").read_text(encoding="utf-8").splitlines()
        )
    return ids


def _event(record_id: str, review_type: str, *, decision: str = "approved", reviewer_id: str | None = None) -> dict:
    event = {
        "review_id": f"review-{record_id}-{review_type}-{decision}",
        "record_id": record_id,
        "review_type": review_type,
        "reviewer_id": reviewer_id or {
            "technical": "reviewer_technical",
            "rights": "reviewer_rights",
            "security": "reviewer_security",
            "independent_qa": "reviewer_independent_qa",
            "publication_approval": "reviewer_publisher",
        }[review_type],
        "decision": decision,
        "quality_score": None,
        "risk_class": None,
        "rights_status": None,
        "evidence_status": None,
        "language": None,
        "task_type": None,
        "publication_authorized": False,
        "reviewed_at": {
            "technical": "2026-01-01T00:00:00Z",
            "rights": "2026-01-01T00:01:00Z",
            "security": "2026-01-01T00:02:00Z",
            "independent_qa": "2026-01-01T00:03:00Z",
            "publication_approval": "2026-01-01T00:04:00Z",
        }[review_type],
    }
    if review_type == "technical":
        event.update({
            "quality_score": 5,
            "evidence_status": "implementation_observation",
            "language": "id",
            "task_type": "technical_instruction",
        })
    elif review_type == "rights":
        event["rights_status"] = "approved"
    elif review_type == "security":
        event["risk_class"] = "low"
    elif review_type == "independent_qa":
        event.update({
            "quality_score": 5,
            "risk_class": "low",
            "evidence_status": "test_assertion",
        })
    elif review_type == "publication_approval":
        event["publication_authorized"] = True
    return event


def _write_ledger(path: Path, record_ids: list[str], *, omit: tuple[str, str] | None = None) -> Path:
    events = []
    for record_id in record_ids:
        for review_type in (
            "technical",
            "rights",
            "security",
            "independent_qa",
            "publication_approval",
        ):
            if omit == (record_id, review_type):
                continue
            events.append(_event(record_id, review_type))
    return _write_jsonl(path, events)


def test_complete_independent_review_and_publication_approval_passes(tmp_path: Path) -> None:
    release = _release(tmp_path)
    ids = _record_ids(release)
    ledger = _write_ledger(tmp_path / "reviews.jsonl", ids)
    report_path = tmp_path / "reports" / "review-report.json"

    result = evaluate_dataset_reviews(release, ledger, report_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert result["approved_for_release"] is True
    assert result["approved_records"] == len(ids)
    assert result["blocked_records"] == 0
    assert report["required_review_types"] == [
        "technical",
        "rights",
        "security",
        "independent_qa",
        "publication_approval",
    ]
    assert len(report["report_payload_sha256"]) == 64


def test_missing_review_type_blocks_publication(tmp_path: Path) -> None:
    release = _release(tmp_path)
    ids = _record_ids(release)
    ledger = _write_ledger(
        tmp_path / "reviews.jsonl",
        ids,
        omit=(ids[0], "rights"),
    )

    result = evaluate_dataset_reviews(release, ledger, tmp_path / "report.json")

    assert result["approved_for_release"] is False
    assert result["blocked_records"] == 1


def test_latest_rejected_review_overrides_earlier_approval(tmp_path: Path) -> None:
    release = _release(tmp_path)
    ids = _record_ids(release)
    ledger = _write_ledger(tmp_path / "reviews.jsonl", ids)
    with ledger.open("a", encoding="utf-8") as handle:
        event = _event(ids[0], "security", decision="rejected")
        event["review_id"] = "review-latest-security-rejection"
        event["reviewed_at"] = "2026-01-02T00:00:00Z"
        handle.write(json.dumps(event) + "\n")

    result = evaluate_dataset_reviews(release, ledger, tmp_path / "report.json")

    assert result["approved_for_release"] is False
    assert result["blocked_records"] == 1


def test_independent_qa_reviewer_must_differ_from_technical_reviewer(tmp_path: Path) -> None:
    release = _release(tmp_path)
    ids = _record_ids(release)
    ledger = _write_ledger(tmp_path / "reviews.jsonl", ids)
    with ledger.open("a", encoding="utf-8") as handle:
        event = _event(ids[0], "independent_qa", reviewer_id="reviewer_technical")
        event["review_id"] = "review-conflicted-independent-qa"
        event["reviewed_at"] = "2026-01-02T00:00:00Z"
        handle.write(json.dumps(event) + "\n")

    result = evaluate_dataset_reviews(release, ledger, tmp_path / "report.json")

    assert result["approved_for_release"] is False


def test_publication_approver_must_be_independent(tmp_path: Path) -> None:
    release = _release(tmp_path)
    ids = _record_ids(release)
    ledger = _write_ledger(tmp_path / "reviews.jsonl", ids)
    with ledger.open("a", encoding="utf-8") as handle:
        event = _event(ids[0], "publication_approval", reviewer_id="reviewer_independent_qa")
        event["review_id"] = "review-conflicted-publication"
        event["reviewed_at"] = "2026-01-02T00:00:00Z"
        handle.write(json.dumps(event) + "\n")

    result = evaluate_dataset_reviews(release, ledger, tmp_path / "report.json")

    assert result["approved_for_release"] is False


def test_low_quality_and_high_risk_reviews_block_publication(tmp_path: Path) -> None:
    release = _release(tmp_path)
    ids = _record_ids(release)
    ledger = _write_ledger(tmp_path / "reviews.jsonl", ids)
    with ledger.open("a", encoding="utf-8") as handle:
        technical = _event(ids[0], "technical")
        technical["review_id"] = "review-low-quality"
        technical["quality_score"] = 2
        technical["reviewed_at"] = "2026-01-02T00:00:00Z"
        security = _event(ids[0], "security")
        security["review_id"] = "review-high-risk"
        security["risk_class"] = "critical"
        security["reviewed_at"] = "2026-01-02T00:01:00Z"
        handle.write(json.dumps(technical) + "\n")
        handle.write(json.dumps(security) + "\n")

    result = evaluate_dataset_reviews(release, ledger, tmp_path / "report.json")

    assert result["approved_for_release"] is False
    report = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    assert report["blocker_counts"]["technical_quality_below_threshold"] == 1
    assert report["blocker_counts"]["security_risk_above_threshold"] == 1


def test_review_ledger_rejects_unknown_records_and_duplicate_review_ids(tmp_path: Path) -> None:
    release = _release(tmp_path)
    ids = _record_ids(release)
    ledger = _write_ledger(tmp_path / "reviews.jsonl", ids)
    with ledger.open("a", encoding="utf-8") as handle:
        event = _event("record_" + "0" * 24, "technical")
        event["review_id"] = "review-unknown-record"
        handle.write(json.dumps(event) + "\n")

    with pytest.raises(DatasetReviewError, match="unknown record_id"):
        evaluate_dataset_reviews(release, ledger, tmp_path / "report.json")


def test_review_report_cannot_be_written_inside_immutable_release(tmp_path: Path) -> None:
    release = _release(tmp_path)
    ids = _record_ids(release)
    ledger = _write_ledger(tmp_path / "reviews.jsonl", ids)

    with pytest.raises(DatasetReviewError, match="outside the immutable dataset release"):
        evaluate_dataset_reviews(release, ledger, release / "review-report.json")


def test_review_events_require_timezone_and_pseudonymous_reviewer_ids(tmp_path: Path) -> None:
    release = _release(tmp_path)
    ids = _record_ids(release)
    ledger = _write_ledger(tmp_path / "reviews.jsonl", ids)
    lines = ledger.read_text(encoding="utf-8").splitlines()
    event = json.loads(lines[0])
    event["reviewed_at"] = "2026-01-01T00:00:00"
    lines[0] = json.dumps(event)
    ledger.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with pytest.raises(DatasetReviewError, match="must include a timezone"):
        evaluate_dataset_reviews(release, ledger, tmp_path / "report.json")


def test_duplicate_review_event_ids_are_rejected(tmp_path: Path) -> None:
    release = _release(tmp_path)
    ids = _record_ids(release)
    ledger = _write_ledger(tmp_path / "reviews.jsonl", ids)
    events = ledger.read_text(encoding="utf-8").splitlines()
    duplicate = json.loads(events[0])
    duplicate["record_id"] = ids[1]
    events.append(json.dumps(duplicate))
    ledger.write_text("\n".join(events) + "\n", encoding="utf-8")

    with pytest.raises(DatasetReviewError, match="duplicate review_id"):
        evaluate_dataset_reviews(release, ledger, tmp_path / "report.json")


def test_malformed_review_types_fail_closed_without_type_errors(tmp_path: Path) -> None:
    release = _release(tmp_path)
    ids = _record_ids(release)
    ledger = _write_ledger(tmp_path / "reviews.jsonl", ids)
    events = ledger.read_text(encoding="utf-8").splitlines()
    event = json.loads(events[0])
    event["decision"] = ["approved"]
    events[0] = json.dumps(event)
    ledger.write_text("\n".join(events) + "\n", encoding="utf-8")

    with pytest.raises(DatasetReviewError, match="unsupported review decision"):
        evaluate_dataset_reviews(release, ledger, tmp_path / "report.json")


def test_review_gate_rejects_symlink_ledger(tmp_path: Path) -> None:
    release = _release(tmp_path)
    ids = _record_ids(release)
    ledger = _write_ledger(tmp_path / "reviews.jsonl", ids)
    symlink = tmp_path / "reviews-link.jsonl"
    try:
        symlink.symlink_to(ledger)
    except (OSError, NotImplementedError):
        pytest.skip("symlink creation is not supported in this environment")

    with pytest.raises(DatasetReviewError, match="symlink paths"):
        evaluate_dataset_reviews(release, symlink, tmp_path / "report.json")
