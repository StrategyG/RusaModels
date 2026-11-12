import hashlib
import json
from pathlib import Path

import pytest

from rusamodels.frozen_evaluation import (
    FrozenEvaluationError,
    FrozenEvaluationPolicy,
    validate_frozen_evaluation_suite,
)


_DOMAINS = (
    "indonesian_instruction",
    "english_bilingual",
    "reasoning_problem_solving",
    "coding_debugging",
    "architecture_devops",
    "tool_json_contracts",
    "rag_grounding_attribution",
    "safety_prompt_injection",
)


def _records() -> list[dict]:
    return [
        {
            "schema_version": "1.0.0",
            "id": f"record_{index:024x}",
            "domain": domain,
            "language": "id" if domain == "indonesian_instruction" else (
                "en" if domain == "english_bilingual" else "mixed"
            ),
            "prompt": f"Unique frozen evaluation prompt for domain {domain}",
            "expected_outcome": f"Expected observable behavior for {domain}",
            "rubric": [
                {
                    "criterion": "Correctness",
                    "weight": 5,
                    "pass_condition": "The final answer satisfies the stated task without unsupported claims.",
                }
            ],
            "provenance": "human_authored",
            "source_refs": [],
            "risk_class": "low",
        }
        for index, domain in enumerate(_DOMAINS, start=1)
    ]


def _write_suite(tmp_path: Path, records: list[dict] | None = None) -> tuple[Path, Path]:
    records = _records() if records is None else records
    evaluation = tmp_path / "frozen-evaluation.jsonl"
    payload = "".join(
        json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
        for record in records
    ).encode("utf-8")
    evaluation.write_bytes(payload)
    manifest = {
        "schema_version": "1.0.0",
        "purpose": "frozen_evaluation",
        "dataset_id": "rusa-frozen-eval",
        "dataset_version": "1.0.0",
        "locked": True,
        "file": evaluation.name,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
        "records": len(records),
        "created_at": "2026-01-01T00:00:00Z",
    }
    manifest_path = tmp_path / "frozen-evaluation.manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return evaluation, manifest_path


def _policy() -> FrozenEvaluationPolicy:
    return FrozenEvaluationPolicy(
        minimum_records=8,
        maximum_records=8,
        mix_tolerance=1.0,
    )


def test_complete_frozen_evaluation_suite_passes(tmp_path: Path) -> None:
    evaluation, manifest = _write_suite(tmp_path)

    result = validate_frozen_evaluation_suite(
        evaluation,
        manifest,
        tmp_path / "reports" / "suite-validation.json",
        policy=_policy(),
    )
    report = json.loads(
        (tmp_path / "reports" / "suite-validation.json").read_text(encoding="utf-8")
    )

    assert result["suite_valid"] is True
    assert result["records"] == 8
    assert len(report["evaluation_sha256"]) == 64
    assert set(report["domain_counts"].values()) == {1}
    assert "Unique frozen evaluation prompt" not in json.dumps(report)


def test_suite_outside_size_range_is_reported_as_blocked(tmp_path: Path) -> None:
    evaluation, manifest = _write_suite(tmp_path, _records()[:7])

    result = validate_frozen_evaluation_suite(
        evaluation,
        manifest,
        tmp_path / "blocked-report.json",
        policy=_policy(),
    )
    report = json.loads((tmp_path / "blocked-report.json").read_text(encoding="utf-8"))

    assert result["suite_valid"] is False
    assert "record_count_out_of_range" in report["blockers"]
    assert "missing_evaluation_domains" in report["blockers"]


def test_duplicate_normalized_prompts_block_the_suite(tmp_path: Path) -> None:
    records = _records()
    records[1]["prompt"] = "  " + records[0]["prompt"].upper() + "  "
    evaluation, manifest = _write_suite(tmp_path, records)

    result = validate_frozen_evaluation_suite(
        evaluation,
        manifest,
        tmp_path / "duplicate-report.json",
        policy=_policy(),
    )
    report = json.loads((tmp_path / "duplicate-report.json").read_text(encoding="utf-8"))

    assert result["suite_valid"] is False
    assert report["duplicate_prompts"] == 1
    assert "duplicate_prompts" in report["blockers"]


def test_project_owned_prompt_requires_pinned_source_reference(tmp_path: Path) -> None:
    records = _records()
    records[0]["provenance"] = "project_owned"
    evaluation, manifest = _write_suite(tmp_path, records)

    with pytest.raises(FrozenEvaluationError, match="require source_refs"):
        validate_frozen_evaluation_suite(
            evaluation,
            manifest,
            tmp_path / "invalid-report.json",
            policy=_policy(),
        )


def test_frozen_suite_rejects_direct_identifier_and_malformed_enum_types(tmp_path: Path) -> None:
    records = _records()
    records[0]["id"] = "customer@example.com"
    evaluation, manifest = _write_suite(tmp_path, records)

    with pytest.raises(FrozenEvaluationError, match="pseudonymized record ID"):
        validate_frozen_evaluation_suite(
            evaluation,
            manifest,
            tmp_path / "invalid-id-report.json",
            policy=_policy(),
        )

    records = _records()
    records[0]["language"] = ["id"]
    evaluation, manifest = _write_suite(tmp_path, records)
    with pytest.raises(FrozenEvaluationError, match="language must be"):
        validate_frozen_evaluation_suite(
            evaluation,
            manifest,
            tmp_path / "invalid-language-report.json",
            policy=_policy(),
        )


def test_frozen_suite_manifest_digest_must_match_file(tmp_path: Path) -> None:
    evaluation, manifest_path = _write_suite(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(FrozenEvaluationError, match="SHA-256 does not match"):
        validate_frozen_evaluation_suite(
            evaluation,
            manifest_path,
            tmp_path / "invalid-manifest-report.json",
            policy=_policy(),
        )


def test_frozen_suite_rejects_duplicate_record_ids(tmp_path: Path) -> None:
    records = _records()
    records[1]["id"] = records[0]["id"]
    evaluation, manifest = _write_suite(tmp_path, records)

    with pytest.raises(FrozenEvaluationError, match="duplicate record ID"):
        validate_frozen_evaluation_suite(
            evaluation,
            manifest,
            tmp_path / "duplicate-id-report.json",
            policy=_policy(),
        )
