"""Validate the versioned frozen release evaluation suite before model benchmarking."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import tempfile
import unicodedata
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


class FrozenEvaluationError(ValueError):
    """Raised when a frozen evaluation suite or its manifest is invalid."""


_DOMAIN_TARGETS = {
    "indonesian_instruction": 0.20,
    "english_bilingual": 0.10,
    "reasoning_problem_solving": 0.12,
    "coding_debugging": 0.15,
    "architecture_devops": 0.12,
    "tool_json_contracts": 0.12,
    "rag_grounding_attribution": 0.09,
    "safety_prompt_injection": 0.10,
}
_RECORD_ID = re.compile(r"^record_[0-9a-f]{24}$")
_DATASET_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
_SOURCE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_RECORD_KEYS = {
    "schema_version",
    "id",
    "domain",
    "language",
    "prompt",
    "expected_outcome",
    "rubric",
    "provenance",
    "source_refs",
    "risk_class",
}
_SOURCE_REF_KEYS = {"source_id", "revision", "locator", "sha256"}
_RUBRIC_KEYS = {"criterion", "weight", "pass_condition"}
_MANIFEST_KEYS = {
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


@dataclass(frozen=True)
class FrozenEvaluationPolicy:
    minimum_records: int = 800
    maximum_records: int = 1200
    mix_tolerance: float = 0.20


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
        raise FrozenEvaluationError(f"{source}: invalid JSON: {exc}") from exc


def _hash_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def _validate_manifest(evaluation: Path, manifest_path: Path) -> tuple[dict[str, Any], str, str]:
    if not evaluation.is_file() or not manifest_path.is_file():
        raise FrozenEvaluationError("frozen evaluation file or manifest is missing")
    manifest_bytes = manifest_path.read_bytes()
    manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    manifest = _load_json(manifest_bytes.decode("utf-8"), manifest_path.name)
    if not isinstance(manifest, dict) or set(manifest) != _MANIFEST_KEYS:
        raise FrozenEvaluationError("frozen evaluation manifest has unsupported or missing fields")
    if manifest["schema_version"] != "1.0.0" or manifest["purpose"] != "frozen_evaluation":
        raise FrozenEvaluationError("manifest is not a supported frozen evaluation manifest")
    if manifest["locked"] is not True:
        raise FrozenEvaluationError("frozen evaluation manifest must declare locked=true")
    for field in ("dataset_id", "dataset_version"):
        if not isinstance(manifest[field], str) or not _DATASET_ID.fullmatch(manifest[field]):
            raise FrozenEvaluationError(f"frozen evaluation {field} is invalid")
    if (
        not isinstance(manifest["file"], str)
        or manifest["file"] != evaluation.name
        or Path(manifest["file"]).name != manifest["file"]
    ):
        raise FrozenEvaluationError("manifest file name does not match the frozen evaluation file")
    if not isinstance(manifest["sha256"], str) or not _SHA256.fullmatch(manifest["sha256"]):
        raise FrozenEvaluationError("frozen evaluation SHA-256 is invalid")
    for field in ("size_bytes", "records"):
        if type(manifest[field]) is not int or manifest[field] < 1:
            raise FrozenEvaluationError(f"frozen evaluation {field} must be a positive integer")
    try:
        created_at = datetime.fromisoformat(manifest["created_at"].replace("Z", "+00:00"))
    except (AttributeError, TypeError, ValueError) as exc:
        raise FrozenEvaluationError("frozen evaluation created_at must be ISO-8601") from exc
    if created_at.tzinfo is None or created_at.utcoffset() is None:
        raise FrozenEvaluationError("frozen evaluation created_at must include a timezone")
    actual_sha256, actual_size = _hash_file(evaluation)
    if actual_sha256 != manifest["sha256"]:
        raise FrozenEvaluationError("frozen evaluation file SHA-256 does not match its manifest")
    if actual_size != manifest["size_bytes"]:
        raise FrozenEvaluationError("frozen evaluation file size does not match its manifest")
    return manifest, manifest_sha256, actual_sha256


def _validate_record(value: Any, source: str) -> tuple[str, str]:
    if not isinstance(value, dict) or set(value) != _RECORD_KEYS:
        raise FrozenEvaluationError(f"{source}: record has unsupported or missing fields")
    if value["schema_version"] != "1.0.0":
        raise FrozenEvaluationError(f"{source}: unsupported record schema_version")
    if not isinstance(value["id"], str) or not _RECORD_ID.fullmatch(value["id"]):
        raise FrozenEvaluationError(f"{source}: id must be a pseudonymized record ID")
    if not isinstance(value["domain"], str) or value["domain"] not in _DOMAIN_TARGETS:
        raise FrozenEvaluationError(f"{source}: unsupported evaluation domain")
    if value["language"] not in {"id", "en", "mixed"}:
        raise FrozenEvaluationError(f"{source}: language must be id, en, or mixed")
    if not isinstance(value["prompt"], str) or not value["prompt"].strip() or len(value["prompt"]) > 20000:
        raise FrozenEvaluationError(f"{source}: prompt must contain 1-20000 characters")
    if (
        not isinstance(value["expected_outcome"], str)
        or not value["expected_outcome"].strip()
        or len(value["expected_outcome"]) > 8000
    ):
        raise FrozenEvaluationError(f"{source}: expected_outcome must contain 1-8000 characters")
    if value["provenance"] not in {
        "project_owned", "licensed_source", "human_authored", "synthetic"
    }:
        raise FrozenEvaluationError(f"{source}: unsupported provenance")
    if value["risk_class"] not in {"low", "medium", "high", "critical"}:
        raise FrozenEvaluationError(f"{source}: risk_class is invalid")

    rubric = value["rubric"]
    if not isinstance(rubric, list) or not 1 <= len(rubric) <= 8:
        raise FrozenEvaluationError(f"{source}: rubric must contain 1-8 criteria")
    criteria: set[str] = set()
    for item in rubric:
        if not isinstance(item, dict) or set(item) != _RUBRIC_KEYS:
            raise FrozenEvaluationError(f"{source}: rubric item has unsupported or missing fields")
        criterion = item["criterion"]
        if not isinstance(criterion, str) or not criterion.strip() or len(criterion) > 160:
            raise FrozenEvaluationError(f"{source}: rubric criterion is invalid")
        normalized_criterion = criterion.strip().casefold()
        if normalized_criterion in criteria:
            raise FrozenEvaluationError(f"{source}: duplicate rubric criterion")
        criteria.add(normalized_criterion)
        if type(item["weight"]) is not int or not 1 <= item["weight"] <= 5:
            raise FrozenEvaluationError(f"{source}: rubric weight must be an integer from 1 to 5")
        if (
            not isinstance(item["pass_condition"], str)
            or not item["pass_condition"].strip()
            or len(item["pass_condition"]) > 1000
        ):
            raise FrozenEvaluationError(f"{source}: rubric pass_condition is invalid")

    source_refs = value["source_refs"]
    if not isinstance(source_refs, list) or len(source_refs) > 10:
        raise FrozenEvaluationError(f"{source}: source_refs must be a list of at most 10 items")
    seen_refs: set[tuple[str, str, str]] = set()
    for item in source_refs:
        if not isinstance(item, dict) or set(item) != _SOURCE_REF_KEYS:
            raise FrozenEvaluationError(f"{source}: source reference has unsupported or missing fields")
        if not isinstance(item["source_id"], str) or not _SOURCE_ID.fullmatch(item["source_id"]):
            raise FrozenEvaluationError(f"{source}: source reference source_id is invalid")
        if not isinstance(item["revision"], str) or not item["revision"].strip() or len(item["revision"]) > 128:
            raise FrozenEvaluationError(f"{source}: source reference revision is invalid")
        if not isinstance(item["locator"], str) or not item["locator"].strip() or len(item["locator"]) > 1024:
            raise FrozenEvaluationError(f"{source}: source reference locator is invalid")
        if not isinstance(item["sha256"], str) or not _SHA256.fullmatch(item["sha256"]):
            raise FrozenEvaluationError(f"{source}: source reference SHA-256 is invalid")
        identity = (item["source_id"], item["revision"], item["locator"])
        if identity in seen_refs:
            raise FrozenEvaluationError(f"{source}: duplicate source reference")
        seen_refs.add(identity)
    if value["provenance"] in {"project_owned", "licensed_source"} and not source_refs:
        raise FrozenEvaluationError(f"{source}: project-owned/licensed prompts require source_refs")

    normalized_prompt = unicodedata.normalize("NFC", " ".join(value["prompt"].split())).casefold()
    return value["domain"], hashlib.sha256(normalized_prompt.encode()).hexdigest()


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


def validate_frozen_evaluation_suite(
    evaluation_path: str | Path,
    manifest_path: str | Path,
    report_path: str | Path,
    *,
    policy: FrozenEvaluationPolicy = FrozenEvaluationPolicy(),
) -> dict[str, Any]:
    """Validate frozen prompt contracts, manifest integrity, suite size, and domain mix."""
    if (
        type(policy.minimum_records) is not int
        or type(policy.maximum_records) is not int
        or policy.minimum_records < 1
        or policy.maximum_records < policy.minimum_records
    ):
        raise FrozenEvaluationError("record-count policy must be positive and ordered")
    if (
        type(policy.mix_tolerance) not in {int, float}
        or not math.isfinite(policy.mix_tolerance)
        or not 0 <= policy.mix_tolerance <= 1
    ):
        raise FrozenEvaluationError("mix_tolerance must be between 0 and 1")

    evaluation_input = Path(evaluation_path).expanduser()
    manifest_input = Path(manifest_path).expanduser()
    report_input = Path(report_path).expanduser()
    if evaluation_input.is_symlink() or manifest_input.is_symlink() or report_input.is_symlink():
        raise FrozenEvaluationError("symlink paths are not allowed for frozen suite inputs or report")
    evaluation = evaluation_input.parent.resolve() / evaluation_input.name
    manifest = manifest_input.parent.resolve() / manifest_input.name
    output = report_input.parent.resolve() / report_input.name
    if output in {evaluation, manifest}:
        raise FrozenEvaluationError("report path must differ from frozen evaluation inputs")
    if output.exists():
        raise FrozenEvaluationError(f"report already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    frozen_manifest, manifest_sha256, evaluation_sha256 = _validate_manifest(evaluation, manifest)
    domain_counts: Counter[str] = Counter()
    prompt_hashes: set[str] = set()
    record_ids: set[str] = set()
    duplicate_prompts = 0
    records = 0
    with evaluation.open("rb") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                raise FrozenEvaluationError(f"{evaluation.name}:{line_number}: blank lines are not allowed")
            if len(raw_line) > 256 * 1024:
                raise FrozenEvaluationError(f"{evaluation.name}:{line_number}: record exceeds 256 KiB")
            try:
                line = raw_line.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise FrozenEvaluationError(f"{evaluation.name}:{line_number}: file must be UTF-8") from exc
            domain, prompt_digest = _validate_record(
                _load_json(line, f"{evaluation.name}:{line_number}"),
                f"{evaluation.name}:{line_number}",
            )
            record_id = _load_json(line, f"{evaluation.name}:{line_number}")["id"]
            if record_id in record_ids:
                raise FrozenEvaluationError(f"{evaluation.name}:{line_number}: duplicate record ID")
            record_ids.add(record_id)
            if prompt_digest in prompt_hashes:
                duplicate_prompts += 1
            prompt_hashes.add(prompt_digest)
            domain_counts[domain] += 1
            records += 1

    if records != frozen_manifest["records"]:
        raise FrozenEvaluationError("frozen evaluation record count does not match its manifest")
    evaluation_sha256_after, evaluation_size_after = _hash_file(evaluation)
    if (
        evaluation_sha256_after != evaluation_sha256
        or evaluation_size_after != frozen_manifest["size_bytes"]
    ):
        raise FrozenEvaluationError("frozen evaluation file changed during suite validation")

    missing_domains = sorted(domain for domain in _DOMAIN_TARGETS if domain_counts[domain] == 0)
    out_of_band: dict[str, dict[str, int]] = {}
    for domain, proportion in _DOMAIN_TARGETS.items():
        target = records * proportion
        minimum = max(1, math.floor(target * (1 - policy.mix_tolerance)))
        maximum = max(minimum, math.ceil(target * (1 + policy.mix_tolerance)))
        actual = domain_counts[domain]
        if actual < minimum or actual > maximum:
            out_of_band[domain] = {
                "actual": actual,
                "minimum": minimum,
                "target": round(target, 2),
                "maximum": maximum,
            }

    blockers: list[str] = []
    if not policy.minimum_records <= records <= policy.maximum_records:
        blockers.append("record_count_out_of_range")
    if missing_domains:
        blockers.append("missing_evaluation_domains")
    if out_of_band:
        blockers.append("domain_mix_out_of_range")
    if duplicate_prompts:
        blockers.append("duplicate_prompts")

    report = {
        "schema_version": "1.0.0",
        "report_type": "frozen_evaluation_suite_validation",
        "dataset_id": frozen_manifest["dataset_id"],
        "dataset_version": frozen_manifest["dataset_version"],
        "evaluation_sha256": evaluation_sha256,
        "manifest_sha256": manifest_sha256,
        "records": records,
        "minimum_records": policy.minimum_records,
        "maximum_records": policy.maximum_records,
        "domain_counts": {domain: domain_counts[domain] for domain in _DOMAIN_TARGETS},
        "target_domain_proportions": _DOMAIN_TARGETS,
        "mix_tolerance": policy.mix_tolerance,
        "missing_domains": missing_domains,
        "out_of_band_domains": out_of_band,
        "duplicate_prompts": duplicate_prompts,
        "blockers": blockers,
        "suite_valid": not blockers,
    }
    report_payload_sha256 = hashlib.sha256(
        json.dumps(report, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    report["report_payload_sha256"] = report_payload_sha256
    report_sha256 = _write_report(output, report)
    return {
        "dataset_id": report["dataset_id"],
        "dataset_version": report["dataset_version"],
        "records": records,
        "suite_valid": report["suite_valid"],
        "blockers": blockers,
        "report_path": str(output),
        "report_sha256": report_sha256,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--minimum-records", type=int, default=800)
    parser.add_argument("--maximum-records", type=int, default=1200)
    parser.add_argument("--mix-tolerance", type=float, default=0.20)
    args = parser.parse_args(argv)
    try:
        result = validate_frozen_evaluation_suite(
            args.evaluation,
            args.manifest,
            args.report,
            policy=FrozenEvaluationPolicy(
                minimum_records=args.minimum_records,
                maximum_records=args.maximum_records,
                mix_tolerance=args.mix_tolerance,
            ),
        )
    except (FrozenEvaluationError, OSError, UnicodeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["suite_valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
