"""Production evidence gate for a specific RusaModels release candidate."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


class ProductionEvidenceError(ValueError):
    """Raised when a production evidence bundle cannot be read safely."""


_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_GIT_SHA = re.compile(r"^[0-9a-f]{40}$")


def _pairs_without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_digest(value: Any) -> bool:
    return isinstance(value, str) and _SHA256.fullmatch(value) is not None


def _is_nonempty(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _is_finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _gate(name: str, checks: list[tuple[bool, str]]) -> dict[str, Any]:
    failed = [description for passed, description in checks if not passed]
    return {"name": name, "passed": not failed, "failures": failed}


def _atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(file_descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def check_production_evidence(
    evidence: dict[str, Any],
    evidence_sha256: str,
    *,
    min_soak_hours: float = 24.0,
    max_soak_error_rate: float = 0.001,
    min_soak_requests: int = 100,
    max_backup_age_hours: float = 24.0,
) -> dict[str, Any]:
    if not isinstance(evidence, dict) or evidence.get("schema_version") != "1.0.0":
        raise ProductionEvidenceError("unsupported production evidence schema_version")
    if not _is_digest(evidence_sha256):
        raise ProductionEvidenceError("evidence_sha256 must be a lowercase SHA-256 digest")
    if (
        not _is_finite_number(min_soak_hours)
        or min_soak_hours <= 0
        or not _is_finite_number(max_soak_error_rate)
        or not 0 <= max_soak_error_rate <= 1
        or type(min_soak_requests) is not int
        or min_soak_requests < 1
        or not _is_finite_number(max_backup_age_hours)
        or max_backup_age_hours <= 0
    ):
        raise ProductionEvidenceError("production readiness thresholds are invalid")
    release = evidence.get("release") if isinstance(evidence.get("release"), dict) else {}
    ci = evidence.get("ci") if isinstance(evidence.get("ci"), dict) else {}
    security = evidence.get("security") if isinstance(evidence.get("security"), dict) else {}
    model = evidence.get("model") if isinstance(evidence.get("model"), dict) else {}
    evaluation = evidence.get("evaluation") if isinstance(evidence.get("evaluation"), dict) else {}
    runtime = evidence.get("runtime") if isinstance(evidence.get("runtime"), dict) else {}
    soak = evidence.get("soak") if isinstance(evidence.get("soak"), dict) else {}
    backup = evidence.get("backup_restore") if isinstance(evidence.get("backup_restore"), dict) else {}
    rollback = evidence.get("rollback") if isinstance(evidence.get("rollback"), dict) else {}
    deployment = evidence.get("deployment") if isinstance(evidence.get("deployment"), dict) else {}

    release_commit = release.get("git_commit_sha")
    release_model_id = release.get("model_id")
    release_model_version = release.get("model_version")
    release_artifact_sha = release.get("model_artifact_sha256")

    gates = [
        _gate(
            "release_identity",
            [
                (_is_nonempty(release.get("release_id")), "release.release_id is required"),
                (
                    isinstance(release_commit, str) and _GIT_SHA.fullmatch(release_commit) is not None,
                    "release.git_commit_sha must be a 40-character Git SHA",
                ),
                (
                    isinstance(release.get("container_image_digest"), str)
                    and release.get("container_image_digest", "").startswith("sha256:")
                    and _is_digest(release.get("container_image_digest", "")[7:]),
                    "release.container_image_digest must be sha256:<64 lowercase hex>",
                ),
                (_is_digest(release.get("deployment_config_sha256")), "deployment config SHA-256 is required"),
                (_is_nonempty(release_model_id), "release.model_id is required"),
                (_is_nonempty(release_model_version), "release.model_version is required"),
                (_is_digest(release_artifact_sha), "release.model_artifact_sha256 is required"),
            ],
        ),
        _gate(
            "ci",
            [
                (ci.get("conclusion") == "success", "required CI conclusion must be success"),
                (ci.get("required_checks_passed") is True, "all required CI checks must pass"),
                (ci.get("commit_sha") == release_commit, "CI commit must equal release commit"),
                (_is_nonempty(ci.get("run_url")), "CI run URL is required for audit"),
            ],
        ),
        _gate(
            "security",
            [
                (security.get("status") == "passed", "security scan status must be passed"),
                (type(security.get("critical_findings")) is int and security["critical_findings"] == 0, "critical security findings must be zero"),
                (type(security.get("high_findings")) is int and security["high_findings"] == 0, "high security findings must be zero"),
                (type(security.get("secret_findings")) is int and security["secret_findings"] == 0, "secret findings must be zero"),
                (_is_nonempty(security.get("report_url")), "security report reference is required"),
            ],
        ),
        _gate(
            "model_release",
            [
                (model.get("lifecycle_status") == "approved", "model lifecycle status must be approved"),
                (model.get("model_id") == release_model_id, "model ID must match release identity"),
                (model.get("model_version") == release_model_version, "model version must match release identity"),
                (model.get("artifact_sha256") == release_artifact_sha, "model artifact digest must match release identity"),
                (_is_digest(model.get("manifest_sha256")), "model manifest SHA-256 is required"),
            ],
        ),
        _gate(
            "evaluation",
            [
                (evaluation.get("model_id") == release_model_id, "evaluation model ID must match release"),
                (evaluation.get("model_version") == release_model_version, "evaluation model version must match release"),
                (evaluation.get("artifact_sha256") == release_artifact_sha, "evaluation artifact digest must match release"),
                (evaluation.get("benchmark_gate_passed") is True, "benchmark gate must pass"),
                (evaluation.get("safety_gate_passed") is True, "safety gate must pass"),
                (evaluation.get("regression_gate_passed") is True, "regression gate must pass"),
                (_is_digest(evaluation.get("report_sha256")), "evaluation report SHA-256 is required"),
            ],
        ),
        _gate(
            "runtime",
            [
                (runtime.get("model_artifact_sha256") == release_artifact_sha, "runtime artifact digest must match release"),
                (runtime.get("health_check_passed") is True, "runtime health check must pass"),
                (runtime.get("cpu_only") is True, "release runtime must confirm CPU-only execution"),
                (type(runtime.get("oom_events")) is int and runtime["oom_events"] == 0, "runtime OOM event count must be zero"),
                (_is_nonempty(runtime.get("hardware_profile")), "named CPU hardware profile is required"),
                (_is_finite_number(runtime.get("peak_ram_mb")) and runtime["peak_ram_mb"] > 0, "runtime peak RAM measurement is required"),
                (_is_finite_number(runtime.get("free_ram_reserve_mb")) and runtime["free_ram_reserve_mb"] > 0, "runtime free-RAM reserve measurement is required"),
                (_is_finite_number(runtime.get("ttft_ms")) and runtime["ttft_ms"] >= 0, "runtime TTFT measurement is required"),
                (_is_finite_number(runtime.get("tokens_per_second")) and runtime["tokens_per_second"] > 0, "runtime tokens/second measurement is required"),
                (_is_nonempty(runtime.get("benchmark_report_url")), "runtime benchmark report reference is required"),
            ],
        ),
        _gate(
            "soak_test",
            [
                (_is_finite_number(soak.get("duration_hours")) and soak["duration_hours"] >= min_soak_hours, f"soak test must run at least {min_soak_hours} hours"),
                (type(soak.get("requests")) is int and soak["requests"] >= min_soak_requests, f"soak test must include at least {min_soak_requests} requests"),
                (_is_finite_number(soak.get("error_rate")) and 0 <= soak["error_rate"] <= max_soak_error_rate, f"soak error rate must be at most {max_soak_error_rate}"),
                (type(soak.get("oom_events")) is int and soak["oom_events"] == 0, "soak OOM event count must be zero"),
                (type(soak.get("crash_count")) is int and soak["crash_count"] == 0, "soak crash count must be zero"),
                (_is_finite_number(soak.get("p95_latency_ms")) and soak["p95_latency_ms"] > 0, "soak p95 latency measurement is required"),
                (_is_nonempty(soak.get("report_url")), "soak-test report reference is required"),
            ],
        ),
        _gate(
            "backup_restore",
            [
                (backup.get("database_backup_verified") is True, "database backup verification must pass"),
                (backup.get("object_storage_backup_verified") is True, "object-storage backup verification must pass"),
                (backup.get("restore_test_passed") is True, "database and artifact restore test must pass"),
                (_is_finite_number(backup.get("backup_age_hours")) and 0 <= backup["backup_age_hours"] <= max_backup_age_hours, f"latest verified backup must be no older than {max_backup_age_hours} hours"),
                (_is_nonempty(backup.get("report_url")), "backup/restore report reference is required"),
            ],
        ),
        _gate(
            "rollback",
            [
                (rollback.get("software_rollback_test_passed") is True, "software rollback test must pass"),
                (rollback.get("model_rollback_test_passed") is True, "model rollback test must pass"),
                (_is_nonempty(rollback.get("report_url")), "rollback report reference is required"),
            ],
        ),
        _gate(
            "deployment",
            [
                (deployment.get("https_enforced") is True, "HTTPS must be enforced"),
                (deployment.get("postgres_sslmode") == "verify-full", "PostgreSQL must use sslmode=verify-full"),
                (deployment.get("secrets_from_secret_manager") is True, "production secrets must come from a secret manager"),
                (deployment.get("object_storage_external") is True, "production object storage must be external"),
                (deployment.get("persistent_storage") is True, "persistent model/artifact storage must be configured"),
                (deployment.get("monitoring_enabled") is True, "monitoring must be enabled"),
                (deployment.get("alerting_enabled") is True, "alerting must be enabled"),
                (_is_nonempty(deployment.get("deployment_profile")), "deployment profile name is required"),
            ],
        ),
    ]
    failed = [gate["name"] for gate in gates if not gate["passed"]]
    return {
        "schema_version": "1.0.0",
        "release_id": release.get("release_id"),
        "evidence_sha256": evidence_sha256,
        "status": "production_ready" if not failed else "not_ready",
        "generated_at": datetime.now(UTC).isoformat(),
        "summary": {
            "total_gates": len(gates),
            "passed_gates": sum(gate["passed"] for gate in gates),
            "failed_gates": failed,
        },
        "gates": gates,
    }


def validate_evidence_file(evidence_path: str | Path) -> tuple[dict[str, Any], str]:
    raw_path = Path(evidence_path).expanduser()
    if raw_path.is_symlink():
        raise ProductionEvidenceError("production evidence file must not be a symlink")
    path = raw_path.resolve()
    if not path.is_file():
        raise ProductionEvidenceError(f"production evidence file does not exist: {path}")
    try:
        evidence = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-standard JSON constant: {value}")
            ),
            object_pairs_hook=_pairs_without_duplicates,
        )
    except (json.JSONDecodeError, UnicodeError, ValueError) as exc:
        raise ProductionEvidenceError(f"production evidence is invalid JSON: {exc}") from exc
    return evidence, _sha256_file(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--min-soak-hours", type=float, default=24.0)
    parser.add_argument("--max-soak-error-rate", type=float, default=0.001)
    parser.add_argument("--min-soak-requests", type=int, default=100)
    parser.add_argument("--max-backup-age-hours", type=float, default=24.0)
    args = parser.parse_args(argv)
    try:
        evidence, evidence_sha = validate_evidence_file(args.evidence)
        report = check_production_evidence(
            evidence,
            evidence_sha,
            min_soak_hours=args.min_soak_hours,
            max_soak_error_rate=args.max_soak_error_rate,
            min_soak_requests=args.min_soak_requests,
            max_backup_age_hours=args.max_backup_age_hours,
        )
        output = args.output.expanduser()
        if output.is_symlink():
            raise ProductionEvidenceError("production readiness report output must not be a symlink")
        output = output.resolve()
        if output == args.evidence.expanduser().resolve() or output.exists():
            raise ProductionEvidenceError("production readiness report must be a new file")
        _atomic_write_json(output, report)
    except (ProductionEvidenceError, OSError, UnicodeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "production_ready" else 1


if __name__ == "__main__":
    raise SystemExit(main())
