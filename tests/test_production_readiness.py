import json
from pathlib import Path

from rusamodels.production_readiness import (
    check_production_evidence,
    validate_evidence_file,
)


def _valid_evidence() -> dict:
    commit = "a" * 40
    artifact = "b" * 64
    return {
        "schema_version": "1.0.0",
        "release": {
            "release_id": "rusamodels-1.0.0-rc1",
            "git_commit_sha": commit,
            "container_image_digest": "sha256:" + "c" * 64,
            "deployment_config_sha256": "d" * 64,
            "model_id": "rusaai-edge",
            "model_version": "1.0.0",
            "model_artifact_sha256": artifact,
        },
        "ci": {
            "commit_sha": commit,
            "conclusion": "success",
            "required_checks_passed": True,
            "run_url": "https://github.com/StrategyG/RusaModels/actions/runs/1",
        },
        "security": {
            "status": "passed",
            "critical_findings": 0,
            "high_findings": 0,
            "secret_findings": 0,
            "report_url": "https://example.org/security-report",
        },
        "model": {
            "model_id": "rusaai-edge",
            "model_version": "1.0.0",
            "artifact_sha256": artifact,
            "manifest_sha256": "e" * 64,
            "lifecycle_status": "approved",
        },
        "evaluation": {
            "model_id": "rusaai-edge",
            "model_version": "1.0.0",
            "artifact_sha256": artifact,
            "benchmark_gate_passed": True,
            "safety_gate_passed": True,
            "regression_gate_passed": True,
            "report_sha256": "f" * 64,
        },
        "runtime": {
            "model_artifact_sha256": artifact,
            "health_check_passed": True,
            "cpu_only": True,
            "oom_events": 0,
            "hardware_profile": "Intel Core i5 / 16 GB RAM / Windows 11",
            "peak_ram_mb": 8200,
            "free_ram_reserve_mb": 2048,
            "ttft_ms": 1200,
            "tokens_per_second": 8.5,
            "benchmark_report_url": "https://example.org/runtime-report",
        },
        "soak": {
            "duration_hours": 24,
            "requests": 1000,
            "error_rate": 0.0005,
            "oom_events": 0,
            "crash_count": 0,
            "p95_latency_ms": 1800,
            "report_url": "https://example.org/soak-report",
        },
        "backup_restore": {
            "database_backup_verified": True,
            "object_storage_backup_verified": True,
            "restore_test_passed": True,
            "backup_age_hours": 4,
            "report_url": "https://example.org/backup-report",
        },
        "rollback": {
            "software_rollback_test_passed": True,
            "model_rollback_test_passed": True,
            "report_url": "https://example.org/rollback-report",
        },
        "deployment": {
            "https_enforced": True,
            "postgres_sslmode": "verify-full",
            "secrets_from_secret_manager": True,
            "object_storage_external": True,
            "persistent_storage": True,
            "monitoring_enabled": True,
            "alerting_enabled": True,
            "deployment_profile": "cpu-production-16gb",
        },
    }


def test_complete_evidence_bundle_passes_all_production_gates() -> None:
    report = check_production_evidence(_valid_evidence(), "1" * 64)
    assert report["status"] == "production_ready"
    assert report["summary"]["total_gates"] == 10
    assert report["summary"]["passed_gates"] == 10
    assert report["summary"]["failed_gates"] == []


def test_production_gate_fails_when_ci_commit_does_not_match_release() -> None:
    evidence = _valid_evidence()
    evidence["ci"]["commit_sha"] = "9" * 40
    report = check_production_evidence(evidence, "1" * 64)
    assert report["status"] == "not_ready"
    ci_gate = next(gate for gate in report["gates"] if gate["name"] == "ci")
    assert ci_gate["passed"] is False


def test_production_gate_fails_on_model_artifact_mismatch() -> None:
    evidence = _valid_evidence()
    evidence["evaluation"]["artifact_sha256"] = "9" * 64
    report = check_production_evidence(evidence, "1" * 64)
    assert "evaluation" in report["summary"]["failed_gates"]


def test_production_gate_requires_full_soak_window_and_zero_oom() -> None:
    evidence = _valid_evidence()
    evidence["soak"]["duration_hours"] = 23.5
    evidence["runtime"]["oom_events"] = 1
    report = check_production_evidence(evidence, "1" * 64)
    assert "soak_test" in report["summary"]["failed_gates"]
    assert "runtime" in report["summary"]["failed_gates"]


def test_production_gate_rejects_boolean_security_finding_counts() -> None:
    evidence = _valid_evidence()
    evidence["security"]["critical_findings"] = False
    report = check_production_evidence(evidence, "1" * 64)
    assert "security" in report["summary"]["failed_gates"]


def test_validate_evidence_file_hashes_exact_input(tmp_path: Path) -> None:
    evidence_path = tmp_path / "production-evidence.json"
    evidence_path.write_text(json.dumps(_valid_evidence()), encoding="utf-8")
    evidence, digest = validate_evidence_file(evidence_path)
    assert evidence["release"]["release_id"] == "rusamodels-1.0.0-rc1"
    assert len(digest) == 64


def test_production_gate_reports_missing_evidence_instead_of_certifying() -> None:
    report = check_production_evidence({"schema_version": "1.0.0"}, "1" * 64)
    assert report["status"] == "not_ready"
    assert report["summary"]["failed_gates"]
