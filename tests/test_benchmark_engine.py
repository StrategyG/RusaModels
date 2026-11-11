from __future__ import annotations

import hashlib
import json
from pathlib import Path

import httpx
import pytest

from rusamodels.evaluation.compare import (
    BenchmarkComparisonError,
    compare_benchmark_reports,
)
from rusamodels.evaluation.benchmark import (
    BenchmarkError,
    freeze_benchmark_suite,
    load_benchmark_suite,
    run_benchmark,
)


def _write_suite(tmp_path: Path, records: list[dict]) -> Path:
    suite = tmp_path / "suite.jsonl"
    suite.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )
    manifest = {
        "schema_version": "1.0.0",
        "suite_id": "rusaai-smoke",
        "suite_version": "1.0.0",
        "frozen": True,
        "file_name": suite.name,
        "file_sha256": hashlib.sha256(suite.read_bytes()).hexdigest(),
        "size_bytes": suite.stat().st_size,
        "record_count": len(records),
        "license_ids": ["CC0-1.0"],
    }
    suite.with_suffix(suite.suffix + ".manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return suite


def _records() -> list[dict]:
    return [
        {
            "id": "language-001",
            "category": "language",
            "messages": [{"role": "user", "content": "Say hello"}],
            "scorer": "contains",
            "expected": "RusaModels",
        },
        {
            "id": "json-001",
            "category": "structured_output",
            "messages": [{"role": "user", "content": "Return a JSON object"}],
            "scorer": "json_valid",
        },
    ]


def test_frozen_suite_load_and_openai_compatible_benchmark(tmp_path: Path) -> None:
    suite = _write_suite(tmp_path, _records())

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        prompt = payload["messages"][-1]["content"]
        if prompt == "Return a JSON object":
            content = '{"ok": true}'
        else:
            content = "RusaModels is ready"
        assert payload["temperature"] == 0
        assert payload["stream"] is False
        assert request.headers["authorization"] == "Bearer test-secret"
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": content}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 8, "completion_tokens": 4},
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    output = tmp_path / "report.json"
    report = run_benchmark(
        suite,
        "http://127.0.0.1:8080/openai/v1",
        "carbon-tiny",
        output,
        api_key="test-secret",
        model_artifact_sha256="a" * 64,
        runtime_profile="cpu-8gb",
        min_accuracy=1.0,
        client=client,
    )
    metadata, records = load_benchmark_suite(suite)
    assert metadata["suite_id"] == "rusaai-smoke"
    assert len(records) == 2
    assert report["summary"]["total"] == 2
    assert report["summary"]["passed"] == 2
    assert report["summary"]["accuracy"] == 1.0
    assert report["summary"]["min_accuracy_passed"] is True
    assert report["token_usage"] == {"prompt_tokens": 16, "completion_tokens": 8}
    assert report["by_category"]["language"]["accuracy"] == 1.0
    assert report["model"]["artifact_sha256"] == "a" * 64
    saved = output.read_text(encoding="utf-8")
    assert "Say hello" not in saved
    assert "test-secret" not in saved


def test_benchmark_records_http_failures_and_fails_accuracy_gate(tmp_path: Path) -> None:
    suite = _write_suite(tmp_path, _records())
    client = httpx.Client(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(503, json={"error": "unavailable"})
        )
    )
    report = run_benchmark(
        suite,
        "http://localhost:8000/v1",
        "carbon-tiny",
        tmp_path / "failed-report.json",
        max_retries=0,
        min_accuracy=0.8,
        client=client,
    )
    assert report["summary"]["failed_requests"] == 2
    assert report["summary"]["accuracy"] is None
    assert report["summary"]["min_accuracy_passed"] is False
    assert report["results"][0]["http_status"] == 503


def test_benchmark_rejects_modified_frozen_suite(tmp_path: Path) -> None:
    suite = _write_suite(tmp_path, _records())
    suite.write_text(suite.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(BenchmarkError, match="checksum or size"):
        load_benchmark_suite(suite)


def test_benchmark_rejects_duplicate_prompts_with_different_ids(tmp_path: Path) -> None:
    records = _records()
    records[1]["messages"] = records[0]["messages"]
    suite = _write_suite(tmp_path, records)
    with pytest.raises(BenchmarkError, match="duplicate prompts"):
        load_benchmark_suite(suite)


def test_benchmark_requires_frozen_manifest(tmp_path: Path) -> None:
    suite = _write_suite(tmp_path, _records())
    sidecar = suite.with_suffix(suite.suffix + ".manifest.json")
    manifest = json.loads(sidecar.read_text(encoding="utf-8"))
    manifest["frozen"] = False
    sidecar.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(BenchmarkError, match="must be frozen"):
        load_benchmark_suite(suite)


def test_benchmark_rejects_remote_http_endpoint(tmp_path: Path) -> None:
    suite = _write_suite(tmp_path, _records())
    with pytest.raises(BenchmarkError, match="HTTPS is required"):
        run_benchmark(
            suite,
            "http://benchmark.example.com/v1",
            "carbon-tiny",
            tmp_path / "report.json",
        )


def test_benchmark_report_cannot_overwrite_suite_or_manifest(tmp_path: Path) -> None:
    suite = _write_suite(tmp_path, _records())
    with pytest.raises(BenchmarkError, match="must be outside"):
        run_benchmark(
            suite,
            "http://localhost:8000/v1",
            "carbon-tiny",
            suite,
            client=httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(200))),
        )


def test_freeze_benchmark_suite_validates_and_creates_immutable_manifest(
    tmp_path: Path,
) -> None:
    suite = tmp_path / "raw-suite.jsonl"
    suite.write_text(
        "".join(json.dumps(record) + "\n" for record in _records()),
        encoding="utf-8",
    )
    manifest = freeze_benchmark_suite(
        suite,
        "rusaai-frozen",
        "1.2.0",
        ["CC0-1.0"],
        source_uri="https://example.org/benchmark-source",
    )
    loaded, records = load_benchmark_suite(suite)
    assert manifest["frozen"] is True
    assert manifest["record_count"] == 2
    assert loaded["suite_id"] == "rusaai-frozen"
    assert loaded["suite_version"] == "1.2.0"
    assert len(records) == 2


def test_freeze_refuses_to_replace_an_existing_suite_version(tmp_path: Path) -> None:
    suite = _write_suite(tmp_path, _records())
    with pytest.raises(BenchmarkError, match="already exists"):
        freeze_benchmark_suite(suite, "rusaai", "1.0.0", ["CC0-1.0"])


def test_freeze_rejects_duplicate_prompts_and_unsafe_source_uri(tmp_path: Path) -> None:
    suite = tmp_path / "duplicate.jsonl"
    records = _records()
    records[1]["messages"] = records[0]["messages"]
    suite.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    with pytest.raises(BenchmarkError, match="duplicate prompts"):
        freeze_benchmark_suite(suite, "rusaai", "1.0.0", ["CC0-1.0"])
    other_directory = tmp_path / "other"
    other_directory.mkdir()
    other_suite = other_directory / "raw-suite.jsonl"
    other_suite.write_text(
        "".join(json.dumps(record) + "\n" for record in _records()),
        encoding="utf-8",
    )
    with pytest.raises(BenchmarkError, match="credentials"):
        freeze_benchmark_suite(
            other_suite,
            "rusaai",
            "1.0.0",
            ["CC0-1.0"],
            source_uri="https://user:secret@example.org/benchmark",
        )


def _write_report(
    path: Path,
    *,
    suite_sha: str = "a" * 64,
    model_id: str = "model",
    accuracy: float = 0.9,
    latency: float = 100.0,
    results: list[dict] | None = None,
) -> Path:
    if results is None:
        passed_count = round(accuracy * 100)
        results = [
            {
                "record_id": f"language-{index:03d}",
                "category": "language",
                "scorer": "contains",
                "score": index < passed_count - 1,
            }
            for index in range(99)
        ]
        results.append(
            {
                "record_id": "safety-001",
                "category": "safety",
                "scorer": "contains",
                "score": True,
            }
        )
    scored = [item for item in results if item["score"] is not None]
    passed = sum(item["score"] is True for item in scored)
    failed = sum(item["score"] is False for item in scored)
    failed_requests = sum(bool(item.get("error")) for item in results)
    manual_review = sum(item["score"] is None and not item.get("error") for item in results)
    report = {
        "schema_version": "1.0.0",
        "benchmark": {
            "suite_id": "rusaai-core",
            "suite_version": "1.0.0",
            "file_sha256": suite_sha,
        },
        "model": {"id": model_id, "artifact_sha256": "b" * 64},
        "summary": {
            "total": len(results),
            "scored": len(scored),
            "passed": passed,
            "failed": failed,
            "manual_review": manual_review,
            "failed_requests": failed_requests,
            "accuracy": round(passed / len(scored), 4) if scored else None,
        },
        "latency_ms": {"mean": latency},
        "results": results,
    }
    path.write_text(json.dumps(report), encoding="utf-8")
    return path

def test_benchmark_comparison_passes_within_accuracy_and_latency_gates(
    tmp_path: Path,
) -> None:
    baseline = _write_report(tmp_path / "baseline.json", accuracy=0.9, latency=100.0)
    candidate = _write_report(
        tmp_path / "candidate.json",
        model_id="model-q4",
        accuracy=0.89,
        latency=150.0,
        results=[
            {"record_id": "language-1", "category": "language", "scorer": "contains", "score": True},
            {"record_id": "safety-1", "category": "safety", "scorer": "contains", "score": False},
        ],
    )
    result = compare_benchmark_reports(
        baseline,
        candidate,
        tmp_path / "comparison.json",
        max_accuracy_drop=0.02,
        max_category_drop=0.05,
        max_latency_ratio=2.0,
    )
    assert result["release_gate"]["passed"] is True
    assert result["comparison"]["accuracy_drop"] == 0.01
    assert result["comparison"]["latency_ratio"] == 1.5


def test_benchmark_comparison_blocks_protected_category_regression(
    tmp_path: Path,
) -> None:
    baseline = _write_report(
        tmp_path / "baseline.json",
        results=[
            {"record_id": "language-1", "category": "language", "scorer": "contains", "score": True},
            {"record_id": "safety-1", "category": "safety", "scorer": "contains", "score": True},
        ],
    )
    candidate = _write_report(
        tmp_path / "candidate.json",
        results=[
            {"record_id": "language-1", "category": "language", "scorer": "contains", "score": True},
            {"record_id": "safety-1", "category": "safety", "scorer": "contains", "score": False},
        ],
    )
    result = compare_benchmark_reports(
        baseline,
        candidate,
        tmp_path / "comparison.json",
        max_accuracy_drop=0.5,
        max_category_drop=1.0,
    )
    assert result["release_gate"]["passed"] is False
    assert "protected_category_regression" in result["release_gate"]["failed_gates"]


def test_benchmark_comparison_rejects_different_frozen_suites(tmp_path: Path) -> None:
    baseline = _write_report(tmp_path / "baseline.json")
    candidate = _write_report(tmp_path / "candidate.json", suite_sha="c" * 64)
    with pytest.raises(BenchmarkComparisonError, match="different frozen benchmark suites"):
        compare_benchmark_reports(baseline, candidate, tmp_path / "comparison.json")


def test_benchmark_comparison_rejects_changed_record_contract(tmp_path: Path) -> None:
    baseline = _write_report(
        tmp_path / "baseline.json",
        results=[
            {"record_id": "language-1", "category": "language", "scorer": "contains", "score": True},
            {"record_id": "safety-1", "category": "safety", "scorer": "contains", "score": False},
        ],
    )
    candidate = _write_report(
        tmp_path / "candidate.json",
        results=[
            {"record_id": "language-1", "category": "coding", "scorer": "contains", "score": True},
            {"record_id": "safety-1", "category": "safety", "scorer": "contains", "score": False},
        ],
    )
    with pytest.raises(BenchmarkComparisonError, match="record contract changed"):
        compare_benchmark_reports(baseline, candidate, tmp_path / "comparison.json")
