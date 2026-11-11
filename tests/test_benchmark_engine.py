from __future__ import annotations

import hashlib
import json
from pathlib import Path

import httpx
import pytest

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
