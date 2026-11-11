"""Versioned OpenAI-compatible benchmark runner for RusaModels model releases."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import httpx


class BenchmarkError(ValueError):
    """Raised when a benchmark suite or run configuration is invalid."""


_SCORERS = {"exact_match", "contains", "regex", "contains_all", "contains_any", "json_valid", "manual"}
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_ALLOWED_ROLES = {"system", "user", "assistant"}


def _sha256_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def _object_pairs_without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _canonical_digest(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _normalize(value: str) -> str:
    return " ".join(value.split()).casefold()


def _validate_record(record: Any, line_number: int) -> dict[str, Any]:
    if not isinstance(record, dict):
        raise BenchmarkError(f"benchmark.jsonl:{line_number}: record must be an object")
    record_id = record.get("id")
    category = record.get("category")
    messages = record.get("messages")
    scorer = record.get("scorer", "manual")
    if not isinstance(record_id, str) or not record_id.strip() or len(record_id) > 256:
        raise BenchmarkError(f"benchmark.jsonl:{line_number}: invalid record id")
    if not isinstance(category, str) or not category.strip() or len(category) > 128:
        raise BenchmarkError(f"benchmark.jsonl:{line_number}: invalid category")
    if scorer not in _SCORERS:
        raise BenchmarkError(f"benchmark.jsonl:{line_number}: unsupported scorer {scorer!r}")
    if not isinstance(messages, list) or not messages:
        raise BenchmarkError(f"benchmark.jsonl:{line_number}: messages must be a non-empty array")
    has_user = False
    clean_messages = []
    for index, message in enumerate(messages):
        if not isinstance(message, dict) or message.get("role") not in _ALLOWED_ROLES:
            raise BenchmarkError(
                f"benchmark.jsonl:{line_number}: messages[{index}] has an invalid role"
            )
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            raise BenchmarkError(
                f"benchmark.jsonl:{line_number}: messages[{index}].content must be non-empty text"
            )
        has_user = has_user or message["role"] == "user"
        clean_messages.append({"role": message["role"], "content": content})
    if not has_user:
        raise BenchmarkError(f"benchmark.jsonl:{line_number}: at least one user message is required")

    expected = record.get("expected")
    if scorer in {"exact_match", "contains", "regex"} and (
        not isinstance(expected, str) or not expected
    ):
        raise BenchmarkError(f"benchmark.jsonl:{line_number}: scorer requires a string expected value")
    if scorer in {"contains_all", "contains_any"} and (
        not isinstance(expected, list)
        or not expected
        or any(not isinstance(item, str) or not item for item in expected)
    ):
        raise BenchmarkError(f"benchmark.jsonl:{line_number}: scorer requires a non-empty string array")
    if scorer == "regex":
        try:
            re.compile(expected, re.IGNORECASE | re.MULTILINE)
        except re.error as exc:
            raise BenchmarkError(f"benchmark.jsonl:{line_number}: invalid regex: {exc}") from exc
    return {
        "id": record_id,
        "category": category,
        "messages": clean_messages,
        "scorer": scorer,
        "expected": expected,
    }


def load_benchmark_suite(suite_path: str | Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    raw_path = Path(suite_path).expanduser()
    if raw_path.is_symlink():
        raise BenchmarkError("benchmark suite must not be a symlink")
    path = raw_path.resolve()
    sidecar = path.with_suffix(path.suffix + ".manifest.json")
    if sidecar.is_symlink() or not path.is_file() or not sidecar.is_file():
        raise BenchmarkError("benchmark suite and its frozen manifest must both exist as regular files")
    try:
        manifest = json.loads(sidecar.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeError) as exc:
        raise BenchmarkError("benchmark manifest is invalid JSON") from exc
    if not isinstance(manifest, dict) or manifest.get("schema_version") != "1.0.0":
        raise BenchmarkError("unsupported benchmark manifest schema")
    if manifest.get("frozen") is not True:
        raise BenchmarkError("benchmark suite must be frozen before evaluation")
    for key in ("suite_id", "suite_version", "file_name", "file_sha256"):
        if not isinstance(manifest.get(key), str) or not manifest[key]:
            raise BenchmarkError(f"benchmark manifest is missing {key}")
    if manifest["file_name"] != path.name or not _SHA256.fullmatch(manifest["file_sha256"]):
        raise BenchmarkError("benchmark manifest file identity or digest is invalid")
    digest, size = _sha256_file(path)
    if digest != manifest["file_sha256"] or manifest.get("size_bytes") != size:
        raise BenchmarkError("benchmark suite checksum or size does not match its frozen manifest")
    licenses = manifest.get("license_ids")
    if (
        not isinstance(licenses, list)
        or not licenses
        or any(not isinstance(item, str) or not item.strip() for item in licenses)
        or licenses != sorted(set(licenses))
    ):
        raise BenchmarkError("benchmark manifest must contain sorted, declared license_ids")
    if type(manifest.get("record_count")) is not int or manifest["record_count"] < 1:
        raise BenchmarkError("benchmark manifest has invalid record_count")

    records: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    seen_prompts: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise BenchmarkError(f"benchmark.jsonl:{line_number}: blank lines are not allowed")
            try:
                raw = json.loads(
                    line,
                    parse_constant=lambda value: (_ for _ in ()).throw(
                        ValueError(f"non-standard JSON constant: {value}")
                    ),
                    object_pairs_hook=_object_pairs_without_duplicates,
                )
            except (json.JSONDecodeError, ValueError) as exc:
                raise BenchmarkError(f"benchmark.jsonl:{line_number}: invalid JSON") from exc
            record = _validate_record(raw, line_number)
            if record["id"] in seen_ids:
                raise BenchmarkError(f"benchmark suite contains duplicate id: {record['id']}")
            prompt_digest = _canonical_digest(record["messages"])
            if prompt_digest in seen_prompts:
                raise BenchmarkError("benchmark suite contains duplicate prompts with different IDs")
            seen_ids.add(record["id"])
            seen_prompts.add(prompt_digest)
            records.append(record)
    if len(records) != manifest["record_count"]:
        raise BenchmarkError("benchmark record count does not match frozen manifest")
    return {
        "suite_id": manifest["suite_id"],
        "suite_version": manifest["suite_version"],
        "manifest_sha256": _sha256_file(sidecar)[0],
        "file_sha256": digest,
        "license_ids": licenses,
    }, records


def _score(prediction: str, scorer: str, expected: Any) -> bool | None:
    if scorer == "manual":
        return None
    if scorer == "json_valid":
        try:
            json.loads(prediction)
            return True
        except json.JSONDecodeError:
            return False
    if scorer == "exact_match":
        return _normalize(prediction) == _normalize(expected)
    if scorer == "contains":
        return _normalize(expected) in _normalize(prediction)
    if scorer == "regex":
        return re.search(expected, prediction, re.IGNORECASE | re.MULTILINE) is not None
    if scorer == "contains_all":
        normalized = _normalize(prediction)
        return all(_normalize(item) in normalized for item in expected)
    if scorer == "contains_any":
        normalized = _normalize(prediction)
        return any(_normalize(item) in normalized for item in expected)
    raise BenchmarkError(f"unsupported scorer: {scorer}")


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((percentile / 100) * (len(ordered) - 1)))))
    return round(ordered[index], 3)


def _aggregate(results: list[dict[str, Any]], field: str) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for result in results:
        groups.setdefault(str(result[field]), []).append(result)
    summary = {}
    for key, items in sorted(groups.items()):
        scored = [item for item in items if item["score"] is not None]
        passed = sum(item["score"] is True for item in scored)
        summary[key] = {
            "total": len(items),
            "scored": len(scored),
            "passed": passed,
            "failed": sum(item["score"] is False for item in scored),
            "accuracy": round(passed / len(scored), 4) if scored else None,
        }
    return summary


def _completion(
    client: httpx.Client,
    url: str,
    model: str,
    messages: list[dict[str, str]],
    headers: dict[str, str],
    max_tokens: int,
    max_retries: int,
) -> tuple[str, str | None, dict[str, Any], int]:
    retry_statuses = {429, 500, 502, 503, 504}
    last_error = "request_failed"
    for attempt in range(max_retries + 1):
        try:
            response = client.post(
                url,
                json={
                    "model": model,
                    "messages": messages,
                    "temperature": 0,
                    "top_p": 1,
                    "max_tokens": max_tokens,
                    "stream": False,
                },
                headers=headers,
            )
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            last_error = type(exc).__name__
            if attempt < max_retries:
                time.sleep(min(2**attempt, 4))
                continue
            raise BenchmarkError(last_error) from exc
        if response.status_code in retry_statuses and attempt < max_retries:
            time.sleep(min(2**attempt, 4))
            continue
        if response.status_code >= 400:
            raise BenchmarkError(f"http_status_{response.status_code}")
        try:
            payload = response.json()
            choice = payload["choices"][0]
            message = choice["message"]
            content = message.get("content")
            if isinstance(content, list):
                content = "".join(
                    item.get("text", "")
                    for item in content
                    if isinstance(item, dict) and isinstance(item.get("text", ""), str)
                )
            if not isinstance(content, str):
                raise ValueError("completion content is not text")
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise BenchmarkError("invalid_openai_compatible_response") from exc
        usage = payload.get("usage", {})
        if not isinstance(usage, dict):
            usage = {}
        return content, choice.get("finish_reason"), usage, response.status_code
    raise BenchmarkError(last_error)


def run_benchmark(
    suite_path: str | Path,
    base_url: str,
    model: str,
    output_path: str | Path,
    *,
    api_key: str | None = None,
    timeout: float = 60.0,
    max_tokens: int = 256,
    max_retries: int = 2,
    model_artifact_sha256: str | None = None,
    runtime_profile: str | None = None,
    min_accuracy: float | None = None,
    client: httpx.Client | None = None,
    completion_path: str = "chat/completions",
) -> dict[str, Any]:
    if not isinstance(model, str) or not model.strip():
        raise BenchmarkError("model must be a non-empty string")
    if not isinstance(base_url, str) or not base_url.strip():
        raise BenchmarkError("base_url must be a non-empty URL")
    parsed_url = urlsplit(base_url)
    if parsed_url.scheme not in {"http", "https"} or not parsed_url.netloc:
        raise BenchmarkError("base_url must be an absolute HTTP(S) URL")
    if parsed_url.username or parsed_url.password or parsed_url.query or parsed_url.fragment:
        raise BenchmarkError("base_url must not contain credentials, query, or fragment")
    if not isinstance(completion_path, str) or not completion_path.strip():
        raise BenchmarkError("completion_path must be a non-empty relative path")
    parsed_completion_path = urlsplit(completion_path)
    if (
        parsed_completion_path.scheme
        or parsed_completion_path.netloc
        or parsed_completion_path.query
        or parsed_completion_path.fragment
        or ".." in Path(completion_path).parts
    ):
        raise BenchmarkError("completion_path must be a safe relative endpoint path")
    if type(max_tokens) is not int or max_tokens < 1:
        raise BenchmarkError("max_tokens must be a positive integer")
    if type(max_retries) is not int or not 0 <= max_retries <= 5:
        raise BenchmarkError("max_retries must be between 0 and 5")
    if (
        isinstance(timeout, bool)
        or not isinstance(timeout, (int, float))
        or not 0 < timeout < float("inf")
    ):
        raise BenchmarkError("timeout must be finite and positive")
    if min_accuracy is not None and (
        isinstance(min_accuracy, bool) or not isinstance(min_accuracy, (int, float))
        or not 0 <= min_accuracy <= 1
    ):
        raise BenchmarkError("min_accuracy must be between 0 and 1")
    if model_artifact_sha256 is not None and not _SHA256.fullmatch(model_artifact_sha256):
        raise BenchmarkError("model_artifact_sha256 must be a lowercase SHA-256 digest")

    suite_metadata, records = load_benchmark_suite(suite_path)
    suite_file = Path(suite_path).expanduser().resolve()
    output = Path(output_path).expanduser()
    if output.is_symlink():
        raise BenchmarkError("benchmark report path must not be a symlink")
    output = output.resolve()
    suite_manifest_file = suite_file.with_suffix(suite_file.suffix + ".manifest.json")
    if suite_file.parent in output.parents or output in {suite_file, suite_manifest_file}:
        raise BenchmarkError("benchmark report must be outside the immutable suite directory")
    if output.exists():
        raise BenchmarkError("benchmark report already exists; choose a new output path")
    output.parent.mkdir(parents=True, exist_ok=True)

    url = base_url.rstrip("/") + "/" + completion_path.lstrip("/")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    owns_client = client is None
    session = client or httpx.Client(timeout=timeout)
    started_at = datetime.now(UTC).isoformat()
    results: list[dict[str, Any]] = []
    latencies: list[float] = []
    total_prompt_tokens = 0
    total_completion_tokens = 0
    failed_requests = 0
    try:
        for record in records:
            prompt_sha256 = _canonical_digest(record["messages"])
            request_id = hashlib.sha256(record["id"].encode()).hexdigest()[:32]
            request_headers = {**headers, "X-Request-ID": request_id}
            started = time.perf_counter()
            try:
                prediction, finish_reason, usage, status_code = _completion(
                    session,
                    url,
                    model,
                    record["messages"],
                    request_headers,
                    max_tokens,
                    max_retries,
                )
                latency_ms = (time.perf_counter() - started) * 1000
                latencies.append(latency_ms)
                score = _score(prediction, record["scorer"], record.get("expected"))
                try:
                    prompt_tokens = max(0, int(usage.get("prompt_tokens", 0) or 0))
                except (TypeError, ValueError):
                    prompt_tokens = 0
                try:
                    completion_tokens = max(0, int(usage.get("completion_tokens", 0) or 0))
                except (TypeError, ValueError):
                    completion_tokens = 0
                total_prompt_tokens += max(prompt_tokens, 0)
                total_completion_tokens += max(completion_tokens, 0)
                result = {
                    "record_id": record["id"],
                    "category": record["category"],
                    "scorer": record["scorer"],
                    "prompt_sha256": prompt_sha256,
                    "expected": record.get("expected"),
                    "prediction": prediction[:4000],
                    "prediction_sha256": hashlib.sha256(prediction.encode()).hexdigest(),
                    "score": score,
                    "latency_ms": round(latency_ms, 3),
                    "finish_reason": finish_reason,
                    "http_status": status_code,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "error": None,
                }
            except BenchmarkError as exc:
                failed_requests += 1
                latency_ms = (time.perf_counter() - started) * 1000
                result = {
                    "record_id": record["id"],
                    "category": record["category"],
                    "scorer": record["scorer"],
                    "prompt_sha256": prompt_sha256,
                    "expected": record.get("expected"),
                    "prediction": None,
                    "prediction_sha256": None,
                    "score": None,
                    "latency_ms": round(latency_ms, 3),
                    "finish_reason": None,
                    "http_status": (
                        int(str(exc).rsplit("_", 1)[-1])
                        if str(exc).startswith("http_status_")
                        else None
                    ),
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "error": str(exc),
                }
            results.append(result)
    finally:
        if owns_client:
            session.close()

    scored = [item for item in results if item["score"] is not None]
    passed = sum(item["score"] is True for item in scored)
    accuracy = passed / len(scored) if scored else None
    min_accuracy_passed = (
        accuracy >= min_accuracy
        if accuracy is not None and min_accuracy is not None
        else False if min_accuracy is not None else None
    )
    report = {
        "schema_version": "1.0.0",
        "benchmark": suite_metadata,
        "model": {
            "id": model,
            "artifact_sha256": model_artifact_sha256,
            "runtime_profile": runtime_profile,
        },
        "endpoint": f"{parsed_url.scheme}://{parsed_url.netloc}{parsed_url.path}",
        "started_at": started_at,
        "completed_at": datetime.now(UTC).isoformat(),
        "summary": {
            "total": len(results),
            "scored": len(scored),
            "passed": passed,
            "failed": sum(item["score"] is False for item in scored),
            "manual_review": sum(item["score"] is None and item["error"] is None for item in results),
            "failed_requests": failed_requests,
            "accuracy": round(accuracy, 4) if accuracy is not None else None,
            "min_accuracy_gate": min_accuracy,
            "min_accuracy_passed": min_accuracy_passed,
        },
        "latency_ms": {
            "mean": round(sum(latencies) / len(latencies), 3) if latencies else None,
            "p50": _percentile(latencies, 50),
            "p95": _percentile(latencies, 95),
        },
        "token_usage": {
            "prompt_tokens": total_prompt_tokens,
            "completion_tokens": total_completion_tokens,
        },
        "by_category": _aggregate(results, "category"),
        "by_scorer": _aggregate(results, "scorer"),
        "results": results,
    }
    _atomic_write_json(output, report)
    report["report_path"] = str(output)
    return report


def _atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    import tempfile

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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", required=True, type=Path)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--completion-path", default="chat/completions")
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--api-key-env", default="RUSAMODELS_EVAL_API_KEY")
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--model-artifact-sha256")
    parser.add_argument("--runtime-profile")
    parser.add_argument("--min-accuracy", type=float)
    parser.add_argument("--allow-http", action="store_true")
    args = parser.parse_args(argv)
    parsed = urlsplit(args.base_url)
    if parsed.scheme != "https" and not args.allow_http:
        if parsed.hostname not in {"localhost", "127.0.0.1", "::1"}:
            parser.error("HTTPS is required for non-local benchmark endpoints; use --allow-http only for development")
    api_key = os.environ.get(args.api_key_env)
    try:
        report = run_benchmark(
            args.suite,
            args.base_url,
            args.model,
            args.output,
            api_key=api_key,
            timeout=args.timeout,
            max_tokens=args.max_tokens,
            max_retries=args.max_retries,
            model_artifact_sha256=args.model_artifact_sha256,
            runtime_profile=args.runtime_profile,
            min_accuracy=args.min_accuracy,
            completion_path=args.completion_path,
        )
    except (BenchmarkError, OSError, UnicodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({key: value for key, value in report.items() if key != "results"}, indent=2))
    gate = report["summary"]["min_accuracy_passed"]
    return 1 if report["summary"]["failed_requests"] or gate is False else 0


if __name__ == "__main__":
    raise SystemExit(main())
