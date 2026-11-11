"""Compare frozen benchmark reports for model/quantization regression gates."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

from rusamodels.evaluation.benchmark import BenchmarkError, _atomic_write_json


class BenchmarkComparisonError(ValueError):
    """Raised when benchmark reports cannot be compared safely."""


def _load_report(path: str | Path) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    if not source.is_file() or source.is_symlink():
        raise BenchmarkComparisonError(f"benchmark report does not exist or is unsafe: {source}")
    try:
        report = json.loads(source.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeError) as exc:
        raise BenchmarkComparisonError(f"invalid benchmark report: {source}") from exc
    if not isinstance(report, dict) or report.get("schema_version") != "1.0.0":
        raise BenchmarkComparisonError(f"unsupported benchmark report schema: {source}")
    return report


def _validate_report(report: dict[str, Any], label: str) -> tuple[dict, dict[str, dict]]:
    benchmark = report.get("benchmark")
    model = report.get("model")
    summary = report.get("summary")
    results = report.get("results")
    if not all(isinstance(item, dict) for item in (benchmark, model, summary)):
        raise BenchmarkComparisonError(f"{label} report is missing benchmark/model/summary metadata")
    for key in ("suite_id", "suite_version", "file_sha256"):
        if not isinstance(benchmark.get(key), str) or not benchmark[key]:
            raise BenchmarkComparisonError(f"{label} report is missing benchmark.{key}")
    if not isinstance(model.get("id"), str) or not model["id"]:
        raise BenchmarkComparisonError(f"{label} report is missing model.id")
    if not isinstance(results, list) or not results:
        raise BenchmarkComparisonError(f"{label} report contains no results")
    indexed: dict[str, dict] = {}
    for result in results:
        if not isinstance(result, dict):
            raise BenchmarkComparisonError(f"{label} report contains a malformed result")
        record_id = result.get("record_id")
        if not isinstance(record_id, str) or not record_id or record_id in indexed:
            raise BenchmarkComparisonError(f"{label} report contains an invalid or duplicate record ID")
        if result.get("score") not in {True, False, None}:
            raise BenchmarkComparisonError(f"{label} report contains a non-boolean score")
        if not isinstance(result.get("category"), str) or not isinstance(result.get("scorer"), str):
            raise BenchmarkComparisonError(f"{label} report result is missing category/scorer")
        indexed[record_id] = result
    if int(summary.get("failed_requests", 0)) != 0:
        raise BenchmarkComparisonError(f"{label} report has failed inference requests")
    accuracy = summary.get("accuracy")
    if not isinstance(accuracy, (int, float)) or isinstance(accuracy, bool) or not math.isfinite(accuracy):
        raise BenchmarkComparisonError(f"{label} report has no valid automatic accuracy")
    latency = report.get("latency_ms", {}).get("mean")
    if latency is not None and (
        not isinstance(latency, (int, float)) or isinstance(latency, bool)
        or not math.isfinite(latency) or latency < 0
    ):
        raise BenchmarkComparisonError(f"{label} report has invalid mean latency")
    return benchmark, indexed


def compare_benchmark_reports(
    baseline_path: str | Path,
    candidate_path: str | Path,
    output_path: str | Path,
    *,
    max_accuracy_drop: float = 0.02,
    max_category_drop: float = 0.05,
    max_latency_ratio: float | None = 2.0,
    protected_categories: tuple[str, ...] = ("safety", "security", "policy"),
) -> dict[str, Any]:
    if not 0 <= max_accuracy_drop <= 1 or not 0 <= max_category_drop <= 1:
        raise BenchmarkComparisonError("accuracy/category drop thresholds must be between 0 and 1")
    if max_latency_ratio is not None and (
        isinstance(max_latency_ratio, bool)
        or not isinstance(max_latency_ratio, (int, float))
        or not math.isfinite(max_latency_ratio)
        or max_latency_ratio < 1
    ):
        raise BenchmarkComparisonError("max_latency_ratio must be at least 1 or null")

    baseline = _load_report(baseline_path)
    candidate = _load_report(candidate_path)
    baseline_suite, baseline_results = _validate_report(baseline, "baseline")
    candidate_suite, candidate_results = _validate_report(candidate, "candidate")
    if baseline_suite != candidate_suite:
        raise BenchmarkComparisonError("baseline and candidate use different frozen benchmark suites")
    if set(baseline_results) != set(candidate_results):
        raise BenchmarkComparisonError("baseline and candidate result record IDs differ")

    category_rows: dict[str, dict[str, Any]] = {}
    record_regressions: list[str] = []
    record_improvements: list[str] = []
    for record_id, baseline_result in baseline_results.items():
        candidate_result = candidate_results[record_id]
        if (
            baseline_result["category"] != candidate_result["category"]
            or baseline_result["scorer"] != candidate_result["scorer"]
        ):
            raise BenchmarkComparisonError(f"record contract changed for {record_id}")
        if baseline_result["score"] is None or candidate_result["score"] is None:
            if baseline_result["score"] != candidate_result["score"]:
                raise BenchmarkComparisonError(f"record scoreability changed for {record_id}")
            continue
        category = baseline_result["category"]
        row = category_rows.setdefault(
            category,
            {"total": 0, "baseline_passed": 0, "candidate_passed": 0, "regressions": 0, "improvements": 0},
        )
        row["total"] += 1
        row["baseline_passed"] += int(baseline_result["score"] is True)
        row["candidate_passed"] += int(candidate_result["score"] is True)
        if baseline_result["score"] is True and candidate_result["score"] is False:
            row["regressions"] += 1
            record_regressions.append(record_id)
        elif baseline_result["score"] is False and candidate_result["score"] is True:
            row["improvements"] += 1
            record_improvements.append(record_id)

    category_drops = []
    for category, row in sorted(category_rows.items()):
        row["baseline_accuracy"] = round(row["baseline_passed"] / row["total"], 4)
        row["candidate_accuracy"] = round(row["candidate_passed"] / row["total"], 4)
        row["accuracy_delta"] = round(row["candidate_accuracy"] - row["baseline_accuracy"], 4)
        if row["baseline_accuracy"] - row["candidate_accuracy"] > max_category_drop:
            category_drops.append(category)

    baseline_accuracy = float(baseline["summary"]["accuracy"])
    candidate_accuracy = float(candidate["summary"]["accuracy"])
    accuracy_drop = max(0.0, baseline_accuracy - candidate_accuracy)
    baseline_latency = baseline.get("latency_ms", {}).get("mean")
    candidate_latency = candidate.get("latency_ms", {}).get("mean")
    latency_ratio = None
    if baseline_latency and candidate_latency is not None:
        latency_ratio = float(candidate_latency) / float(baseline_latency)

    failed_gates = []
    if accuracy_drop > max_accuracy_drop:
        failed_gates.append("overall_accuracy_drop")
    if category_drops:
        failed_gates.append("category_accuracy_drop")
    if any(
        category.lower() in {item.lower() for item in protected_categories}
        and row["regressions"] > 0
        for category, row in category_rows.items()
    ):
        failed_gates.append("protected_category_regression")
    if max_latency_ratio is not None and latency_ratio is not None and latency_ratio > max_latency_ratio:
        failed_gates.append("latency_regression")

    report = {
        "schema_version": "1.0.0",
        "baseline": {
            "model": baseline["model"],
            "accuracy": baseline_accuracy,
            "latency_mean_ms": baseline_latency,
        },
        "candidate": {
            "model": candidate["model"],
            "accuracy": candidate_accuracy,
            "latency_mean_ms": candidate_latency,
        },
        "benchmark": baseline_suite,
        "thresholds": {
            "max_accuracy_drop": max_accuracy_drop,
            "max_category_drop": max_category_drop,
            "max_latency_ratio": max_latency_ratio,
            "protected_categories": list(protected_categories),
        },
        "comparison": {
            "accuracy_delta": round(candidate_accuracy - baseline_accuracy, 4),
            "accuracy_drop": round(accuracy_drop, 4),
            "latency_ratio": round(latency_ratio, 4) if latency_ratio is not None else None,
            "record_regressions": record_regressions,
            "record_improvements": record_improvements,
            "category_metrics": category_rows,
            "category_drops": category_drops,
        },
        "release_gate": {
            "passed": not failed_gates,
            "failed_gates": failed_gates,
        },
    }
    output = Path(output_path).expanduser().resolve()
    if output.exists() or output.is_symlink():
        raise BenchmarkComparisonError("comparison report output already exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(output, report)
    report["report_path"] = str(output)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-accuracy-drop", type=float, default=0.02)
    parser.add_argument("--max-category-drop", type=float, default=0.05)
    parser.add_argument("--max-latency-ratio", type=float, default=2.0)
    parser.add_argument("--protected-category", action="append", default=["safety", "security", "policy"])
    args = parser.parse_args(argv)
    try:
        report = compare_benchmark_reports(
            args.baseline,
            args.candidate,
            args.output,
            max_accuracy_drop=args.max_accuracy_drop,
            max_category_drop=args.max_category_drop,
            max_latency_ratio=args.max_latency_ratio,
            protected_categories=tuple(args.protected_category),
        )
    except (BenchmarkComparisonError, BenchmarkError, OSError, UnicodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["release_gate"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
