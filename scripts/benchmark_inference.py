#!/usr/bin/env python3
"""Reproducible local/staging benchmark for the RusaModels OpenAI-compatible Model Server API.

Remote targets require --allow-remote-staging. Production targets are not supported.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import httpx


def percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * quantile)))
    return round(ordered[index], 4)


def git_revision() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def hardware_metadata() -> dict[str, object]:
    result: dict[str, object] = {"platform": platform.platform(), "python": sys.version.split()[0]}
    try:
        import torch

        result["torch"] = torch.__version__
        result["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            result["gpus"] = [
                {
                    "name": torch.cuda.get_device_name(index),
                    "total_memory_bytes": torch.cuda.get_device_properties(index).total_memory,
                }
                for index in range(torch.cuda.device_count())
            ]
    except ImportError:
        result["torch"] = None
    return result


async def stream_request(
    client: httpx.AsyncClient,
    url: str,
    headers: dict[str, str],
    payload: dict[str, object],
) -> dict[str, object]:
    started = time.perf_counter()
    first_token: float | None = None
    usage: dict[str, int] = {}
    chunks = 0
    async with client.stream("POST", url, headers=headers, json=payload) as response:
        if response.status_code != 200:
            body = (await response.aread()).decode("utf-8", errors="replace")[:500]
            raise RuntimeError(f"HTTP {response.status_code}: {body}")
        async for line in response.aiter_lines():
            if not line.startswith("data: "):
                continue
            data = line[6:]
            if data == "[DONE]":
                break
            event = json.loads(data)
            if "error" in event:
                raise RuntimeError(str(event["error"]))
            if "usage" in event and event["usage"]:
                usage = event["usage"]
            choices = event.get("choices", [])
            if choices:
                delta = choices[0].get("delta", {}).get("content", "")
                if delta:
                    chunks += 1
                    if first_token is None:
                        first_token = time.perf_counter() - started
    elapsed = time.perf_counter() - started
    return {
        "latency_seconds": elapsed,
        "ttft_seconds": first_token,
        "completion_tokens": usage.get("completion_tokens", 0),
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "chunks": chunks,
    }


async def run(args: argparse.Namespace) -> dict[str, object]:
    api_key = os.getenv("RUSAMODELS_API_TOKEN")
    if not api_key:
        raise SystemExit("RUSA_API_KEY must be set in the environment")
    parsed = urlparse(args.base_url)
    local_hosts = {"localhost", "127.0.0.1", "::1"}
    if parsed.hostname not in local_hosts and not args.allow_remote_staging:
        raise SystemExit("Remote benchmarks require --allow-remote-staging; production is not supported")
    if args.allow_remote_staging and args.environment != "staging":
        raise SystemExit("Remote benchmark target must explicitly use --environment staging")

    headers = {"Authorization": f"Bearer {api_key}"}
    health_url = args.base_url.rstrip("/") + "/health/ready"
    chat_url = args.base_url.rstrip("/") + "/openai/v1/chat/completions"
    timeout = httpx.Timeout(args.timeout)
    limits = httpx.Limits(max_connections=args.concurrency, max_keepalive_connections=args.concurrency)
    semaphore = asyncio.Semaphore(args.concurrency)
    prompts = [
        "Jelaskan fungsi utama sistem pernapasan manusia secara ringkas.",
        "Tuliskan tiga manfaat menjaga kebersihan lingkungan sekolah.",
        "Explain the difference between authentication and authorization.",
        "Buat ringkasan singkat tentang energi terbarukan.",
    ]

    async with httpx.AsyncClient(timeout=timeout, limits=limits) as client:
        health = await client.get(health_url)
        if health.status_code != 200:
            raise SystemExit(f"Target is not ready: HTTP {health.status_code} {health.text[:300]}")

        async def one(index: int) -> dict[str, object]:
            async with semaphore:
                payload = {
                    "model": args.model,
                    "messages": [{"role": "user", "content": prompts[index % len(prompts)]}],
                    "max_tokens": args.max_tokens,
                    "temperature": 0,
                    "stream": True,
                }
                try:
                    result = await stream_request(client, chat_url, headers, payload)
                    result["ok"] = True
                    return result
                except Exception as exc:
                    return {"ok": False, "error": type(exc).__name__, "message": str(exc)[:300]}

        wall_started = time.perf_counter()
        results = await asyncio.gather(*(one(index) for index in range(args.requests)))
        wall_seconds = time.perf_counter() - wall_started

    successes = [item for item in results if item.get("ok")]
    latencies = [float(item["latency_seconds"]) for item in successes]
    ttfts = [float(item["ttft_seconds"]) for item in successes if item.get("ttft_seconds") is not None]
    completion_tokens = sum(int(item.get("completion_tokens", 0)) for item in successes)
    prompt_tokens = sum(int(item.get("prompt_tokens", 0)) for item in successes)
    report = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "target_environment": args.environment,
        "base_url": args.base_url,
        "model": args.model,
        "git_revision": git_revision(),
        "benchmark_client_hardware": hardware_metadata(),
        "server_readiness": health.json(),
        "workload": {
            "requests": args.requests,
            "concurrency": args.concurrency,
            "max_tokens": args.max_tokens,
            "stream": True,
            "warmup_requests": 0,
        },
        "results": {
            "successful_requests": len(successes),
            "failed_requests": len(results) - len(successes),
            "error_rate": round((len(results) - len(successes)) / len(results), 6),
            "wall_seconds": round(wall_seconds, 4),
            "request_throughput_per_second": round(len(successes) / wall_seconds, 4) if wall_seconds else 0,
            "completion_tokens": completion_tokens,
            "prompt_tokens": prompt_tokens,
            "completion_tokens_per_second": round(completion_tokens / wall_seconds, 4) if wall_seconds else 0,
            "latency_seconds": {
                "p50": percentile(latencies, 0.50),
                "p95": percentile(latencies, 0.95),
                "p99": percentile(latencies, 0.99),
                "mean": round(statistics.mean(latencies), 4) if latencies else None,
            },
            "ttft_seconds": {
                "p50": percentile(ttfts, 0.50),
                "p95": percentile(ttfts, 0.95),
                "p99": percentile(ttfts, 0.99),
            },
        },
        "failures": [item for item in results if not item.get("ok")][:20],
        "limitations": [
            "This is a bounded smoke benchmark, not a sustained production load or soak test.",
            "No warmup requests are included; compare runs with identical hardware, model, prompts, and settings.",
            "Token throughput uses usage reported by the serving API.",
        ],
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default=os.getenv("RUSAMODELS_INFERENCE_MODEL_ID", "rusa-general-4b"))
    parser.add_argument("--requests", type=int, default=20)
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--environment", choices=("local", "staging"), default="local")
    parser.add_argument("--allow-remote-staging", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("benchmark-report.json"))
    args = parser.parse_args()
    if not 1 <= args.requests <= 1000 or not 1 <= args.concurrency <= 128:
        parser.error("requests must be 1..1000 and concurrency must be 1..128")
    if not 1 <= args.max_tokens <= 4096 or args.timeout <= 0:
        parser.error("max-tokens must be 1..4096 and timeout must be positive")

    report = asyncio.run(run(args))
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"\nReport written to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
