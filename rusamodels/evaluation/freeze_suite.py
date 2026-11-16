"""CLI for validating and freezing a versioned benchmark suite."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from rusamodels.evaluation.benchmark import BenchmarkError, freeze_benchmark_suite


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", required=True, type=Path)
    parser.add_argument("--suite-id", required=True)
    parser.add_argument("--suite-version", required=True)
    parser.add_argument("--license-id", action="append", required=True)
    parser.add_argument("--source-uri")
    args = parser.parse_args(argv)
    try:
        manifest = freeze_benchmark_suite(
            args.suite,
            args.suite_id,
            args.suite_version,
            args.license_id,
            source_uri=args.source_uri,
        )
    except (BenchmarkError, OSError, UnicodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
