#!/usr/bin/env python3
"""Validate a versioned RusaModels architecture blueprint and optionally instantiate it."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rusamodels.architecture import (  # noqa: E402
    build_model,
    load_architecture_blueprint,
    validate_architecture_config,
)
from rusamodels.architecture.config import ArchitectureConfigError  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--instantiate",
        action="store_true",
        help="Construct the model after checking the parameter safety limit.",
    )
    parser.add_argument("--max-parameters", type=int, default=120_000_000)
    args = parser.parse_args(argv)
    try:
        blueprint = load_architecture_blueprint(args.config)
        report = {
            "model_id": blueprint.model_id,
            "model_version": blueprint.model_version,
            "status": blueprint.status,
            **validate_architecture_config(blueprint.architecture),
            "instantiated": False,
        }
        if args.instantiate:
            model = build_model(
                blueprint.architecture,
                max_parameters=args.max_parameters,
            )
            report["instantiated"] = True
            report["actual_parameters"] = sum(
                parameter.numel()
                for parameter in model.parameters()
                if parameter.requires_grad
            )
    except (ArchitectureConfigError, OSError, ValueError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
