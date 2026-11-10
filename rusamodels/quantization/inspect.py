"""Inspect local Hugging Face model directories without executing repository code."""
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModelInspection:
    model_type: str
    architectures: tuple[str, ...]
    parameter_count: int | None
    weight_files: tuple[Path, ...]
    total_weight_bytes: int


def inspect_model_directory(model_dir: Path) -> ModelInspection:
    config_path = model_dir / "config.json"
    if not config_path.is_file():
        raise ValueError("Model directory must contain config.json")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    model_type = config.get("model_type")
    if not isinstance(model_type, str) or not model_type:
        raise ValueError("Model config is missing model_type")
    architectures = config.get("architectures") or []
    if not isinstance(architectures, list) or not all(isinstance(x, str) for x in architectures):
        raise ValueError("Model config architectures must be a list of strings")
    weight_files = tuple(sorted(
        path for path in model_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in {".safetensors", ".bin", ".pt"}
        and ".cache" not in path.parts
    ))
    if not weight_files:
        raise ValueError("No supported model weight files found")
    total_bytes = sum(path.stat().st_size for path in weight_files)
    parameter_count = config.get("num_parameters")
    if isinstance(parameter_count, str) and parameter_count.isdigit():
        parameter_count = int(parameter_count)
    if not isinstance(parameter_count, int) or parameter_count <= 0:
        parameter_count = None
    return ModelInspection(
        model_type=model_type,
        architectures=tuple(architectures),
        parameter_count=parameter_count,
        weight_files=weight_files,
        total_weight_bytes=total_bytes,
    )
