"""Export a verified JSONL dataset split to a plain-text causal-LM corpus."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
import unicodedata
from pathlib import Path
from typing import Any

from rusamodels.dataset_verification import (
    DatasetVerificationError,
    verify_dataset_release,
)


class DatasetTextExportError(ValueError):
    """Raised when a dataset split cannot be exported safely."""


_IGNORED_KEYS = {
    "id",
    "group_id",
    "metadata",
    "source",
    "source_uri",
    "license_id",
    "created_at",
    "updated_at",
}
_ROLE = re.compile(r"[^a-zA-Z0-9_-]+")


def _normalize_text(value: str) -> str:
    return unicodedata.normalize("NFC", value.replace("\r\n", "\n").replace("\r", "\n")).strip()


def _flatten_text(value: Any) -> list[str]:
    if isinstance(value, str):
        normalized = _normalize_text(value)
        return [normalized] if normalized else []
    if isinstance(value, list):
        return [text for item in value for text in _flatten_text(item)]
    if isinstance(value, dict):
        return [
            text
            for key, item in value.items()
            if key not in _IGNORED_KEYS
            for text in _flatten_text(item)
        ]
    return []


def record_to_text(record: dict[str, Any]) -> str:
    """Render supported instruction/chat records without record or group IDs."""
    parts: list[str] = []
    agent_role = record.get("agent_role")
    if isinstance(agent_role, str) and _normalize_text(agent_role):
        parts.append(f"agent: {_normalize_text(agent_role)}")

    messages = record.get("messages")
    if isinstance(messages, list):
        for message in messages:
            if isinstance(message, dict):
                role_value = message.get("role", "message")
                role = _ROLE.sub("_", str(role_value)).strip("_") or "message"
                content = message.get("content")
                text_values = _flatten_text(content)
                if text_values:
                    parts.append(f"{role}: {' '.join(text_values)}")
            elif isinstance(message, str) and _normalize_text(message):
                parts.append(f"message: {_normalize_text(message)}")

    for key in ("text", "instruction", "input", "prompt", "completion", "output", "response"):
        value = record.get(key)
        if isinstance(value, str) and _normalize_text(value):
            label = "assistant" if key in {"completion", "output", "response"} else key
            parts.append(
                _normalize_text(value) if key == "text" else f"{label}: {_normalize_text(value)}"
            )

    if not parts:
        parts = _flatten_text(
            {key: value for key, value in record.items() if key not in _IGNORED_KEYS}
        )
    rendered = "\n".join(part for part in parts if part).strip()
    if not rendered:
        raise DatasetTextExportError("record has no exportable text content")
    return rendered


def _sha256_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
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


def export_dataset_split(
    release_path: str | Path,
    split: str,
    output_path: str | Path,
) -> dict[str, Any]:
    if split not in {"train", "validation", "test"}:
        raise DatasetTextExportError("split must be train, validation, or test")
    root = Path(release_path).expanduser().resolve()
    output = Path(output_path).expanduser().resolve()
    if root == output or root in output.parents:
        raise DatasetTextExportError("text export must be outside the immutable dataset release")
    output.parent.mkdir(parents=True, exist_ok=True)
    sidecar = output.with_suffix(output.suffix + ".manifest.json")
    if output.exists() or sidecar.exists():
        raise DatasetTextExportError("text export or its manifest already exists; choose a new versioned path")

    try:
        verification = verify_dataset_release(root)
    except (DatasetVerificationError, OSError, UnicodeError) as exc:
        raise DatasetTextExportError(f"dataset release verification failed: {exc}") from exc

    source = root / f"{split}.jsonl"
    temporary_output: Path | None = None
    published_output = False
    try:
        file_descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
        )
        os.close(file_descriptor)
        temporary_output = Path(temporary_name)
        records = 0
        with source.open("r", encoding="utf-8") as source_handle, temporary_output.open(
            "w", encoding="utf-8", newline="\n"
        ) as output_handle:
            for line_number, line in enumerate(source_handle, start=1):
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise DatasetTextExportError(
                        f"{source.name}:{line_number}: invalid JSON"
                    ) from exc
                if not isinstance(record, dict):
                    raise DatasetTextExportError(
                        f"{source.name}:{line_number}: record must be an object"
                    )
                rendered = record_to_text(record)
                if records:
                    output_handle.write("\n\n")
                output_handle.write(rendered)
                records += 1
            output_handle.write("\n")
            output_handle.flush()
            os.fsync(output_handle.fileno())
        if records == 0:
            raise DatasetTextExportError(f"{source.name} contains no records")
        os.replace(temporary_output, output)
        temporary_output = None
        published_output = True
        output_sha256, output_size = _sha256_file(output)
        release_manifest_sha256 = verification["manifest_sha256"]
        release_manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        source_metadata = release_manifest["files"][source.name]
        export_manifest = {
            "schema_version": "1.0.0",
            "exporter": "rusamodels.dataset_text_export",
            "dataset_id": verification["dataset_id"],
            "dataset_version": verification["dataset_version"],
            "split": split,
            "license_ids": release_manifest["licenses"],
            "release_manifest_sha256": release_manifest_sha256,
            "source_split_sha256": source_metadata["sha256"],
            "source_split_file": source.name,
            "records": records,
            "encoding": "utf-8",
            "tokenizer": "byte-level-v1",
            "output_file": output.name,
            "output_sha256": output_sha256,
            "output_size_bytes": output_size,
        }
        _atomic_json(sidecar, export_manifest)
        return {
            "dataset_id": verification["dataset_id"],
            "dataset_version": verification["dataset_version"],
            "split": split,
            "records": records,
            "output_file": str(output),
            "output_sha256": output_sha256,
            "manifest_file": str(sidecar),
        }
    except (OSError, UnicodeError) as exc:
        raise DatasetTextExportError(f"dataset text export failed: {exc}") from exc
    finally:
        if temporary_output is not None:
            temporary_output.unlink(missing_ok=True)
        if published_output and not sidecar.exists():
            output.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", required=True, type=Path)
    parser.add_argument("--split", required=True, choices=("train", "validation", "test"))
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        result = export_dataset_split(args.release, args.split, args.output)
    except (DatasetTextExportError, OSError, UnicodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
