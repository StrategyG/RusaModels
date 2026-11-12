"""Tokenize verified dataset splits with a pinned local tokenizer and pack fixed sequences."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Any

from rusamodels.dataset_text_export import (
    DatasetTextExportError,
    record_to_text,
)
from rusamodels.dataset_verification import (
    DatasetVerificationError,
    verify_dataset_release,
)


class DatasetTokenizationError(ValueError):
    """Raised when tokenizer assets or packed dataset artifacts are invalid."""


_TOKENIZER_REVISION = re.compile(r"^[0-9a-f]{40}$")


def hash_tokenizer_directory(tokenizer_path: str | Path) -> str:
    """Hash every tokenizer asset by relative path and file digest."""
    root_input = Path(tokenizer_path).expanduser()
    if root_input.is_symlink():
        raise DatasetTokenizationError("tokenizer directory must not be a symlink")
    root = root_input.resolve()
    if not root.is_dir():
        raise DatasetTokenizationError(f"tokenizer directory does not exist: {root}")
    files = sorted(root.rglob("*"), key=lambda path: path.relative_to(root).as_posix())
    digest = hashlib.sha256()
    file_count = 0
    for path in files:
        if path.is_symlink():
            raise DatasetTokenizationError(
                f"tokenizer directory contains a symlink: {path.relative_to(root)}"
            )
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        file_digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                file_digest.update(chunk)
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_digest.hexdigest().encode("ascii"))
        digest.update(b"\n")
        file_count += 1
    if file_count == 0:
        raise DatasetTokenizationError("tokenizer directory contains no regular files")
    return digest.hexdigest()


def _tokenizer_properties(
    tokenizer: Any,
    *,
    add_bos: bool,
) -> tuple[int, int, int | None, int]:
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    bos_token_id = getattr(tokenizer, "bos_token_id", None)
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if type(eos_token_id) is not int or eos_token_id < 0:
        raise DatasetTokenizationError("tokenizer must define a non-negative eos_token_id")
    if add_bos and (type(bos_token_id) is not int or bos_token_id < 0):
        raise DatasetTokenizationError("add_bos requires a non-negative bos_token_id")
    if pad_token_id is None:
        pad_token_id = eos_token_id
    if type(pad_token_id) is not int or pad_token_id < 0:
        raise DatasetTokenizationError("tokenizer pad_token_id must be a non-negative integer")
    try:
        vocab_size = len(tokenizer)
    except (TypeError, AttributeError):
        vocab_size = getattr(tokenizer, "vocab_size", None)
    if type(vocab_size) is not int or vocab_size < 1:
        raise DatasetTokenizationError("tokenizer must expose a positive vocabulary size")
    if max(eos_token_id, pad_token_id, bos_token_id or 0) >= vocab_size:
        raise DatasetTokenizationError("tokenizer special token ID is outside its vocabulary")
    return eos_token_id, pad_token_id, bos_token_id, vocab_size


def _validate_packing_inputs(
    *,
    tokenizer_id: str,
    tokenizer_revision: str,
    tokenizer_path: str | Path,
    sequence_length: int,
    add_bos: bool,
    add_eos: bool,
) -> None:
    if not isinstance(tokenizer_id, str) or not tokenizer_id.strip() or len(tokenizer_id) > 256:
        raise DatasetTokenizationError("tokenizer_id must be a non-empty identifier")
    if not isinstance(tokenizer_revision, str) or not _TOKENIZER_REVISION.fullmatch(tokenizer_revision):
        raise DatasetTokenizationError("tokenizer_revision must be a pinned 40-character commit SHA")
    if not isinstance(tokenizer_path, (str, Path)):
        raise DatasetTokenizationError("tokenizer_path must be a local directory path")
    if type(sequence_length) is not int or not 2 <= sequence_length <= 65536:
        raise DatasetTokenizationError("sequence_length must be between 2 and 65536")
    if type(add_bos) is not bool or type(add_eos) is not bool:
        raise DatasetTokenizationError("add_bos and add_eos must be booleans")


def _hash_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def _publish_new_file(path: Path, temporary: Path) -> None:
    os.link(temporary, path)


def pack_dataset_split(
    release_path: str | Path,
    split: str,
    output_path: str | Path,
    *,
    tokenizer: Any,
    tokenizer_id: str,
    tokenizer_revision: str,
    tokenizer_path: str | Path,
    sequence_length: int,
    add_bos: bool = False,
    add_eos: bool = True,
) -> dict[str, Any]:
    """Tokenize one verified split and stream fixed-length causal-LM sequences."""
    if split not in {"train", "validation", "test"}:
        raise DatasetTokenizationError("split must be train, validation, or test")
    _validate_packing_inputs(
        tokenizer_id=tokenizer_id,
        tokenizer_revision=tokenizer_revision,
        tokenizer_path=tokenizer_path,
        sequence_length=sequence_length,
        add_bos=add_bos,
        add_eos=add_eos,
    )
    tokenizer_sha256_before = hash_tokenizer_directory(tokenizer_path)
    eos_token_id, pad_token_id, bos_token_id, vocab_size = _tokenizer_properties(
        tokenizer, add_bos=add_bos
    )
    model_max_length = getattr(tokenizer, "model_max_length", None)
    if (
        type(model_max_length) is int
        and model_max_length < 1_000_000_000
        and sequence_length > model_max_length
    ):
        raise DatasetTokenizationError(
            f"sequence_length={sequence_length} exceeds tokenizer model_max_length={model_max_length}"
        )

    root = Path(release_path).expanduser().resolve()
    output_input = Path(output_path).expanduser()
    if output_input.is_symlink():
        raise DatasetTokenizationError("symlink paths are not allowed for packed dataset outputs")
    output = output_input.parent.resolve() / output_input.name
    manifest_path = output.with_suffix(output.suffix + ".manifest.json")
    if root == output or root in output.parents:
        raise DatasetTokenizationError("packed output must be outside the immutable dataset release")
    if output.exists() or manifest_path.exists():
        raise DatasetTokenizationError("packed output or manifest already exists")
    output.parent.mkdir(parents=True, exist_ok=True)

    try:
        verification = verify_dataset_release(root)
    except (DatasetVerificationError, OSError, UnicodeError) as exc:
        raise DatasetTokenizationError(f"dataset release verification failed: {exc}") from exc

    release_manifest_path = root / "manifest.json"
    release_manifest_sha256, _ = _hash_file(release_manifest_path)
    release_manifest = json.loads(release_manifest_path.read_text(encoding="utf-8"))
    source_path = root / f"{split}.jsonl"
    source_sha256_before, source_size_before = _hash_file(source_path)
    expected_source = release_manifest["files"][source_path.name]
    if (
        source_sha256_before != expected_source["sha256"]
        or source_size_before != expected_source["size_bytes"]
    ):
        raise DatasetTokenizationError("source split changed after release verification")

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
    )
    temporary_output = Path(temporary_name)
    output_published = False
    manifest_published = False
    records = 0
    sequences = 0
    token_count = 0
    padding_tokens = 0
    buffer: list[int] = []
    buffer_record_ids: list[str] = []
    buffer_record_id_set: set[str] = set()

    def write_sequence(handle, token_ids: list[int], record_ids: list[str], *, padded: bool) -> None:
        nonlocal sequences, padding_tokens
        padding = sequence_length - len(token_ids)
        input_ids = token_ids + ([pad_token_id] * padding)
        attention_mask = ([1] * len(token_ids)) + ([0] * padding)
        labels = token_ids + ([-100] * padding)
        payload = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "record_ids": record_ids,
        }
        handle.write((json.dumps(payload, separators=(",", ":"), sort_keys=True) + "\n").encode())
        sequences += 1
        if padded:
            padding_tokens += padding

    try:
        with os.fdopen(descriptor, "wb") as output_handle, source_path.open(
            "r", encoding="utf-8"
        ) as source_handle:
            for line_number, line in enumerate(source_handle, start=1):
                if not line.strip():
                    raise DatasetTokenizationError(
                        f"{source_path.name}:{line_number}: blank lines are not allowed"
                    )
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise DatasetTokenizationError(
                        f"{source_path.name}:{line_number}: invalid JSON"
                    ) from exc
                if not isinstance(record, dict) or not isinstance(record.get("id"), str):
                    raise DatasetTokenizationError(
                        f"{source_path.name}:{line_number}: record ID is missing"
                    )
                try:
                    text = record_to_text(record)
                    encoded = list(tokenizer.encode(text, add_special_tokens=False))
                except (DatasetTextExportError, TypeError, ValueError, RuntimeError) as exc:
                    raise DatasetTokenizationError(
                        f"{source_path.name}:{line_number}: tokenization failed: {exc}"
                    ) from exc
                if add_bos:
                    encoded.insert(0, bos_token_id)
                if add_eos:
                    encoded.append(eos_token_id)
                if not encoded:
                    raise DatasetTokenizationError(
                        f"{source_path.name}:{line_number}: tokenizer produced an empty sequence"
                    )
                if any(type(token_id) is not int or not 0 <= token_id < vocab_size for token_id in encoded):
                    raise DatasetTokenizationError(
                        f"{source_path.name}:{line_number}: tokenizer returned an ID outside its vocabulary"
                    )
                records += 1
                token_count += len(encoded)
                offset = 0
                while offset < len(encoded):
                    available = sequence_length - len(buffer)
                    take = min(available, len(encoded) - offset)
                    buffer.extend(encoded[offset : offset + take])
                    if record["id"] not in buffer_record_id_set:
                        buffer_record_ids.append(record["id"])
                        buffer_record_id_set.add(record["id"])
                    offset += take
                    if len(buffer) == sequence_length:
                        write_sequence(output_handle, buffer, buffer_record_ids, padded=False)
                        buffer = []
                        buffer_record_ids = []
                        buffer_record_id_set = set()
            if buffer:
                write_sequence(output_handle, buffer, buffer_record_ids, padded=True)
            output_handle.flush()
            os.fsync(output_handle.fileno())

        if records != expected_source["records"]:
            raise DatasetTokenizationError("source split record count changed during tokenization")
        source_sha256_after, source_size_after = _hash_file(source_path)
        release_manifest_sha256_after, _ = _hash_file(release_manifest_path)
        tokenizer_sha256_after = hash_tokenizer_directory(tokenizer_path)
        if (
            source_sha256_after != source_sha256_before
            or source_size_after != source_size_before
            or release_manifest_sha256_after != release_manifest_sha256
            or release_manifest_sha256_after != verification["manifest_sha256"]
            or tokenizer_sha256_after != tokenizer_sha256_before
        ):
            raise DatasetTokenizationError("dataset release changed during tokenization")
        _publish_new_file(output, temporary_output)
        output_published = True
        temporary_output.unlink(missing_ok=True)
        output_sha256, output_size = _hash_file(output)
        manifest = {
            "schema_version": "1.0.0",
            "tokenizer": "rusamodels.local_hf_tokenizer",
            "dataset_id": verification["dataset_id"],
            "dataset_version": verification["dataset_version"],
            "split": split,
            "release_manifest_sha256": verification["manifest_sha256"],
            "source_split_file": source_path.name,
            "source_split_sha256": source_sha256_before,
            "tokenizer_id": tokenizer_id,
            "tokenizer_revision": tokenizer_revision,
            "tokenizer_sha256": tokenizer_sha256_before,
            "vocab_size": vocab_size,
            "sequence_length": sequence_length,
            "add_bos": add_bos,
            "add_eos": add_eos,
            "bos_token_id": bos_token_id if add_bos else None,
            "eos_token_id": eos_token_id if add_eos else None,
            "pad_token_id": pad_token_id,
            "records": records,
            "sequences": sequences,
            "tokens": token_count,
            "padding_tokens": padding_tokens,
            "encoding": "jsonl-int64",
            "output_file": output.name,
            "output_sha256": output_sha256,
            "output_size_bytes": output_size,
        }
        manifest_bytes = (
            json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
        ).encode()
        descriptor, manifest_temporary_name = tempfile.mkstemp(
            prefix=f".{manifest_path.name}.", suffix=".tmp", dir=manifest_path.parent
        )
        manifest_temporary = Path(manifest_temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(manifest_bytes)
                handle.flush()
                os.fsync(handle.fileno())
            _publish_new_file(manifest_path, manifest_temporary)
            manifest_published = True
        finally:
            manifest_temporary.unlink(missing_ok=True)
    except (OSError, UnicodeError) as exc:
        raise DatasetTokenizationError(f"dataset tokenization failed: {exc}") from exc
    finally:
        temporary_output.unlink(missing_ok=True)
        if output_published and not manifest_published:
            output.unlink(missing_ok=True)

    return {
        "dataset_id": verification["dataset_id"],
        "dataset_version": verification["dataset_version"],
        "split": split,
        "records": records,
        "sequences": sequences,
        "tokens": token_count,
        "padding_tokens": padding_tokens,
        "output_file": str(output),
        "manifest_file": str(manifest_path),
        "output_sha256": output_sha256,
    }


def load_local_tokenizer(tokenizer_path: str | Path) -> Any:
    """Load only local tokenizer assets; remote code and network downloads are disabled."""
    path = Path(tokenizer_path).expanduser()
    if path.is_symlink() or not path.is_dir():
        raise DatasetTokenizationError("tokenizer_path must be a local, non-symlink directory")
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise DatasetTokenizationError(
            'install the optional tokenizer dependency with python -m pip install -e ".[tokenization]"'
        ) from exc
    try:
        return AutoTokenizer.from_pretrained(
            str(path.resolve()),
            local_files_only=True,
            trust_remote_code=False,
            use_fast=True,
        )
    except (OSError, ValueError, TypeError, KeyError) as exc:
        raise DatasetTokenizationError(f"cannot load local tokenizer assets: {exc}") from exc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", required=True, type=Path)
    parser.add_argument("--split", required=True, choices=("train", "validation", "test"))
    parser.add_argument("--tokenizer-path", required=True, type=Path)
    parser.add_argument("--tokenizer-id", required=True)
    parser.add_argument("--tokenizer-revision", required=True)
    parser.add_argument("--sequence-length", type=int, required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--add-bos", action="store_true")
    parser.add_argument("--no-eos", action="store_true")
    args = parser.parse_args(argv)
    try:
        tokenizer = load_local_tokenizer(args.tokenizer_path)
        result = pack_dataset_split(
            args.release,
            args.split,
            args.output,
            tokenizer=tokenizer,
            tokenizer_id=args.tokenizer_id,
            tokenizer_revision=args.tokenizer_revision,
            tokenizer_path=args.tokenizer_path,
            sequence_length=args.sequence_length,
            add_bos=args.add_bos,
            add_eos=not args.no_eos,
        )
    except (DatasetTokenizationError, OSError, UnicodeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
