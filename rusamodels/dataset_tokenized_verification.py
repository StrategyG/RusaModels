"""Verify packed tokenized dataset artifacts and their source/tokenizer bindings."""
from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import re
import sys
from pathlib import Path
from typing import Any

from rusamodels.dataset_tokenization import (
    DatasetTokenizationError,
    hash_tokenizer_directory,
)
from rusamodels.dataset_verification import (
    DatasetVerificationError,
    verify_dataset_release,
)


class TokenizedDatasetVerificationError(ValueError):
    """Raised when a packed tokenized artifact fails integrity verification."""


_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_COMMIT_SHA = re.compile(r"^[0-9a-f]{40}$")
_RECORD_ID = re.compile(r"^record_[0-9a-f]{24}$")
_DATASET_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
_MANIFEST_KEYS = {
    "schema_version",
    "tokenizer",
    "dataset_id",
    "dataset_version",
    "split",
    "release_manifest_sha256",
    "source_split_file",
    "source_split_sha256",
    "tokenizer_id",
    "tokenizer_revision",
    "tokenizer_sha256",
    "vocab_size",
    "sequence_length",
    "add_bos",
    "add_eos",
    "bos_token_id",
    "eos_token_id",
    "pad_token_id",
    "records",
    "sequences",
    "tokens",
    "padding_tokens",
    "encoding",
    "output_file",
    "output_sha256",
    "output_size_bytes",
}


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant is prohibited: {value}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def _hash_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def _load_json(raw: str, source: str) -> Any:
    try:
        return json.loads(
            raw,
            parse_constant=_reject_constant,
            object_pairs_hook=_unique_object,
        )
    except (json.JSONDecodeError, ValueError) as exc:
        raise TokenizedDatasetVerificationError(f"{source}: invalid JSON: {exc}") from exc


def _validate_manifest(manifest: Any, output_path: Path) -> dict[str, Any]:
    if not isinstance(manifest, dict) or set(manifest) != _MANIFEST_KEYS:
        raise TokenizedDatasetVerificationError("tokenization manifest has unsupported or missing fields")
    if manifest["schema_version"] != "1.0.0" or manifest["tokenizer"] != "rusamodels.local_hf_tokenizer":
        raise TokenizedDatasetVerificationError("unsupported tokenization manifest version or tokenizer type")
    if (
        not isinstance(manifest["dataset_id"], str)
        or not _DATASET_ID.fullmatch(manifest["dataset_id"])
        or not isinstance(manifest["dataset_version"], str)
        or not _DATASET_ID.fullmatch(manifest["dataset_version"])
    ):
        raise TokenizedDatasetVerificationError("tokenized dataset identity is invalid")
    if not isinstance(manifest["split"], str) or manifest["split"] not in {"train", "validation", "test"}:
        raise TokenizedDatasetVerificationError("tokenized dataset split is invalid")
    if not isinstance(manifest["source_split_file"], str) or manifest["source_split_file"] != f"{manifest['split']}.jsonl":
        raise TokenizedDatasetVerificationError("source split file does not match the declared split")
    for field in (
        "release_manifest_sha256",
        "source_split_sha256",
        "tokenizer_sha256",
        "output_sha256",
    ):
        if not isinstance(manifest[field], str) or not _SHA256.fullmatch(manifest[field]):
            raise TokenizedDatasetVerificationError(f"{field} must be a lowercase SHA-256 digest")
    if not isinstance(manifest["tokenizer_id"], str) or not manifest["tokenizer_id"].strip():
        raise TokenizedDatasetVerificationError("tokenizer_id is missing")
    if not isinstance(manifest["tokenizer_revision"], str) or not _COMMIT_SHA.fullmatch(
        manifest["tokenizer_revision"]
    ):
        raise TokenizedDatasetVerificationError("tokenizer_revision is not a pinned commit SHA")
    if manifest["output_file"] != output_path.name or Path(manifest["output_file"]).name != manifest["output_file"]:
        raise TokenizedDatasetVerificationError("output file name does not match the manifest")
    if manifest["encoding"] != "jsonl-int64":
        raise TokenizedDatasetVerificationError("unsupported tokenized artifact encoding")
    if not isinstance(manifest["output_file"], str):
        raise TokenizedDatasetVerificationError("output_file must be a basename string")
    for field, minimum in (
        ("vocab_size", 1),
        ("sequence_length", 2),
        ("records", 1),
        ("sequences", 1),
        ("tokens", 1),
        ("output_size_bytes", 1),
    ):
        if type(manifest[field]) is not int or manifest[field] < minimum:
            raise TokenizedDatasetVerificationError(f"{field} must be an integer >= {minimum}")
    if type(manifest["padding_tokens"]) is not int or manifest["padding_tokens"] < 0:
        raise TokenizedDatasetVerificationError("padding_tokens must be a non-negative integer")
    if type(manifest["add_bos"]) is not bool or type(manifest["add_eos"]) is not bool:
        raise TokenizedDatasetVerificationError("add_bos and add_eos must be booleans")
    for field in ("bos_token_id", "eos_token_id"):
        value = manifest[field]
        if value is not None and (type(value) is not int or not 0 <= value < manifest["vocab_size"]):
            raise TokenizedDatasetVerificationError(f"{field} is outside the tokenizer vocabulary")
    if manifest["add_bos"] != (manifest["bos_token_id"] is not None):
        raise TokenizedDatasetVerificationError("add_bos does not match bos_token_id")
    if manifest["add_eos"] != (manifest["eos_token_id"] is not None):
        raise TokenizedDatasetVerificationError("add_eos does not match eos_token_id")
    if (
        type(manifest["pad_token_id"]) is not int
        or not 0 <= manifest["pad_token_id"] < manifest["vocab_size"]
    ):
        raise TokenizedDatasetVerificationError("pad_token_id is outside the tokenizer vocabulary")
    return manifest


def verify_tokenized_split(
    output_path: str | Path,
    *,
    manifest_path: str | Path | None = None,
    release_path: str | Path | None = None,
    tokenizer_path: str | Path | None = None,
) -> dict[str, Any]:
    """Verify token arrays, masks, labels, output checksums, and optional source bindings."""
    output_input = Path(output_path).expanduser()
    manifest_input = (
        Path(manifest_path).expanduser()
        if manifest_path is not None
        else output_input.with_suffix(output_input.suffix + ".manifest.json")
    )
    if output_input.is_symlink() or manifest_input.is_symlink():
        raise TokenizedDatasetVerificationError("symlink paths are not allowed for tokenized artifacts")
    output = output_input.parent.resolve() / output_input.name
    manifest_file = manifest_input.parent.resolve() / manifest_input.name
    release_root = (
        Path(release_path).expanduser().resolve()
        if release_path is not None
        else None
    )
    if release_root is not None and (
        release_root == output
        or release_root in output.parents
        or release_root == manifest_file
        or release_root in manifest_file.parents
    ):
        raise TokenizedDatasetVerificationError(
            "tokenized artifacts must be outside the immutable source release"
        )
    if output == manifest_file:
        raise TokenizedDatasetVerificationError("tokenized output and manifest paths must be different")
    if not output.is_file() or not manifest_file.is_file():
        raise TokenizedDatasetVerificationError("tokenized output or manifest is missing")
    manifest_bytes = manifest_file.read_bytes()
    manifest_sha256_before = hashlib.sha256(manifest_bytes).hexdigest()
    manifest = _validate_manifest(
        _load_json(manifest_bytes.decode("utf-8"), manifest_file.name),
        output,
    )
    output_sha256_before, output_size_before = _hash_file(output)
    if not hmac.compare_digest(output_sha256_before, manifest["output_sha256"]):
        raise TokenizedDatasetVerificationError("tokenized output SHA-256 does not match manifest")
    if output_size_before != manifest["output_size_bytes"]:
        raise TokenizedDatasetVerificationError("tokenized output size does not match manifest")

    tokenizer_root: Path | None = None
    tokenizer_sha256_before: str | None = None
    if tokenizer_path is not None:
        tokenizer_input = Path(tokenizer_path).expanduser()
        if tokenizer_input.is_symlink():
            raise TokenizedDatasetVerificationError("tokenizer path must not be a symlink")
        tokenizer_root = tokenizer_input.resolve()
        try:
            tokenizer_sha256_before = hash_tokenizer_directory(tokenizer_root)
        except (DatasetTokenizationError, OSError) as exc:
            raise TokenizedDatasetVerificationError(f"tokenizer verification failed: {exc}") from exc
        if not hmac.compare_digest(tokenizer_sha256_before, manifest["tokenizer_sha256"]):
            raise TokenizedDatasetVerificationError("tokenizer assets do not match tokenization manifest")

    source_path: Path | None = None
    source_sha256_before: str | None = None
    source_size_before: int | None = None
    release_manifest_path: Path | None = None
    release_manifest_sha256_before: str | None = None
    if release_root is not None:
        if tokenizer_root is not None and (tokenizer_root == release_root or release_root in tokenizer_root.parents):
            raise TokenizedDatasetVerificationError("tokenizer assets must be outside the immutable source release")
        try:
            release = verify_dataset_release(release_root)
        except (DatasetVerificationError, OSError, UnicodeError) as exc:
            raise TokenizedDatasetVerificationError(f"source dataset verification failed: {exc}") from exc
        if (
            release["dataset_id"] != manifest["dataset_id"]
            or release["dataset_version"] != manifest["dataset_version"]
            or release["manifest_sha256"] != manifest["release_manifest_sha256"]
        ):
            raise TokenizedDatasetVerificationError("tokenized artifact is bound to a different dataset release")
        release_manifest_path = release_root / "manifest.json"
        release_manifest_sha256_before, _ = _hash_file(release_manifest_path)
        if release_manifest_sha256_before != release["manifest_sha256"]:
            raise TokenizedDatasetVerificationError("source release manifest changed after verification")
        if tokenizer_root is not None and (
            tokenizer_root == release_root
            or release_root in tokenizer_root.parents
            or tokenizer_root in release_root.parents
        ):
            raise TokenizedDatasetVerificationError("tokenizer assets must be separate from the source release")
        release_manifest = _load_json(
            release_manifest_path.read_text(encoding="utf-8"),
            "manifest.json",
        )
        source_path = release_root / manifest["source_split_file"]
        source_sha256_before, source_size_before = _hash_file(source_path)
        source_metadata = release_manifest["files"][manifest["source_split_file"]]
        if (
            source_sha256_before != manifest["source_split_sha256"]
            or source_sha256_before != source_metadata["sha256"]
            or source_size_before != source_metadata["size_bytes"]
            or source_metadata["records"] != manifest["records"]
        ):
            raise TokenizedDatasetVerificationError("tokenized artifact source split binding is invalid")

    sequence_count = 0
    token_count = 0
    padding_count = 0
    with output.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise TokenizedDatasetVerificationError(
                    f"{output.name}:{line_number}: blank sequence lines are not allowed"
                )
            item = _load_json(line, f"{output.name}:{line_number}")
            if not isinstance(item, dict) or set(item) != {
                "input_ids", "attention_mask", "labels", "record_ids"
            }:
                raise TokenizedDatasetVerificationError(
                    f"{output.name}:{line_number}: sequence fields are invalid"
                )
            input_ids = item["input_ids"]
            attention_mask = item["attention_mask"]
            labels = item["labels"]
            record_ids = item["record_ids"]
            sequence_length = manifest["sequence_length"]
            if any(
                not isinstance(values, list) or len(values) != sequence_length
                for values in (input_ids, attention_mask, labels)
            ):
                raise TokenizedDatasetVerificationError(
                    f"{output.name}:{line_number}: sequence arrays must equal sequence_length"
                )
            if any(
                type(token_id) is not int or not 0 <= token_id < manifest["vocab_size"]
                for token_id in input_ids
            ):
                raise TokenizedDatasetVerificationError(
                    f"{output.name}:{line_number}: input token ID is outside the vocabulary"
                )
            if any(type(mask) is not int or mask not in {0, 1} for mask in attention_mask):
                raise TokenizedDatasetVerificationError(
                    f"{output.name}:{line_number}: attention mask values must be 0 or 1"
                )
            if any(type(label) is not int or label < -100 for label in labels):
                raise TokenizedDatasetVerificationError(
                    f"{output.name}:{line_number}: labels contain an invalid token ID"
                )
            if not isinstance(record_ids, list) or not record_ids or any(
                not isinstance(record_id, str) or not _RECORD_ID.fullmatch(record_id)
                for record_id in record_ids
            ) or len(set(record_ids)) != len(record_ids):
                raise TokenizedDatasetVerificationError(
                    f"{output.name}:{line_number}: record_ids are invalid"
                )
            if not any(attention_mask):
                raise TokenizedDatasetVerificationError(
                    f"{output.name}:{line_number}: sequence contains no unpadded tokens"
                )
            seen_padding = False
            for input_id, mask, label in zip(input_ids, attention_mask, labels, strict=True):
                if mask == 0:
                    seen_padding = True
                    padding_count += 1
                    if input_id != manifest["pad_token_id"] or label != -100:
                        raise TokenizedDatasetVerificationError(
                            f"{output.name}:{line_number}: padding IDs and labels are inconsistent"
                        )
                else:
                    if seen_padding:
                        raise TokenizedDatasetVerificationError(
                            f"{output.name}:{line_number}: attention mask contains non-trailing padding"
                        )
                    token_count += 1
                    if label != input_id:
                        raise TokenizedDatasetVerificationError(
                            f"{output.name}:{line_number}: non-padding labels must equal input IDs"
                        )
            sequence_count += 1

    if (
        sequence_count != manifest["sequences"]
        or token_count != manifest["tokens"]
        or padding_count != manifest["padding_tokens"]
    ):
        raise TokenizedDatasetVerificationError("tokenized sequence and token counts do not match manifest")
    if source_path is not None and release_manifest_path is not None:
        source_sha256_after, source_size_after = _hash_file(source_path)
        release_manifest_sha256_after, _ = _hash_file(release_manifest_path)
        if (
            source_sha256_after != source_sha256_before
            or source_size_after != source_size_before
            or release_manifest_sha256_after != release_manifest_sha256_before
        ):
            raise TokenizedDatasetVerificationError("source dataset changed during tokenized verification")
    if tokenizer_root is not None and tokenizer_sha256_before is not None:
        try:
            tokenizer_sha256_after = hash_tokenizer_directory(tokenizer_root)
        except (DatasetTokenizationError, OSError) as exc:
            raise TokenizedDatasetVerificationError(f"tokenizer verification failed: {exc}") from exc
        if tokenizer_sha256_after != tokenizer_sha256_before:
            raise TokenizedDatasetVerificationError("tokenizer assets changed during verification")
    output_sha256_after, output_size_after = _hash_file(output)
    manifest_sha256_after, _ = _hash_file(manifest_file)
    if (
        output_sha256_after != output_sha256_before
        or output_size_after != output_size_before
        or manifest_sha256_after != manifest_sha256_before
    ):
        raise TokenizedDatasetVerificationError("tokenized artifact changed during verification")

    return {
        "dataset_id": manifest["dataset_id"],
        "dataset_version": manifest["dataset_version"],
        "split": manifest["split"],
        "sequences": sequence_count,
        "tokens": token_count,
        "padding_tokens": padding_count,
        "output_sha256": output_sha256_after,
        "manifest_sha256": manifest_sha256_after,
        "tokenizer_sha256": manifest["tokenizer_sha256"],
        "source_verified": release_path is not None,
        "tokenizer_verified": tokenizer_path is not None,
        "valid": True,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--release", type=Path)
    parser.add_argument("--tokenizer-path", type=Path)
    args = parser.parse_args(argv)
    try:
        result = verify_tokenized_split(
            args.output,
            manifest_path=args.manifest,
            release_path=args.release,
            tokenizer_path=args.tokenizer_path,
        )
    except (TokenizedDatasetVerificationError, OSError, UnicodeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
