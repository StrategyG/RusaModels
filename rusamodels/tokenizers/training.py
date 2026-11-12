"""Train a reproducible byte-level BPE tokenizer from a verified train-only export."""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any

from .loading import load_tokenizer_artifact
from .validation import DEFAULT_SMOKE_SAMPLES, audit_tokenizer
from .vocabulary import SPECIAL_TOKENS, resolve_special_token_ids


class TokenizerTrainingError(ValueError):
    """Raised when tokenizer training inputs or outputs fail validation."""


_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_TOKENIZER_ID = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")
_EXPORT_MANIFEST_KEYS = {
    "schema_version",
    "exporter",
    "dataset_id",
    "dataset_version",
    "split",
    "license_ids",
    "release_manifest_sha256",
    "source_split_sha256",
    "source_split_file",
    "records",
    "encoding",
    "tokenizer",
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


def _sha256_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def _load_export_manifest(corpus: Path, manifest_path: Path) -> tuple[dict[str, Any], str]:
    try:
        raw = manifest_path.read_bytes()
        manifest = json.loads(
            raw.decode("utf-8"),
            parse_constant=_reject_constant,
            object_pairs_hook=_unique_object,
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise TokenizerTrainingError(f"cannot read dataset export manifest: {exc}") from exc
    if not isinstance(manifest, dict) or set(manifest) != _EXPORT_MANIFEST_KEYS:
        raise TokenizerTrainingError("dataset export manifest has unsupported or missing fields")
    if manifest["schema_version"] != "1.0.0" or manifest["exporter"] != "rusamodels.dataset_text_export":
        raise TokenizerTrainingError("unsupported dataset text export manifest")
    if manifest["split"] != "train":
        raise TokenizerTrainingError("tokenizer training may use only the dataset train split")
    if manifest["encoding"] != "utf-8" or manifest["tokenizer"] != "byte-level-v1":
        raise TokenizerTrainingError("dataset text export encoding/tokenizer contract is unsupported")
    for field in ("dataset_id", "dataset_version"):
        if not isinstance(manifest[field], str) or not manifest[field].strip():
            raise TokenizerTrainingError(f"dataset export {field} is invalid")
    for field in ("release_manifest_sha256", "source_split_sha256", "output_sha256"):
        if not isinstance(manifest[field], str) or not _SHA256.fullmatch(manifest[field]):
            raise TokenizerTrainingError(f"dataset export {field} is invalid")
    for field in ("records", "output_size_bytes"):
        if type(manifest[field]) is not int or manifest[field] < 1:
            raise TokenizerTrainingError(f"dataset export {field} must be a positive integer")
    for field in ("source_split_file", "output_file"):
        value = manifest[field]
        if (
            not isinstance(value, str)
            or not value.strip()
            or Path(value).name != value
            or "/" in value
            or "\\" in value
        ):
            raise TokenizerTrainingError(f"dataset export {field} must be a safe file name")
    if manifest["output_file"] != corpus.name:
        raise TokenizerTrainingError("dataset export manifest does not identify the supplied corpus file")
    licenses = manifest["license_ids"]
    if (
        not isinstance(licenses, list)
        or not licenses
        or any(
            not isinstance(item, str)
            or not item.strip()
            or "replace" in item.casefold()
            for item in licenses
        )
        or len(licenses) != len(set(licenses))
    ):
        raise TokenizerTrainingError("dataset export license_ids are missing, duplicated, or placeholders")
    actual_sha256, actual_size = _sha256_file(corpus)
    if actual_sha256 != manifest["output_sha256"] or actual_size != manifest["output_size_bytes"]:
        raise TokenizerTrainingError("training corpus digest/size does not match its export manifest")
    return manifest, hashlib.sha256(raw).hexdigest()


def train_tokenizer(
    corpus_path: str | Path,
    corpus_manifest_path: str | Path,
    output_dir: str | Path,
    *,
    vocab_size: int = 32000,
    min_frequency: int = 2,
    tokenizer_id: str = "rusaai-research-100m-bpe-32k",
    tokenizer_version: str = "0.1.0",
) -> dict[str, Any]:
    """Train and atomically publish an immutable tokenizer artifact."""
    if type(vocab_size) is not int or vocab_size < 260:
        raise TokenizerTrainingError("vocab_size must be an integer of at least 260")
    if type(min_frequency) is not int or min_frequency < 1:
        raise TokenizerTrainingError("min_frequency must be a positive integer")
    if not isinstance(tokenizer_id, str) or not _TOKENIZER_ID.fullmatch(tokenizer_id):
        raise TokenizerTrainingError("tokenizer_id is invalid")
    if (
        not isinstance(tokenizer_version, str)
        or not tokenizer_version.strip()
        or len(tokenizer_version) > 64
    ):
        raise TokenizerTrainingError("tokenizer_version is invalid")

    corpus_input = Path(corpus_path).expanduser()
    manifest_input = Path(corpus_manifest_path).expanduser()
    output_input = Path(output_dir).expanduser()
    if corpus_input.is_symlink() or manifest_input.is_symlink() or output_input.is_symlink():
        raise TokenizerTrainingError("symlink paths are not allowed for tokenizer inputs or output")
    corpus = corpus_input.resolve()
    corpus_manifest = manifest_input.resolve()
    if output_input.name in {"", ".", ".."}:
        raise TokenizerTrainingError("output_dir must name a new artifact directory")
    output = output_input.parent.resolve() / output_input.name
    if output in {corpus, corpus_manifest}:
        raise TokenizerTrainingError("output directory must differ from corpus inputs")
    if not corpus.is_file() or not corpus_manifest.is_file():
        raise TokenizerTrainingError("training corpus and its export manifest must exist")
    if output.exists():
        raise TokenizerTrainingError(f"tokenizer output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    export_manifest, export_manifest_sha256 = _load_export_manifest(corpus, corpus_manifest)

    try:
        from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    except ImportError as exc:
        raise RuntimeError("Install the tokenizer extra to train a BPE tokenizer") from exc

    staging: Path | None = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    published = False
    try:
        backend = Tokenizer(models.BPE(unk_token=SPECIAL_TOKENS["unk"]))
        backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        backend.decoder = decoders.ByteLevel()
        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=list(SPECIAL_TOKENS.values()),
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            show_progress=False,
        )
        backend.train(files=[str(corpus)], trainer=trainer)
        actual_vocab_size = backend.get_vocab_size(with_added_tokens=True)
        if actual_vocab_size != vocab_size:
            raise TokenizerTrainingError(
                f"corpus produced vocabulary size {actual_vocab_size}, requested {vocab_size}; "
                "use a larger approved training corpus rather than padding the vocabulary"
            )
        special_ids = resolve_special_token_ids(backend)
        audit = audit_tokenizer(backend, special_ids)
        tokenizer_path = staging / "tokenizer.json"
        backend.save(str(tokenizer_path))
        tokenizer_sha256, tokenizer_size = _sha256_file(tokenizer_path)

        corpus_sha256_before, corpus_size_before = _sha256_file(corpus)
        manifest_sha256_before, _ = _sha256_file(corpus_manifest)
        if (
            corpus_sha256_before != export_manifest["output_sha256"]
            or corpus_size_before != export_manifest["output_size_bytes"]
            or manifest_sha256_before != export_manifest_sha256
        ):
            raise TokenizerTrainingError("training corpus or export manifest changed during tokenization")

        artifact_manifest = {
            "schema_version": "1.0.0",
            "tokenizer_id": tokenizer_id,
            "tokenizer_version": tokenizer_version,
            "algorithm": "byte_level_bpe",
            "tokenizer_file": "tokenizer.json",
            "tokenizer_sha256": tokenizer_sha256,
            "tokenizer_size_bytes": tokenizer_size,
            "vocab_size": actual_vocab_size,
            "special_tokens": {
                name: {"token": token, "id": special_ids[name]}
                for name, token in SPECIAL_TOKENS.items()
            },
            "training_corpus": {
                "dataset_id": export_manifest["dataset_id"],
                "dataset_version": export_manifest["dataset_version"],
                "split": "train",
                "release_manifest_sha256": export_manifest["release_manifest_sha256"],
                "corpus_manifest_sha256": export_manifest_sha256,
                "corpus_sha256": export_manifest["output_sha256"],
                "corpus_size_bytes": export_manifest["output_size_bytes"],
                "license_ids": export_manifest["license_ids"],
            },
            "trainer": {
                "requested_vocab_size": vocab_size,
                "min_frequency": min_frequency,
                "pre_tokenizer": "bytelevel",
                "normalizer": "none",
                "library_version": version("tokenizers"),
            },
            "created_at": datetime.now(UTC).isoformat(),
        }
        manifest_path = staging / "rusa-tokenizer-manifest.json"
        manifest_path.write_text(
            json.dumps(artifact_manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        with manifest_path.open("rb") as handle:
            os.fsync(handle.fileno())
        if output.exists():
            raise TokenizerTrainingError(f"tokenizer output appeared during training: {output}")
        os.rename(staging, output)
        staging = None
        published = True
        loaded = load_tokenizer_artifact(output)
        loaded_audit = audit_tokenizer(loaded.backend, loaded.special_token_ids)
        return {
            "tokenizer_id": tokenizer_id,
            "tokenizer_version": tokenizer_version,
            "vocab_size": actual_vocab_size,
            "special_token_ids": special_ids,
            "tokenizer_sha256": tokenizer_sha256,
            "dataset_id": export_manifest["dataset_id"],
            "dataset_version": export_manifest["dataset_version"],
            "split": export_manifest["split"],
            "audit": loaded_audit,
            "output_dir": str(output),
        }
    except Exception:
        if staging is not None and staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
        if published and output.exists():
            shutil.rmtree(output, ignore_errors=True)
        raise
