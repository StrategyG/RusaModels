"""Integrity-checked loading of immutable RusaModels tokenizer artifacts."""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from .vocabulary import SPECIAL_TOKENS, resolve_special_token_ids


class TokenizerArtifactError(ValueError):
    """Raised when a tokenizer artifact or manifest fails validation."""


_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_MANIFEST_KEYS = {
    "schema_version",
    "tokenizer_id",
    "tokenizer_version",
    "algorithm",
    "tokenizer_file",
    "tokenizer_sha256",
    "tokenizer_size_bytes",
    "vocab_size",
    "special_tokens",
    "training_corpus",
    "trainer",
    "created_at",
}
_CORPUS_KEYS = {
    "dataset_id",
    "dataset_version",
    "split",
    "release_manifest_sha256",
    "corpus_manifest_sha256",
    "corpus_sha256",
    "corpus_size_bytes",
    "license_ids",
}
_TRAINER_KEYS = {
    "requested_vocab_size",
    "min_frequency",
    "pre_tokenizer",
    "normalizer",
    "library_version",
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


@dataclass
class TokenizerArtifact:
    backend: Any
    manifest: dict[str, Any]

    @property
    def vocab_size(self) -> int:
        return int(self.manifest["vocab_size"])

    @property
    def special_token_ids(self) -> dict[str, int]:
        return {
            name: value["id"]
            for name, value in self.manifest["special_tokens"].items()
        }

    @property
    def bos_token_id(self) -> int:
        return self.special_token_ids["bos"]

    @property
    def eos_token_id(self) -> int:
        return self.special_token_ids["eos"]

    @property
    def pad_token_id(self) -> int:
        return self.special_token_ids["pad"]

    @property
    def unk_token_id(self) -> int:
        return self.special_token_ids["unk"]

    def encode(
        self,
        text: str,
        *,
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> list[int]:
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        if type(add_bos) is not bool or type(add_eos) is not bool:
            raise TypeError("add_bos and add_eos must be booleans")
        token_ids = list(self.backend.encode(text, add_special_tokens=False).ids)
        if add_bos:
            token_ids.insert(0, self.bos_token_id)
        if add_eos:
            token_ids.append(self.eos_token_id)
        return token_ids

    def decode(self, token_ids, *, skip_special_tokens: bool = True) -> str:
        if type(skip_special_tokens) is not bool:
            raise TypeError("skip_special_tokens must be a boolean")
        values = list(token_ids)
        if any(type(value) is not int or not 0 <= value < self.vocab_size for value in values):
            raise ValueError("token IDs must be integers inside the tokenizer vocabulary")
        return self.backend.decode(values, skip_special_tokens=skip_special_tokens)


def load_tokenizer_artifact(directory: str | Path) -> TokenizerArtifact:
    """Load tokenizer.json only after its manifest, size and digest are verified."""
    input_path = Path(directory).expanduser()
    if input_path.is_symlink():
        raise TokenizerArtifactError("tokenizer artifact directory must not be a symlink")
    root = input_path.resolve()
    manifest_path = root / "rusa-tokenizer-manifest.json"
    tokenizer_path = root / "tokenizer.json"
    if not root.is_dir() or manifest_path.is_symlink() or tokenizer_path.is_symlink():
        raise TokenizerArtifactError("tokenizer artifact directory or files are missing/unsafe")
    try:
        raw_manifest = manifest_path.read_bytes()
        manifest = json.loads(
            raw_manifest.decode("utf-8"),
            parse_constant=_reject_constant,
            object_pairs_hook=_unique_object,
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise TokenizerArtifactError(f"cannot read tokenizer manifest: {exc}") from exc
    if not isinstance(manifest, dict) or set(manifest) != _MANIFEST_KEYS:
        raise TokenizerArtifactError("tokenizer manifest has unsupported or missing fields")
    if manifest["schema_version"] != "1.0.0" or manifest["algorithm"] != "byte_level_bpe":
        raise TokenizerArtifactError("unsupported tokenizer artifact version or algorithm")
    if (
        not isinstance(manifest["tokenizer_id"], str)
        or not re.fullmatch(r"^[a-z0-9][a-z0-9._-]{0,63}$", manifest["tokenizer_id"])
    ):
        raise TokenizerArtifactError("tokenizer_id is invalid")
    if (
        not isinstance(manifest["tokenizer_version"], str)
        or not manifest["tokenizer_version"].strip()
        or len(manifest["tokenizer_version"]) > 64
    ):
        raise TokenizerArtifactError("tokenizer_version is invalid")
    if manifest["tokenizer_file"] != "tokenizer.json":
        raise TokenizerArtifactError("tokenizer_file must be tokenizer.json")
    if not isinstance(manifest["tokenizer_sha256"], str) or not _SHA256.fullmatch(manifest["tokenizer_sha256"]):
        raise TokenizerArtifactError("tokenizer SHA-256 is invalid")
    if type(manifest["tokenizer_size_bytes"]) is not int or manifest["tokenizer_size_bytes"] < 1:
        raise TokenizerArtifactError("tokenizer_size_bytes must be a positive integer")
    if type(manifest["vocab_size"]) is not int or manifest["vocab_size"] < 260:
        raise TokenizerArtifactError("vocab_size must be an integer of at least 260")
    special = manifest["special_tokens"]
    if not isinstance(special, dict) or set(special) != set(SPECIAL_TOKENS):
        raise TokenizerArtifactError("special_tokens must contain unk, bos, eos, and pad")
    for name, token in SPECIAL_TOKENS.items():
        item = special[name]
        if (
            not isinstance(item, dict)
            or set(item) != {"token", "id"}
            or item["token"] != token
            or type(item["id"]) is not int
            or item["id"] < 0
        ):
            raise TokenizerArtifactError(f"special token metadata for {name} is invalid")
    ids = [special[name]["id"] for name in SPECIAL_TOKENS]
    if len(set(ids)) != 4:
        raise TokenizerArtifactError("special token IDs must be unique")
    corpus = manifest["training_corpus"]
    if not isinstance(corpus, dict) or set(corpus) != _CORPUS_KEYS or corpus.get("split") != "train":
        raise TokenizerArtifactError("tokenizer must record a valid training-only corpus manifest")
    trainer = manifest["trainer"]
    if not isinstance(trainer, dict) or set(trainer) != _TRAINER_KEYS:
        raise TokenizerArtifactError("tokenizer trainer metadata is invalid")
    for field in ("release_manifest_sha256", "corpus_manifest_sha256", "corpus_sha256"):
        if not isinstance(corpus[field], str) or not _SHA256.fullmatch(corpus[field]):
            raise TokenizerArtifactError(f"training corpus {field} is invalid")
    if type(corpus["corpus_size_bytes"]) is not int or corpus["corpus_size_bytes"] < 1:
        raise TokenizerArtifactError("training corpus size is invalid")
    if (
        not isinstance(corpus["dataset_id"], str)
        or not corpus["dataset_id"].strip()
        or len(corpus["dataset_id"]) > 128
        or not isinstance(corpus["dataset_version"], str)
        or not corpus["dataset_version"].strip()
        or len(corpus["dataset_version"]) > 64
        or not isinstance(corpus["license_ids"], list)
        or not corpus["license_ids"]
        or any(
            not isinstance(item, str) or not item.strip() or "replace" in item.casefold()
            for item in corpus["license_ids"]
        )
        or len(corpus["license_ids"]) != len(set(corpus["license_ids"]))
    ):
        raise TokenizerArtifactError("training corpus dataset/license metadata is invalid")
    if (
        type(trainer["requested_vocab_size"]) is not int
        or type(trainer["min_frequency"]) is not int
        or trainer["requested_vocab_size"] < 260
        or trainer["requested_vocab_size"] != manifest["vocab_size"]
        or trainer["min_frequency"] < 1
        or trainer["pre_tokenizer"] != "bytelevel"
        or trainer["normalizer"] != "none"
        or not isinstance(trainer["library_version"], str)
        or not trainer["library_version"].strip()
    ):
        raise TokenizerArtifactError("tokenizer trainer settings are invalid")
    try:
        created_at = datetime.fromisoformat(manifest["created_at"].replace("Z", "+00:00"))
    except (AttributeError, TypeError, ValueError) as exc:
        raise TokenizerArtifactError("created_at must be ISO-8601") from exc
    if created_at.tzinfo is None or created_at.utcoffset() is None:
        raise TokenizerArtifactError("created_at must include a timezone")
    actual_sha256, actual_size = _sha256_file(tokenizer_path)
    if actual_sha256 != manifest["tokenizer_sha256"] or actual_size != manifest["tokenizer_size_bytes"]:
        raise TokenizerArtifactError("tokenizer.json integrity does not match its manifest")
    try:
        from tokenizers import Tokenizer
        backend = Tokenizer.from_file(str(tokenizer_path))
    except ImportError as exc:
        raise RuntimeError("Install the tokenizer extra to load tokenizer artifacts") from exc
    except Exception as exc:
        raise TokenizerArtifactError(f"tokenizer.json could not be loaded: {exc}") from exc
    if backend.get_vocab_size(with_added_tokens=True) != manifest["vocab_size"]:
        raise TokenizerArtifactError("loaded tokenizer vocabulary size does not match its manifest")
    try:
        resolved = resolve_special_token_ids(backend)
    except ValueError as exc:
        raise TokenizerArtifactError(str(exc)) from exc
    expected = {name: special[name]["id"] for name in special}
    if resolved != expected:
        raise TokenizerArtifactError("loaded tokenizer special-token IDs do not match its manifest")
    return TokenizerArtifact(backend=backend, manifest=manifest)
