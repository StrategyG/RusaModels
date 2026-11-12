import hashlib
import json
from pathlib import Path

import pytest

pytest.importorskip("tokenizers")

from rusamodels.tokenizers.loading import (  # noqa: E402
    TokenizerArtifactError,
    load_tokenizer_artifact,
)
from rusamodels.tokenizers.training import (  # noqa: E402
    TokenizerTrainingError,
    train_tokenizer,
)
from rusamodels.tokenizers.validation import audit_tokenizer  # noqa: E402


def _write_export(tmp_path: Path, *, split: str = "train", license_ids=None):
    corpus = tmp_path / "train.txt"
    lines = [
        f"RusaAi bahasa Indonesia technical architecture worker tokenizer model pipeline {index} "
        "mengembangkan sistem yang aman dan dapat diverifikasi."
        for index in range(500)
    ]
    corpus.write_text("\n\n".join(lines) + "\n", encoding="utf-8")
    payload = corpus.read_bytes()
    manifest = {
        "schema_version": "1.0.0",
        "exporter": "rusamodels.dataset_text_export",
        "dataset_id": "rusaai-tokenizer-pilot",
        "dataset_version": "0.1.0",
        "split": split,
        "license_ids": ["LicenseRef-Project-Owned"] if license_ids is None else license_ids,
        "release_manifest_sha256": "a" * 64,
        "source_split_sha256": "b" * 64,
        "source_split_file": "train.jsonl",
        "records": 500,
        "encoding": "utf-8",
        "tokenizer": "byte-level-v1",
        "output_file": corpus.name,
        "output_sha256": hashlib.sha256(payload).hexdigest(),
        "output_size_bytes": len(payload),
    }
    manifest_path = tmp_path / "train.txt.manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return corpus, manifest_path


def test_bpe_tokenizer_train_load_and_domain_round_trip(tmp_path: Path) -> None:
    corpus, manifest = _write_export(tmp_path)
    result = train_tokenizer(
        corpus,
        manifest,
        tmp_path / "tokenizer-v1",
        vocab_size=300,
        min_frequency=2,
    )
    artifact = load_tokenizer_artifact(tmp_path / "tokenizer-v1")

    assert result["vocab_size"] == 300
    assert artifact.special_token_ids == {"unk": 0, "bos": 1, "eos": 2, "pad": 3}
    samples = [
        "RusaAi mengembangkan model bahasa Indonesia.",
        "Tokenizer compatibility must be verified.",
        "def main():\n    return {'status': 'ready'}",
        '{"tool":"search_knowledge","arguments":{"query":"hak akses"}}',
    ]
    for text in samples:
        ids = artifact.encode(text)
        assert artifact.decode(ids) == text
    wrapped = artifact.encode("uji tokenizer", add_bos=True, add_eos=True)
    assert wrapped[0] == artifact.bos_token_id
    assert wrapped[-1] == artifact.eos_token_id
    assert artifact.decode(wrapped) == "uji tokenizer"
    assert result["audit"]["round_trip_passed"] is True


def test_tokenizer_training_rejects_validation_or_test_split(tmp_path: Path) -> None:
    corpus, manifest = _write_export(tmp_path, split="validation")
    with pytest.raises(TokenizerTrainingError, match="only the dataset train split"):
        train_tokenizer(corpus, manifest, tmp_path / "tokenizer-v1", vocab_size=300)


def test_tokenizer_training_rejects_unresolved_license_placeholder(tmp_path: Path) -> None:
    corpus, manifest = _write_export(
        tmp_path,
        license_ids=["LicenseRef-REPLACE-WITH-APPROVED-LICENSE"],
    )
    with pytest.raises(TokenizerTrainingError, match="placeholders"):
        train_tokenizer(corpus, manifest, tmp_path / "tokenizer-v1", vocab_size=300)


def test_tokenizer_training_rejects_corpus_digest_mismatch(tmp_path: Path) -> None:
    corpus, manifest = _write_export(tmp_path)
    corpus.write_text(corpus.read_text(encoding="utf-8") + "tampered\n", encoding="utf-8")
    with pytest.raises(TokenizerTrainingError, match="digest/size"):
        train_tokenizer(corpus, manifest, tmp_path / "tokenizer-v1", vocab_size=300)


def test_tokenizer_training_refuses_to_overwrite_existing_artifact(tmp_path: Path) -> None:
    corpus, manifest = _write_export(tmp_path)
    output = tmp_path / "existing"
    output.mkdir()
    with pytest.raises(TokenizerTrainingError, match="already exists"):
        train_tokenizer(corpus, manifest, output, vocab_size=300)


def test_tokenizer_loader_detects_tampered_tokenizer_file(tmp_path: Path) -> None:
    corpus, manifest = _write_export(tmp_path)
    train_tokenizer(corpus, manifest, tmp_path / "tokenizer-v1", vocab_size=300)
    tokenizer_path = tmp_path / "tokenizer-v1" / "tokenizer.json"
    tokenizer_path.write_text(tokenizer_path.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(TokenizerArtifactError, match="integrity"):
        load_tokenizer_artifact(tmp_path / "tokenizer-v1")


def test_tokenizer_audit_reports_indonesian_english_code_and_json() -> None:
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from rusamodels.tokenizers.vocabulary import SPECIAL_TOKENS, resolve_special_token_ids

    backend = Tokenizer(models.BPE(unk_token=SPECIAL_TOKENS["unk"]))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    backend.train_from_iterator(
        ["rusaai bahasa indonesia " * 200],
        trainer=trainers.BpeTrainer(
            vocab_size=270,
            min_frequency=1,
            special_tokens=list(SPECIAL_TOKENS.values()),
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            show_progress=False,
        ),
    )
    report = audit_tokenizer(backend, resolve_special_token_ids(backend))
    assert set(report["categories"]) == {"indonesian", "english", "code", "json"}
    assert report["round_trip_passed"] is True


def test_tokenizer_loader_rejects_manifest_with_non_bpe_implementation(tmp_path: Path) -> None:
    corpus, manifest = _write_export(tmp_path)
    artifact_dir = tmp_path / "tokenizer-v1"
    train_tokenizer(corpus, manifest, artifact_dir, vocab_size=300)
    tokenizer_path = artifact_dir / "tokenizer.json"
    tokenizer_payload = json.loads(tokenizer_path.read_text(encoding="utf-8"))
    tokenizer_payload["pre_tokenizer"] = {"type": "Whitespace"}
    tokenizer_path.write_text(json.dumps(tokenizer_payload), encoding="utf-8")

    artifact_manifest_path = artifact_dir / "rusa-tokenizer-manifest.json"
    artifact_manifest = json.loads(artifact_manifest_path.read_text(encoding="utf-8"))
    tokenizer_bytes = tokenizer_path.read_bytes()
    artifact_manifest["tokenizer_sha256"] = hashlib.sha256(tokenizer_bytes).hexdigest()
    artifact_manifest["tokenizer_size_bytes"] = len(tokenizer_bytes)
    artifact_manifest_path.write_text(json.dumps(artifact_manifest), encoding="utf-8")

    with pytest.raises(TokenizerArtifactError, match="ByteLevel BPE contract"):
        load_tokenizer_artifact(artifact_dir)
