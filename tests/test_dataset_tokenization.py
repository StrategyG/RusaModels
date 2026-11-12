import hashlib
import json
from pathlib import Path

import pytest

from rusamodels.data_pipeline import (
    DatasetReleaseConfig,
    DatasetSource,
    build_dataset_release,
)
from rusamodels.dataset_tokenization import (
    DatasetTokenizationError,
    hash_tokenizer_directory,
    pack_dataset_split,
)


class FakeTokenizer:
    eos_token_id = 2
    bos_token_id = 1
    pad_token_id = 0
    vocab_size = 16
    model_max_length = 32

    def __len__(self) -> int:
        return self.vocab_size

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert add_special_tokens is False
        return [3 + (ord(character) % 10) for character in text]


class InvalidTokenizer(FakeTokenizer):
    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        return [99]


def _write_jsonl(path: Path, records: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )
    return path


def _release(tmp_path: Path) -> Path:
    records = [
        {
            "id": f"example-{index}",
            "group_id": f"document-{index}",
            "text": (
                f"Tokenizer packing example {index} includes a sufficiently long technical "
                "description for multiple fixed-length sequences."
            ),
        }
        for index in range(1, 4)
    ]
    source = _write_jsonl(tmp_path / "source.jsonl", records)
    return build_dataset_release(
        DatasetReleaseConfig(
            dataset_id="tokenization-test",
            version="1.0.0",
            sources=(DatasetSource("approved-source", source, "CC0-1.0"),),
            output_root=tmp_path / "releases",
            seed=53,
            validation_fraction=0.2,
            test_fraction=0.2,
        )
    )


def _pack(release: Path, output: Path, **kwargs):
    return pack_dataset_split(
        release,
        "train",
        output,
        tokenizer=kwargs.pop("tokenizer", FakeTokenizer()),
        tokenizer_id="example/tokenizer",
        tokenizer_revision="a" * 40,
        tokenizer_sha256="b" * 64,
        sequence_length=16,
        **kwargs,
    )


def test_pack_verified_split_into_fixed_length_sequences(tmp_path: Path) -> None:
    release = _release(tmp_path)
    output = tmp_path / "packed" / "train.jsonl"

    result = _pack(release, output, add_bos=True, add_eos=True)
    sequences = [
        json.loads(line)
        for line in output.read_text(encoding="utf-8").splitlines()
    ]
    manifest = json.loads(
        output.with_suffix(output.suffix + ".manifest.json").read_text(encoding="utf-8")
    )

    assert result["records"] == 1
    assert result["sequences"] >= 2
    assert all(len(item["input_ids"]) == 16 for item in sequences)
    assert all(len(item["labels"]) == 16 for item in sequences)
    assert all(len(item["attention_mask"]) == 16 for item in sequences)
    assert all(item["record_ids"] for item in sequences)
    assert sequences[0]["input_ids"][0] == FakeTokenizer.bos_token_id
    assert sequences[-1]["attention_mask"][-1] in {0, 1}
    if result["padding_tokens"]:
        assert -100 in sequences[-1]["labels"]
        assert 0 in sequences[-1]["attention_mask"]
    assert manifest["split"] == "train"
    assert manifest["tokenizer_revision"] == "a" * 40
    assert manifest["tokenizer_sha256"] == "b" * 64
    assert manifest["source_split_sha256"]
    assert manifest["output_sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()


def test_packer_rejects_tokenizer_ids_outside_vocabulary(tmp_path: Path) -> None:
    release = _release(tmp_path)
    with pytest.raises(DatasetTokenizationError, match="outside its vocabulary"):
        _pack(release, tmp_path / "bad.jsonl", tokenizer=InvalidTokenizer())


def test_packer_rejects_sequence_length_above_tokenizer_limit(tmp_path: Path) -> None:
    release = _release(tmp_path)
    tokenizer = FakeTokenizer()
    tokenizer.model_max_length = 8
    with pytest.raises(DatasetTokenizationError, match="model_max_length"):
        pack_dataset_split(
            release,
            "train",
            tmp_path / "too-long.jsonl",
            tokenizer=tokenizer,
            tokenizer_id="example/tokenizer",
            tokenizer_revision="a" * 40,
            tokenizer_sha256="b" * 64,
            sequence_length=16,
        )


def test_packer_rejects_unpinned_tokenizer_revision(tmp_path: Path) -> None:
    release = _release(tmp_path)
    with pytest.raises(DatasetTokenizationError, match="40-character commit SHA"):
        pack_dataset_split(
            release,
            "train",
            tmp_path / "unpinned.jsonl",
            tokenizer=FakeTokenizer(),
            tokenizer_id="example/tokenizer",
            tokenizer_revision="main",
            tokenizer_sha256="b" * 64,
            sequence_length=16,
        )


def test_packed_output_cannot_modify_immutable_release(tmp_path: Path) -> None:
    release = _release(tmp_path)
    with pytest.raises(DatasetTokenizationError, match="outside the immutable dataset release"):
        _pack(release, release / "train.tokens.jsonl")


def test_packer_refuses_to_overwrite_existing_output(tmp_path: Path) -> None:
    release = _release(tmp_path)
    output = tmp_path / "train.jsonl"
    output.write_text("existing", encoding="utf-8")
    with pytest.raises(DatasetTokenizationError, match="already exists"):
        _pack(release, output)
    assert output.read_text(encoding="utf-8") == "existing"


def test_tokenizer_directory_hash_is_stable_and_content_sensitive(tmp_path: Path) -> None:
    tokenizer_dir = tmp_path / "tokenizer"
    tokenizer_dir.mkdir()
    asset = tokenizer_dir / "tokenizer.json"
    asset.write_text('{"version":"1"}', encoding="utf-8")
    first = hash_tokenizer_directory(tokenizer_dir)
    assert first == hash_tokenizer_directory(tokenizer_dir)
    asset.write_text('{"version":"2"}', encoding="utf-8")
    assert first != hash_tokenizer_directory(tokenizer_dir)


def test_tokenizer_directory_rejects_symlink_assets(tmp_path: Path) -> None:
    tokenizer_dir = tmp_path / "tokenizer"
    tokenizer_dir.mkdir()
    outside = tmp_path / "vocab.json"
    outside.write_text("{}", encoding="utf-8")
    link = tokenizer_dir / "vocab.json"
    try:
        link.symlink_to(outside)
    except (OSError, NotImplementedError):
        pytest.skip("symlink creation is not supported in this environment")
    with pytest.raises(DatasetTokenizationError, match="symlink"):
        hash_tokenizer_directory(tokenizer_dir)
