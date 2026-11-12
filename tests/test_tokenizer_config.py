import importlib.util
from pathlib import Path

import pytest

from rusamodels.tokenizers.training import TokenizerTrainingError


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "train_rusa_tokenizer.py"
SPEC = importlib.util.spec_from_file_location("train_rusa_tokenizer_cli", SCRIPT)
CLI = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(CLI)


def test_versioned_research_tokenizer_config_is_valid() -> None:
    config = CLI._load_config(
        ROOT / "configs/tokenizers/rusaai-research-100m-bpe.json"
    )

    assert config["vocab_size"] == 32000
    assert config["training_split"] == "train"
    assert config["special_tokens"]["eos"] == "[EOS]"


def test_tokenizer_config_rejects_duplicate_keys(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.json"
    path.write_text('{"schema_version":"1.0.0","schema_version":"1.0.0"}', encoding="utf-8")

    with pytest.raises(TokenizerTrainingError, match="duplicate tokenizer config key"):
        CLI._load_config(path)


def test_tokenizer_config_rejects_special_token_contract_drift(tmp_path: Path) -> None:
    source = ROOT / "configs/tokenizers/rusaai-research-100m-bpe.json"
    payload = __import__("json").loads(source.read_text(encoding="utf-8"))
    payload["special_tokens"]["eos"] = "<|END|>"
    path = tmp_path / "wrong-special-tokens.json"
    path.write_text(__import__("json").dumps(payload), encoding="utf-8")

    with pytest.raises(TokenizerTrainingError, match="special tokens differ"):
        CLI._load_config(path)
