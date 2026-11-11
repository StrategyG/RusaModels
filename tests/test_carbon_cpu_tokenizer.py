from __future__ import annotations

import pytest

from rusamodels.carbon_cpu.tokenizer import (
    BOS_TOKEN_ID,
    BYTE_VOCAB_SIZE,
    EOS_TOKEN_ID,
    CarbonByteTokenizer,
)


def test_utf8_byte_tokenizer_round_trip() -> None:
    tokenizer = CarbonByteTokenizer()
    text = "RusaAi — model milik sendiri 🇮🇩"
    tokens = tokenizer.encode(text)
    assert all(0 <= token <= 255 for token in tokens)
    assert tokenizer.decode(tokens) == text
    assert tokenizer.vocab_size == BYTE_VOCAB_SIZE == 258


def test_byte_tokenizer_special_tokens() -> None:
    tokenizer = CarbonByteTokenizer()
    tokens = tokenizer.encode("hello", add_bos=True, add_eos=True)
    assert tokens[0] == BOS_TOKEN_ID
    assert tokens[-1] == EOS_TOKEN_ID
    assert tokenizer.decode(tokens) == "hello"
    assert tokenizer.decode(tokens, skip_special_tokens=False) == "<|bos|>hello<|eos|>"


def test_byte_tokenizer_rejects_invalid_ids() -> None:
    with pytest.raises(ValueError, match="outside the byte vocabulary"):
        CarbonByteTokenizer().decode([999])
