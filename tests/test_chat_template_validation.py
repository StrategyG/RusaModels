import pytest

from rusamodels.tokenizers.validation import validate_chat_template


class FakeChatTokenizer:
    chat_template = "{{ messages }}"
    eos_token_id = 2
    pad_token_id = 0

    def __len__(self):
        return 32

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        assert tokenize is True
        assert add_generation_prompt is True
        assert any(message["role"] == "user" for message in messages)
        return [1, 4, 7, 2]

    def decode(self, token_ids, *, skip_special_tokens):
        assert skip_special_tokens is False
        return "<|system|>RusaAi<|user|>Jelaskan<|assistant|>"


def test_foundation_chat_template_and_generation_prompt_are_compatible() -> None:
    report = validate_chat_template(FakeChatTokenizer())

    assert report["chat_template_present"] is True
    assert report["generation_prompt_passed"] is True
    assert report["prompt_tokens"] == 4
    assert report["eos_token_id"] == 2


def test_chat_template_validator_rejects_missing_template() -> None:
    tokenizer = FakeChatTokenizer()
    tokenizer.chat_template = None

    with pytest.raises(ValueError, match="non-empty chat_template"):
        validate_chat_template(tokenizer)


def test_chat_template_validator_rejects_out_of_vocabulary_ids() -> None:
    tokenizer = FakeChatTokenizer()
    tokenizer.apply_chat_template = lambda *args, **kwargs: [1, 32]

    with pytest.raises(ValueError, match="out-of-vocabulary"):
        validate_chat_template(tokenizer)


def test_chat_template_validator_rejects_empty_message_list() -> None:
    with pytest.raises(ValueError, match="valid system/user/assistant turns"):
        validate_chat_template(FakeChatTokenizer(), messages=[])
