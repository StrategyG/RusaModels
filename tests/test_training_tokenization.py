import importlib.util
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

SCRIPT = Path(__file__).resolve().parents[1] / "rusamodels" / "training" / "train_sft.py"
SPEC = importlib.util.spec_from_file_location("rusa_train_sft", SCRIPT)
TRAIN = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(TRAIN)


class FakeTokenizer:
    eos_token_id = 9
    pad_token_id = 0

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        assert tokenize is True
        assert add_generation_prompt is True
        return [1, 2, 3]

    def encode(self, text, *, add_special_tokens):
        assert add_special_tokens is False
        return [4, 5]


def record():
    return {
        "id": "unit-1",
        "agent_role": "analyst",
        "messages": [
            {"role": "system", "content": "System"},
            {"role": "user", "content": "Inspect"},
        ],
        "response": {"message": "Done", "actions": [], "done": True},
    }


def test_completion_only_loss_masks_prompt_tokens():
    dataset = TRAIN._build_torch_dataset([record()], FakeTokenizer(), 16, torch)
    item = dataset[0]
    assert item["input_ids"] == [1, 2, 3, 4, 5, 9]
    assert item["labels"] == [-100, -100, -100, 4, 5, 9]


def test_long_examples_are_rejected_instead_of_truncated():
    with pytest.raises(ValueError, match="silently truncating"):
        TRAIN._build_torch_dataset([record()], FakeTokenizer(), 5, torch)


def test_collator_pads_labels_with_ignore_index():
    collate = TRAIN._collator(FakeTokenizer(), torch)
    batch = collate([
        {"input_ids": [1, 2], "attention_mask": [1, 1], "labels": [-100, 2]},
        {"input_ids": [1], "attention_mask": [1], "labels": [-100]},
    ])
    assert batch["input_ids"].tolist() == [[1, 2], [1, 0]]
    assert batch["attention_mask"].tolist() == [[1, 1], [1, 0]]
    assert batch["labels"].tolist() == [[-100, 2], [-100, -100]]
