from collections import Counter
from pathlib import Path

from rusamodels.training.dataset import (
    load_jsonl,
    validate_dataset_splits,
    validate_eval_distribution,
)

ROOT = Path(__file__).resolve().parents[1]
TRAIN_FILE = ROOT / "data" / "agent_sft" / "train.jsonl"
EVAL_FILE = ROOT / "data" / "agent_sft" / "eval.jsonl"
EXPECTED_ROLES = {"analyst", "architect", "developer", "debugger", "tester", "security"}


def test_curated_sft_dataset_meets_seed_experiment_guardrails():
    train = load_jsonl(TRAIN_FILE)
    evaluation = load_jsonl(EVAL_FILE)

    assert len(train) == 120
    assert len(evaluation) == 36
    assert Counter(item["agent_role"] for item in train) == {
        role: 20 for role in EXPECTED_ROLES
    }
    assert Counter(item["agent_role"] for item in evaluation) == {
        role: 6 for role in EXPECTED_ROLES
    }

    validate_dataset_splits(train, evaluation)
    validate_eval_distribution(
        evaluation,
        min_blocked_fraction=0.10,
        min_unblocked_fraction=0.50,
    )

    assert sum(bool(item["response"].get("blocked", False)) for item in evaluation) == 12
