import json

import pytest

from rusamodels.training.dataset import (
    DatasetValidationError,
    load_jsonl,
    validate_agent_response,
    validate_dataset_splits,
    validate_eval_distribution,
    validate_record,
)


def make_record(record_id="analyst-1", role="analyst", response=None):
    return {
        "id": record_id,
        "agent_role": role,
        "messages": [
            {"role": "system", "content": "Return one JSON object only."},
            {"role": "user", "content": "Inspect the repository structure."},
        ],
        "response": response or {
            "message": "I will inspect the repository structure.",
            "actions": [{"name": "list_files", "arguments": {"path": "."}}],
            "done": True,
        },
    }


def test_valid_agent_example_is_accepted():
    assert validate_record(make_record())["agent_role"] == "analyst"


def test_unknown_worker_role_is_rejected():
    with pytest.raises(DatasetValidationError, match="unsupported agent_role"):
        validate_record(make_record(role="release"))


def test_tool_outside_role_allowlist_is_rejected():
    record = make_record(response={
        "message": "I will run tests.",
        "actions": [{"name": "run_check", "arguments": {"profile": "pytest"}}],
        "done": True,
    })
    with pytest.raises(DatasetValidationError, match="not allowed for analyst"):
        validate_record(record)


def test_tester_cannot_run_security_profile():
    response = {
        "message": "I will inspect security findings.",
        "actions": [{"name": "run_check", "arguments": {"profile": "bandit"}}],
        "done": True,
    }
    with pytest.raises(DatasetValidationError, match="profile is not allowed"):
        validate_agent_response("tester", response)


def test_developer_cannot_write_protected_config():
    response = {
        "message": "I will change the dependency manifest.",
        "actions": [{"name": "write_file", "arguments": {"path": "pyproject.toml", "content": "[project]"}}],
        "done": True,
    }
    with pytest.raises(DatasetValidationError, match="human-reviewed path"):
        validate_agent_response("developer", response)


def test_blocked_response_must_not_contain_actions():
    response = {
        "message": "I cannot access credentials.",
        "actions": [{"name": "read_file", "arguments": {"path": "README.md"}}],
        "done": True,
        "blocked": True,
        "blockers": ["Credential access is prohibited."],
    }
    with pytest.raises(DatasetValidationError, match="no actions"):
        validate_agent_response("security", response)


def test_nonterminal_response_must_request_an_action():
    response = {"message": "I need another step.", "actions": [], "done": False}
    with pytest.raises(DatasetValidationError, match="at least one action"):
        validate_agent_response("analyst", response)


def test_dataset_rejects_duplicate_ids_and_content(tmp_path):
    first = make_record("same-id")
    second = make_record("same-id")
    path = tmp_path / "train.jsonl"
    path.write_text("\n".join(json.dumps(item) for item in (first, second)), encoding="utf-8")
    with pytest.raises(DatasetValidationError, match="duplicate record id"):
        load_jsonl(path)


def test_train_eval_split_leakage_is_rejected():
    record = make_record()
    with pytest.raises(DatasetValidationError, match="split"):
        validate_dataset_splits([record], [record])


def test_message_items_are_type_checked_before_role_access():
    record = make_record()
    record["messages"][0] = "not-an-object"
    with pytest.raises(DatasetValidationError, match=r"messages\[0\]"):
        validate_record(record)


def test_train_eval_split_rejects_same_content_with_different_ids():
    train = make_record("train-1")
    evaluation = make_record("eval-1")
    with pytest.raises(DatasetValidationError, match="duplicate examples"):
        validate_dataset_splits([train], [evaluation])


def test_eval_distribution_requires_both_blocked_and_unblocked_cases():
    blocked = make_record("blocked-1", response={
        "message": "I cannot access credentials.",
        "actions": [],
        "done": True,
        "blocked": True,
        "blockers": ["Credential access is prohibited."],
    })
    unblocked = [make_record(f"normal-{index}") for index in range(9)]
    validate_eval_distribution([blocked, *unblocked], min_blocked_fraction=0.10, min_unblocked_fraction=0.50)
    with pytest.raises(DatasetValidationError, match="unblocked eval fraction"):
        validate_eval_distribution([blocked], min_blocked_fraction=0.10, min_unblocked_fraction=0.50)


def test_training_dataset_rejects_duplicate_json_keys(tmp_path):
    path = tmp_path / "duplicate-key.jsonl"
    path.write_text(
        '{"id":"first","id":"second","agent_role":"analyst","messages":[],"response":{}}\n',
        encoding="utf-8",
    )
    with pytest.raises(DatasetValidationError, match="duplicate JSON object key"):
        load_jsonl(path)


def test_training_dataset_rejects_non_finite_json_numbers(tmp_path):
    path = tmp_path / "non-finite.jsonl"
    path.write_text(
        '{"id":"one","agent_role":"analyst","messages":[],"response":{"message":"x","actions":[],"done":true,"extra":NaN}}\n',
        encoding="utf-8",
    )
    with pytest.raises(DatasetValidationError, match="non-standard JSON constant"):
        load_jsonl(path)
