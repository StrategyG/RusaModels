"""Strict JSONL dataset contracts for RusaModels agent SFT."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Any

from rusamodels.training.agent_contract import ROLE_CHECK_PROFILES, ROLE_TOOL_ALLOWLIST, WorkerRole


class DatasetValidationError(ValueError):
    """Raised when a training example violates the agent contract or policy."""


_RECORD_KEYS = {"id", "agent_role", "messages", "response"}
_RESPONSE_KEYS = {"message", "actions", "done", "blocked", "blockers"}
_ACTION_ARGUMENTS = {
    "list_files": {"path"},
    "read_file": {"path"},
    "write_file": {"path", "content"},
    "run_check": {"profile"},
    "git_diff": set(),
    "search_knowledge": {"query", "collection", "limit"},
    "get_document": {"document_id", "collection"},
    "list_collections": set(),
}
_REQUIRED_ARGUMENTS = {
    "read_file": {"path"},
    "write_file": {"path", "content"},
    "run_check": {"profile"},
    "search_knowledge": {"query"},
    "get_document": {"document_id"},
}
_PROTECTED_PATH_PARTS = {
    ".git", ".secrets", "secrets", "credentials", ".ssh", ".aws", ".azure",
    ".terraform", ".venv", "venv", "node_modules",
}
_PROTECTED_WRITE_NAMES = {
    ".env.example", ".gitignore", ".dockerignore", "requirements.txt",
    "requirements-agent-sandbox.txt", "pyproject.toml", "dockerfile",
    "docker-compose.yml", "docker-compose.yaml", "compose.yml", "compose.yaml",
    "containerfile", "makefile", "procfile", "setup.py", "setup.cfg",
    "package.json", "package-lock.json", "pnpm-lock.yaml", "yarn.lock",
    "poetry.lock", "uv.lock", "pipfile", "pipfile.lock", "cargo.toml", "go.mod",
}
_PROTECTED_WRITE_PREFIXES = (
    ".github/", "migrations/", "schemas/", "llm-project/configs/", "deploy/",
    "deployment/", "infra/", "infrastructure/", "terraform/", "k8s/",
    "kubernetes/", "helm/", "systemd/", ".devcontainer/",
)


def _role(value: Any) -> WorkerRole:
    if not isinstance(value, str):
        raise DatasetValidationError("agent_role must be a string")
    try:
        return WorkerRole(value)
    except ValueError as exc:
        raise DatasetValidationError(f"unsupported agent_role: {value}") from exc


def _validate_relative_path(value: Any, *, write: bool = False, allow_dot: bool = False) -> None:
    if not isinstance(value, str) or not value.strip():
        raise DatasetValidationError("tool path must be a non-empty string")
    if allow_dot and value == ".":
        return
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or "\\" in value:
        raise DatasetValidationError("absolute paths, traversal, and backslashes are prohibited")
    lowered_parts = {part.lower() for part in path.parts}
    if lowered_parts & _PROTECTED_PATH_PARTS or any(
        part.startswith(".env") and part != ".env.example" for part in lowered_parts
    ):
        raise DatasetValidationError("tool path targets protected data")
    if write:
        normalized = path.as_posix().lower()
        if (
            path.name.lower() in _PROTECTED_WRITE_NAMES
            or path.name.lower().startswith(("requirements", "dockerfile.", "docker-compose.", "compose."))
            or any(normalized.startswith(prefix) for prefix in _PROTECTED_WRITE_PREFIXES)
        ):
            raise DatasetValidationError("write_file targets a human-reviewed path")


def validate_agent_response(agent_role: str | WorkerRole, response: Any) -> None:
    """Validate the exact JSON envelope consumed by AutonomousCodeWorker."""
    role = agent_role if isinstance(agent_role, WorkerRole) else _role(agent_role)
    if not isinstance(response, dict):
        raise DatasetValidationError("response must be a JSON object")
    if set(response) - _RESPONSE_KEYS:
        raise DatasetValidationError("response contains unsupported fields")
    if not {"message", "actions", "done"}.issubset(response):
        raise DatasetValidationError("response requires message, actions, and done")
    if not isinstance(response["message"], str) or not response["message"].strip():
        raise DatasetValidationError("response.message must be a non-empty string")
    if not isinstance(response["actions"], list):
        raise DatasetValidationError("response.actions must be an array")
    if len(response["actions"]) > 8:
        raise DatasetValidationError("response.actions exceeds the runtime limit of 8")
    if type(response["done"]) is not bool:
        raise DatasetValidationError("response.done must be a boolean")
    blocked = response.get("blocked", False)
    if type(blocked) is not bool:
        raise DatasetValidationError("response.blocked must be a boolean")
    blockers = response.get("blockers", [])
    if not isinstance(blockers, list) or any(not isinstance(item, str) for item in blockers):
        raise DatasetValidationError("response.blockers must be an array of strings")
    if blocked and (not response["done"] or response["actions"]):
        raise DatasetValidationError("blocked responses must be terminal and contain no actions")
    if not response["done"] and not response["actions"] and not blocked:
        raise DatasetValidationError("non-terminal responses must request at least one action")

    allowed_tools = ROLE_TOOL_ALLOWLIST[role]
    for index, action in enumerate(response["actions"]):
        if not isinstance(action, dict) or set(action) != {"name", "arguments"}:
            raise DatasetValidationError(f"actions[{index}] must contain name and arguments only")
        name = action["name"]
        args = action["arguments"]
        if not isinstance(name, str) or name not in allowed_tools:
            raise DatasetValidationError(f"actions[{index}] uses a tool not allowed for {role.value}")
        if not isinstance(args, dict):
            raise DatasetValidationError(f"actions[{index}].arguments must be an object")
        if set(args) - _ACTION_ARGUMENTS[name]:
            raise DatasetValidationError(f"actions[{index}] contains unsupported tool arguments")
        if not _REQUIRED_ARGUMENTS.get(name, set()).issubset(args):
            raise DatasetValidationError(f"actions[{index}] is missing required tool arguments")

        if name == "list_files":
            _validate_relative_path(args.get("path", "."), allow_dot=True)
        elif name == "read_file":
            _validate_relative_path(args["path"])
        elif name == "write_file":
            _validate_relative_path(args["path"], write=True)
            if not isinstance(args["content"], str) or len(args["content"].encode("utf-8")) > 1_000_000:
                raise DatasetValidationError("write_file content must be text up to 1 MB")
        elif name == "run_check":
            profile = args["profile"]
            if not isinstance(profile, str) or profile not in ROLE_CHECK_PROFILES.get(role, frozenset()):
                raise DatasetValidationError(f"run_check profile is not allowed for {role.value}")
        elif name == "search_knowledge":
            if not isinstance(args["query"], str) or not args["query"].strip() or len(args["query"]) > 2000:
                raise DatasetValidationError("search_knowledge query is invalid")
            if "collection" in args and (not isinstance(args["collection"], str) or not args["collection"]):
                raise DatasetValidationError("search_knowledge collection is invalid")
            if "limit" in args and (type(args["limit"]) is not int or not 1 <= args["limit"] <= 10):
                raise DatasetValidationError("search_knowledge limit must be between 1 and 10")
        elif name == "get_document":
            if not isinstance(args["document_id"], str) or not args["document_id"] or len(args["document_id"]) > 64:
                raise DatasetValidationError("get_document document_id is invalid")
            if "collection" in args and (not isinstance(args["collection"], str) or not args["collection"]):
                raise DatasetValidationError("get_document collection is invalid")


def validate_record(record: Any) -> dict[str, Any]:
    if not isinstance(record, dict) or set(record) != _RECORD_KEYS:
        raise DatasetValidationError("record must contain exactly id, agent_role, messages, and response")
    if not isinstance(record["id"], str) or not record["id"].strip():
        raise DatasetValidationError("record.id must be a non-empty string")
    role = _role(record["agent_role"])
    messages = record["messages"]
    if not isinstance(messages, list) or len(messages) < 2:
        raise DatasetValidationError("messages must contain at least a system and user message")
    for index, message in enumerate(messages):
        if not isinstance(message, dict) or set(message) != {"role", "content"}:
            raise DatasetValidationError(f"messages[{index}] must contain role and content only")
        if not isinstance(message["role"], str) or message["role"] not in {"system", "user", "assistant"}:
            raise DatasetValidationError(f"messages[{index}] has an unsupported role")
        if not isinstance(message["content"], str) or not message["content"].strip():
            raise DatasetValidationError(f"messages[{index}].content must be non-empty text")
    if messages[0]["role"] != "system" or messages[-1]["role"] != "user":
        raise DatasetValidationError("messages must start with system and end with user")
    validate_agent_response(role, record["response"])
    return record


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    records: list[dict[str, Any]] = []
    ids: set[str] = set()
    fingerprints: set[str] = set()
    try:
        lines = source.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise DatasetValidationError(f"cannot read dataset {source}: {exc}") from exc
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            record = validate_record(json.loads(line))
        except (json.JSONDecodeError, DatasetValidationError) as exc:
            raise DatasetValidationError(f"{source}:{line_number}: {exc}") from exc
        if record["id"] in ids:
            raise DatasetValidationError(f"{source}:{line_number}: duplicate record id {record['id']}")
        ids.add(record["id"])
        fingerprint = record_fingerprint(record)
        if fingerprint in fingerprints:
            raise DatasetValidationError(f"{source}:{line_number}: duplicate example content")
        fingerprints.add(fingerprint)
        records.append(record)
    if not records:
        raise DatasetValidationError(f"dataset is empty: {source}")
    return records


def record_fingerprint(record: dict[str, Any]) -> str:
    payload = json.dumps(
        {"agent_role": record["agent_role"], "messages": record["messages"], "response": record["response"]},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def validate_dataset_splits(train_records: list[dict[str, Any]], eval_records: list[dict[str, Any]]) -> None:
    train_ids = {item["id"] for item in train_records}
    eval_ids = {item["id"] for item in eval_records}
    if train_ids & eval_ids:
        raise DatasetValidationError("training and evaluation splits reuse record IDs")
    train_fingerprints = {record_fingerprint(item) for item in train_records}
    overlap = train_fingerprints & {record_fingerprint(item) for item in eval_records}
    if overlap:
        raise DatasetValidationError("training and evaluation splits contain duplicate examples")


def validate_eval_distribution(
    eval_records: list[dict[str, Any]],
    *,
    min_blocked_fraction: float = 0.10,
    min_unblocked_fraction: float = 0.50,
) -> None:
    if not eval_records:
        raise DatasetValidationError("evaluation split is empty")
    blocked_count = sum(bool(item["response"].get("blocked", False)) for item in eval_records)
    total = len(eval_records)
    blocked_fraction = blocked_count / total
    unblocked_fraction = 1.0 - blocked_fraction
    if blocked_fraction < min_blocked_fraction:
        raise DatasetValidationError(
            f"blocked eval fraction {blocked_fraction:.1%} is below {min_blocked_fraction:.1%}"
        )
    if unblocked_fraction < min_unblocked_fraction:
        raise DatasetValidationError(
            f"unblocked eval fraction {unblocked_fraction:.1%} is below {min_unblocked_fraction:.1%}"
        )


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
