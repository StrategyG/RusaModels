"""Versioned training-time snapshot of the RUSA worker contract.

This module validates training examples only. It is not an authorization engine.
RUSA AI's runtime Tool Gateway and Policy Engine remain authoritative at execution.
Source snapshot: StrategyG/RusaAi, commit c2866e2d9747b9de4ea9d5f9696b8ded11121006.
"""
from enum import Enum


class WorkerRole(str, Enum):
    ANALYST = "analyst"
    ARCHITECT = "architect"
    DEVELOPER = "developer"
    DEBUGGER = "debugger"
    TESTER = "tester"
    SECURITY = "security"


ROLE_TOOL_ALLOWLIST = {
    WorkerRole.ANALYST: frozenset({
        "list_files", "read_file", "git_diff",
        "search_knowledge", "get_document", "list_collections",
    }),
    WorkerRole.ARCHITECT: frozenset({
        "list_files", "read_file", "git_diff",
        "search_knowledge", "get_document", "list_collections",
    }),
    WorkerRole.DEVELOPER: frozenset({
        "list_files", "read_file", "write_file", "git_diff",
        "search_knowledge", "get_document", "list_collections",
    }),
    WorkerRole.DEBUGGER: frozenset({
        "list_files", "read_file", "run_check", "git_diff",
        "search_knowledge", "get_document", "list_collections",
    }),
    WorkerRole.TESTER: frozenset({
        "list_files", "read_file", "run_check", "git_diff",
        "search_knowledge", "get_document", "list_collections",
    }),
    WorkerRole.SECURITY: frozenset({
        "list_files", "read_file", "run_check", "git_diff",
        "search_knowledge", "get_document", "list_collections",
    }),
}

ROLE_CHECK_PROFILES = {
    WorkerRole.TESTER: frozenset({"pytest"}),
    WorkerRole.DEBUGGER: frozenset({"pytest", "compile", "ruff"}),
    WorkerRole.SECURITY: frozenset({"bandit", "ruff", "compile"}),
}
