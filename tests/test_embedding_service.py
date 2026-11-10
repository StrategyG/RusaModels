import pytest
from fastapi import HTTPException

from services.embedding_server.app.main import _authorize, _prepare_inputs


def test_e5_embedding_inputs_use_query_and_passage_prefixes():
    assert _prepare_inputs(["find a document"], "query") == ["query: find a document"]
    assert _prepare_inputs(["document content"], "passage") == ["passage: document content"]
    assert _prepare_inputs(["query: already tagged"], "passage") == ["query: already tagged"]


def test_embedding_runtime_requires_internal_bearer_token(monkeypatch):
    monkeypatch.setenv("RUSAMODELS_EMBEDDING_INTERNAL_TOKEN", "x" * 32)
    with pytest.raises(HTTPException) as error:
        _authorize("Bearer " + "y" * 32)
    assert error.value.status_code == 401
    _authorize("Bearer " + "x" * 32)
