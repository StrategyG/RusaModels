"""Local CPU embedding API for RusaModels; model weights must be mounted locally."""
from __future__ import annotations

import hmac
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, Header, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field


class EmbeddingRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: str | None = None
    input: str | list[str]
    input_type: Literal["query", "passage"] = "passage"
    encoding_format: Literal["float"] = "float"


def _load_embedding_model():
    model_path = os.getenv("RUSAMODELS_EMBEDDING_MODEL_PATH", "")
    if not model_path:
        raise RuntimeError("RUSAMODELS_EMBEDDING_MODEL_PATH must point to a local model directory")
    path = Path(model_path).expanduser().resolve(strict=True)
    if not path.is_dir():
        raise RuntimeError("Embedding model path must be a directory")
    internal_token = os.getenv("RUSAMODELS_EMBEDDING_INTERNAL_TOKEN", "")
    if len(internal_token) < 32:
        raise RuntimeError("RUSAMODELS_EMBEDDING_INTERNAL_TOKEN must contain at least 32 characters")

    import torch
    from sentence_transformers import SentenceTransformer

    torch.set_num_threads(max(1, int(os.getenv("RUSAMODELS_EMBEDDING_CPU_THREADS", "1"))))
    return SentenceTransformer(
        str(path),
        device="cpu",
        trust_remote_code=False,
        model_kwargs={"local_files_only": True},
    )


@asynccontextmanager
async def lifespan(application: FastAPI):
    application.state.model = _load_embedding_model()
    application.state.model_id = os.getenv("RUSAMODELS_EMBEDDING_MODEL_ID", "rusa-embedding-small")
    yield
    application.state.model = None


app = FastAPI(
    title="RusaModels Local Embedding Runtime",
    version="1.0.0",
    description="CPU-only local embedding service; no external inference API is used.",
    lifespan=lifespan,
)


def _authorize(authorization: str | None) -> None:
    expected = os.getenv("RUSAMODELS_EMBEDDING_INTERNAL_TOKEN", "")
    supplied = authorization.removeprefix("Bearer ").strip() if authorization else ""
    if not expected or not hmac.compare_digest(supplied, expected):
        raise HTTPException(status_code=401, detail="Invalid embedding service credential")


@app.get("/health/live")
def live():
    return {"status": "alive"}


@app.get("/health/ready")
def ready(request: Request):
    if getattr(request.app.state, "model", None) is None:
        raise HTTPException(status_code=503, detail="Embedding model is not loaded")
    return {"status": "ready", "model": request.app.state.model_id, "device": "cpu"}


@app.get("/v1/models")
def models(request: Request, authorization: str | None = Header(default=None)):
    _authorize(authorization)
    return {
        "object": "list",
        "data": [{"id": request.app.state.model_id, "object": "model", "owned_by": "rusamodels"}],
    }


@app.post("/v1/embeddings")
def embeddings(
    payload: EmbeddingRequest,
    request: Request,
    authorization: str | None = Header(default=None),
):
    _authorize(authorization)
    model_id = request.app.state.model_id
    if payload.model not in {None, model_id, "default"}:
        raise HTTPException(status_code=404, detail="Embedding model is not configured")
    texts = [payload.input] if isinstance(payload.input, str) else payload.input
    if not texts or len(texts) > 128:
        raise HTTPException(status_code=422, detail="input must contain between 1 and 128 texts")
    if any(not isinstance(item, str) or not item.strip() or len(item) > 100_000 for item in texts):
        raise HTTPException(status_code=422, detail="Every input must be non-empty text up to 100000 characters")

    prefix = "query: " if payload.input_type == "query" else "passage: "
    normalized = [item if item.startswith(("query: ", "passage: ")) else prefix + item for item in texts]
    model = request.app.state.model
    vectors = model.encode(
        normalized,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    tokenized = model.tokenize(normalized)
    attention_mask = tokenized.get("attention_mask")
    prompt_tokens = int(attention_mask.sum().item()) if attention_mask is not None else 0
    return {
        "object": "list",
        "model": model_id,
        "data": [
            {"object": "embedding", "index": index, "embedding": vector.astype(float).tolist()}
            for index, vector in enumerate(vectors)
        ],
        "usage": {"prompt_tokens": prompt_tokens, "total_tokens": prompt_tokens},
    }
