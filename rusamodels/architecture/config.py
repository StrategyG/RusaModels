"""Strict, versioned configuration for the supported dense decoder Transformer v1."""
from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


class ArchitectureConfigError(ValueError):
    """Raised when an architecture blueprint is malformed or unsupported."""


_MODEL_ID = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")
_BLUEPRINT_KEYS = {"schema_version", "model_id", "model_version", "status", "architecture"}
_BLUEPRINT_STATUSES = {
    "design_only",
    "design_ready_training_not_started",
    "candidate",
}


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant is prohibited: {value}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


@dataclass(frozen=True)
class DecoderArchitectureConfig:
    """Supported pre-norm dense decoder with GQA, RoPE, RMSNorm and SwiGLU."""

    architecture_type: str = "dense_decoder_transformer"
    norm_layout: str = "pre_norm"
    vocab_size: int
    layers: int
    hidden_size: int
    query_heads: int
    kv_heads: int
    intermediate_size: int
    context_length: int
    position_encoding: str = "rope"
    normalization: str = "rmsnorm"
    rms_norm_eps: float = 1e-5
    activation: str = "swiglu"
    tied_embeddings: bool = True
    attention_bias: bool = False
    dropout: float = 0.0
    rope_theta: float = 10000.0
    eos_token_id: int | None = None
    expected_parameters: int | None = None

    def __post_init__(self) -> None:
        if self.architecture_type != "dense_decoder_transformer":
            raise ArchitectureConfigError("architecture_type must be dense_decoder_transformer")
        if self.norm_layout != "pre_norm":
            raise ArchitectureConfigError("architecture v1 supports norm_layout=pre_norm only")
        integer_fields = {
            "vocab_size": self.vocab_size,
            "layers": self.layers,
            "hidden_size": self.hidden_size,
            "query_heads": self.query_heads,
            "kv_heads": self.kv_heads,
            "intermediate_size": self.intermediate_size,
            "context_length": self.context_length,
        }
        for name, value in integer_fields.items():
            if type(value) is not int or value < 1:
                raise ArchitectureConfigError(f"{name} must be a positive integer")
        if self.context_length < 2:
            raise ArchitectureConfigError("context_length must be at least 2")
        if self.hidden_size % self.query_heads:
            raise ArchitectureConfigError("hidden_size must be divisible by query_heads")
        if self.query_heads % self.kv_heads:
            raise ArchitectureConfigError("query_heads must be divisible by kv_heads")
        if self.head_dim % 2:
            raise ArchitectureConfigError("head_dim must be even for RoPE")
        if not isinstance(self.position_encoding, str) or self.position_encoding != "rope":
            raise ArchitectureConfigError("architecture v1 supports position_encoding=rope only")
        if not isinstance(self.normalization, str) or self.normalization != "rmsnorm":
            raise ArchitectureConfigError("architecture v1 supports normalization=rmsnorm only")
        if not isinstance(self.activation, str) or self.activation != "swiglu":
            raise ArchitectureConfigError("architecture v1 supports activation=swiglu only")
        if type(self.tied_embeddings) is not bool or not self.tied_embeddings:
            raise ArchitectureConfigError("architecture v1 requires tied_embeddings=true")
        if type(self.attention_bias) is not bool or self.attention_bias:
            raise ArchitectureConfigError("architecture v1 requires attention_bias=false")
        for name, value in (
            ("dropout", self.dropout),
            ("rope_theta", self.rope_theta),
            ("rms_norm_eps", self.rms_norm_eps),
        ):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ArchitectureConfigError(f"{name} must be numeric")
            if not math.isfinite(float(value)):
                raise ArchitectureConfigError(f"{name} must be finite")
        if not 0.0 <= self.dropout < 1.0:
            raise ArchitectureConfigError("dropout must be in the range [0, 1)")
        if self.rope_theta <= 0:
            raise ArchitectureConfigError("rope_theta must be positive")
        if self.rms_norm_eps <= 0:
            raise ArchitectureConfigError("rms_norm_eps must be positive")
        if self.eos_token_id is not None and (
            type(self.eos_token_id) is not int
            or not 0 <= self.eos_token_id < self.vocab_size
        ):
            raise ArchitectureConfigError("eos_token_id must be inside the configured vocabulary")
        if self.expected_parameters is not None:
            if type(self.expected_parameters) is not int or self.expected_parameters < 1:
                raise ArchitectureConfigError("expected_parameters must be a positive integer")
            if self.expected_parameters != self.parameter_count:
                raise ArchitectureConfigError(
                    f"expected_parameters={self.expected_parameters} does not match "
                    f"calculated parameter_count={self.parameter_count}"
                )

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.query_heads

    @property
    def kv_group_size(self) -> int:
        return self.query_heads // self.kv_heads

    @property
    def parameter_count(self) -> int:
        """Exact trainable count for tied embeddings, bias-free GQA + SwiGLU."""
        kv_width = self.kv_heads * self.head_dim
        attention_per_layer = (
            2 * self.hidden_size * self.hidden_size
            + 2 * self.hidden_size * kv_width
        )
        ffn_per_layer = 3 * self.hidden_size * self.intermediate_size
        norm_parameters = 2 * self.hidden_size * self.layers + self.hidden_size
        embedding_parameters = self.vocab_size * self.hidden_size
        return (
            embedding_parameters
            + self.layers * (attention_per_layer + ffn_per_layer)
            + norm_parameters
        )

    def to_carbon_config_dict(self) -> dict[str, Any]:
        if self.eos_token_id is None:
            raise ArchitectureConfigError(
                "eos_token_id is required before constructing a trainable model"
            )
        return {
            "vocab_size": self.vocab_size,
            "n_layers": self.layers,
            "d_model": self.hidden_size,
            "n_heads": self.query_heads,
            "n_kv_heads": self.kv_heads,
            "intermediate_size": self.intermediate_size,
            "max_seq_len": self.context_length,
            "dropout": float(self.dropout),
            "rope_theta": float(self.rope_theta),
            "eos_token_id": self.eos_token_id,
            "rms_norm_eps": float(self.rms_norm_eps),
        }

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> DecoderArchitectureConfig:
        if not isinstance(values, dict):
            raise ArchitectureConfigError("architecture configuration must be an object")
        allowed = set(cls.__dataclass_fields__)
        unknown = set(values) - allowed
        if unknown:
            raise ArchitectureConfigError(f"unsupported architecture fields: {sorted(unknown)}")
        try:
            return cls(**values)
        except TypeError as exc:
            raise ArchitectureConfigError(f"missing or invalid architecture fields: {exc}") from exc


@dataclass(frozen=True)
class ArchitectureBlueprint:
    model_id: str
    model_version: str
    status: str
    architecture: DecoderArchitectureConfig


def load_architecture_blueprint(path: str | Path) -> ArchitectureBlueprint:
    """Load a strict JSON blueprint; symlinks and duplicate keys are rejected."""
    source_input = Path(path).expanduser()
    if source_input.is_symlink():
        raise ArchitectureConfigError("architecture blueprint must not be a symlink")
    source = source_input.resolve()
    if not source.is_file():
        raise ArchitectureConfigError(f"architecture blueprint is not a file: {source}")
    try:
        payload = json.loads(
            source.read_text(encoding="utf-8"),
            parse_constant=_reject_constant,
            object_pairs_hook=_unique_object,
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise ArchitectureConfigError(f"cannot load architecture blueprint: {exc}") from exc
    if not isinstance(payload, dict) or set(payload) != _BLUEPRINT_KEYS:
        raise ArchitectureConfigError("blueprint has unsupported or missing top-level fields")
    if payload["schema_version"] != "1.0.0":
        raise ArchitectureConfigError("unsupported architecture blueprint schema_version")
    if not isinstance(payload["model_id"], str) or not _MODEL_ID.fullmatch(payload["model_id"]):
        raise ArchitectureConfigError("model_id is invalid")
    if (
        not isinstance(payload["model_version"], str)
        or not payload["model_version"].strip()
        or len(payload["model_version"]) > 64
    ):
        raise ArchitectureConfigError("model_version is invalid")
    if not isinstance(payload["status"], str) or payload["status"] not in _BLUEPRINT_STATUSES:
        raise ArchitectureConfigError("blueprint status is invalid")
    return ArchitectureBlueprint(
        model_id=payload["model_id"],
        model_version=payload["model_version"],
        status=payload["status"],
        architecture=DecoderArchitectureConfig.from_dict(payload["architecture"]),
    )
