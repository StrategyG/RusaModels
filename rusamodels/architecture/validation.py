"""Configuration and tensor-shape validation for the supported architecture v1."""
from __future__ import annotations

from typing import Any

from .config import DecoderArchitectureConfig


def validate_architecture_config(config: DecoderArchitectureConfig) -> dict[str, int | float | str]:
    """Return a deterministic, auditable summary of the validated architecture."""
    if not isinstance(config, DecoderArchitectureConfig):
        raise TypeError("config must be a DecoderArchitectureConfig")
    return {
        "architecture": "dense_decoder_transformer",
        "layers": config.layers,
        "hidden_size": config.hidden_size,
        "query_heads": config.query_heads,
        "kv_heads": config.kv_heads,
        "head_dim": config.head_dim,
        "kv_group_size": config.kv_group_size,
        "intermediate_size": config.intermediate_size,
        "vocab_size": config.vocab_size,
        "context_length": config.context_length,
        "parameter_count": config.parameter_count,
    }


def _shape(value: Any) -> tuple[int, ...]:
    shape = getattr(value, "shape", None)
    if shape is None:
        raise ValueError("model parameter is missing a tensor shape")
    return tuple(int(dimension) for dimension in shape)


def validate_model_shapes(model: Any, config: DecoderArchitectureConfig) -> dict[str, Any]:
    """Check the instantiated Carbon decoder against the source blueprint."""
    if not isinstance(config, DecoderArchitectureConfig):
        raise TypeError("config must be a DecoderArchitectureConfig")
    embeddings = getattr(model, "token_embeddings", None)
    if embeddings is None or _shape(embeddings.weight) != (config.vocab_size, config.hidden_size):
        raise ValueError("token embedding shape does not match architecture configuration")
    layers = getattr(model, "layers", None)
    if layers is None or len(layers) != config.layers:
        raise ValueError("decoder layer count does not match architecture configuration")
    expected = {
        "q_proj": (config.query_heads * config.head_dim, config.hidden_size),
        "k_proj": (config.kv_heads * config.head_dim, config.hidden_size),
        "v_proj": (config.kv_heads * config.head_dim, config.hidden_size),
        "o_proj": (config.hidden_size, config.hidden_size),
        "gate_proj": (config.intermediate_size, config.hidden_size),
        "up_proj": (config.intermediate_size, config.hidden_size),
        "down_proj": (config.hidden_size, config.intermediate_size),
    }
    for index, layer in enumerate(layers):
        attention = getattr(layer, "attention", None)
        mlp = getattr(layer, "mlp", None)
        if attention is None or mlp is None:
            raise ValueError(f"decoder layer {index} is missing attention or feed-forward modules")
        if (
            getattr(attention, "n_heads", None) != config.query_heads
            or getattr(attention, "n_kv_heads", None) != config.kv_heads
            or getattr(attention, "head_dim", None) != config.head_dim
        ):
            raise ValueError(f"decoder layer {index} attention head configuration is invalid")
        expected_rope_shape = (config.context_length, config.head_dim)
        if (
            _shape(attention.rope_cos) != expected_rope_shape
            or _shape(attention.rope_sin) != expected_rope_shape
        ):
            raise ValueError(f"decoder layer {index} RoPE cache shape is invalid")
        for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            if _shape(getattr(attention, name).weight) != expected[name]:
                raise ValueError(f"decoder layer {index} {name} shape is invalid")
        for name in ("gate_proj", "up_proj", "down_proj"):
            if _shape(getattr(mlp, name).weight) != expected[name]:
                raise ValueError(f"decoder layer {index} {name} shape is invalid")
        for norm_name in ("attn_norm", "ffn_norm"):
            norm = getattr(layer, norm_name, None)
            if (
                norm is None
                or _shape(norm.weight) != (config.hidden_size,)
                or float(norm.eps) != float(config.rms_norm_eps)
            ):
                raise ValueError(f"decoder layer {index} {norm_name} shape/epsilon is invalid")
    final_norm = getattr(model, "final_norm", None)
    if (
        final_norm is None
        or _shape(final_norm.weight) != (config.hidden_size,)
        or float(final_norm.eps) != float(config.rms_norm_eps)
    ):
        raise ValueError("final RMSNorm shape/epsilon is invalid")
    actual_parameters = sum(
        int(parameter.numel())
        for parameter in model.parameters()
        if getattr(parameter, "requires_grad", False)
    )
    if actual_parameters != config.parameter_count:
        raise ValueError(
            f"instantiated parameter count {actual_parameters} does not match "
            f"configured count {config.parameter_count}"
        )
    return {
        "valid": True,
        "layers": len(layers),
        "parameter_count": actual_parameters,
        "embedding_shape": list(_shape(embeddings.weight)),
    }
