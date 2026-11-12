"""Safe, explicitly bounded construction of the supported Carbon decoder."""
from __future__ import annotations

from .config import DecoderArchitectureConfig
from .validation import validate_model_shapes


DEFAULT_MAX_PARAMETERS = 120_000_000


def build_model(
    config: DecoderArchitectureConfig,
    *,
    max_parameters: int = DEFAULT_MAX_PARAMETERS,
):
    """Build a Carbon decoder only when its configured parameter budget is allowed."""
    if not isinstance(config, DecoderArchitectureConfig):
        raise TypeError("config must be a DecoderArchitectureConfig")
    if type(max_parameters) is not int or max_parameters < 1:
        raise ValueError("max_parameters must be a positive integer")
    if config.parameter_count > max_parameters:
        raise ValueError(
            f"model requires {config.parameter_count:,} parameters, exceeding "
            f"the construction limit of {max_parameters:,}"
        )
    if not config.tied_embeddings or config.attention_bias:
        raise ValueError("Carbon model factory supports tied embeddings and bias-free projections only")
    try:
        from rusamodels.carbon_cpu.model import CarbonConfig, CarbonTiny
    except ImportError as exc:
        raise RuntimeError("Install the carbon-cpu extra to instantiate a neural model") from exc
    carbon_config = CarbonConfig.from_dict(config.to_carbon_config_dict())
    model = CarbonTiny(carbon_config)
    validate_model_shapes(model, config)
    return model
