"""Carbon-CPU exports with lazy PyTorch model imports."""

from .constants import BOS_TOKEN_ID, BYTE_VOCAB_SIZE, EOS_TOKEN_ID
from .tokenizer import CarbonByteTokenizer

__all__ = [
    "BOS_TOKEN_ID",
    "BYTE_VOCAB_SIZE",
    "EOS_TOKEN_ID",
    "CarbonByteTokenizer",
    "CarbonConfig",
    "CarbonTiny",
    "count_parameters",
]


def __getattr__(name: str):
    if name in {"CarbonConfig", "CarbonTiny", "count_parameters"}:
        from . import model

        return getattr(model, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
