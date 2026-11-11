"""Small decoder-only Transformer used by the Carbon-CPU from-scratch trainer."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


EOS_TOKEN_ID = 256
BOS_TOKEN_ID = 257
BYTE_VOCAB_SIZE = 258


@dataclass
class CarbonConfig:
    vocab_size: int = BYTE_VOCAB_SIZE
    n_layers: int = 4
    d_model: int = 256
    n_heads: int = 4
    n_kv_heads: int = 2
    intermediate_size: int = 768
    max_seq_len: int = 256
    dropout: float = 0.0
    rope_theta: float = 10000.0

    def __post_init__(self) -> None:
        integer_fields = {
            "vocab_size": self.vocab_size,
            "n_layers": self.n_layers,
            "d_model": self.d_model,
            "n_heads": self.n_heads,
            "n_kv_heads": self.n_kv_heads,
            "intermediate_size": self.intermediate_size,
            "max_seq_len": self.max_seq_len,
        }
        for name, value in integer_fields.items():
            if type(value) is not int:
                raise ValueError(f"{name} must be an integer")
        if self.vocab_size < BYTE_VOCAB_SIZE:
            raise ValueError("vocab_size must include 256 byte IDs and two special IDs")
        if self.n_layers < 1:
            raise ValueError("n_layers must be positive")
        if self.d_model < 1 or self.intermediate_size < 1:
            raise ValueError("d_model and intermediate_size must be positive")
        if self.n_heads < 1 or self.n_kv_heads < 1:
            raise ValueError("n_heads and n_kv_heads must be positive")
        if self.max_seq_len < 2:
            raise ValueError("max_seq_len must be at least 2")
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError("n_heads must be divisible by n_kv_heads")
        if (self.d_model // self.n_heads) % 2 != 0:
            raise ValueError("head dimension must be even for RoPE")
        if isinstance(self.dropout, bool) or not isinstance(self.dropout, (int, float)):
            raise ValueError("dropout must be a number")
        if not math.isfinite(float(self.dropout)) or not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be finite and in the range [0, 1)")
        if isinstance(self.rope_theta, bool) or not isinstance(self.rope_theta, (int, float)):
            raise ValueError("rope_theta must be a number")
        if not math.isfinite(float(self.rope_theta)) or self.rope_theta <= 0:
            raise ValueError("rope_theta must be finite and positive")

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> CarbonConfig:
        if not isinstance(values, dict):
            raise ValueError("model configuration must be an object")
        allowed = set(cls.__dataclass_fields__)
        unknown = set(values) - allowed
        if unknown:
            raise ValueError(f"unknown CarbonConfig fields: {sorted(unknown)}")
        return cls(**values)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class RMSNorm(nn.Module):
    def __init__(self, width: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x * scale.to(dtype=x.dtype) * self.weight


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    first, second = x.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


class CausalSelfAttention(nn.Module):
    def __init__(self, config: CarbonConfig) -> None:
        super().__init__()
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.d_model // config.n_heads
        self.groups = self.n_heads // self.n_kv_heads
        self.dropout = float(config.dropout)

        self.q_proj = nn.Linear(config.d_model, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.d_model, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.d_model, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(config.d_model, config.d_model, bias=False)

        inv_freq = 1.0 / (
            config.rope_theta
            ** (torch.arange(0, self.head_dim, 2, dtype=torch.float32) / self.head_dim)
        )
        positions = torch.arange(config.max_seq_len, dtype=torch.float32)
        frequencies = torch.outer(positions, inv_freq)
        angles = torch.cat((frequencies, frequencies), dim=-1)
        self.register_buffer("rope_cos", angles.cos(), persistent=False)
        self.register_buffer("rope_sin", angles.sin(), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        q = self.q_proj(x).view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)

        cos = self.rope_cos[:seq_len].view(1, 1, seq_len, self.head_dim).to(dtype=q.dtype)
        sin = self.rope_sin[:seq_len].view(1, 1, seq_len, self.head_dim).to(dtype=q.dtype)
        q = q * cos + _rotate_half(q) * sin
        k = k * cos + _rotate_half(k) * sin

        if self.groups > 1:
            k = k.repeat_interleave(self.groups, dim=1)
            v = v.repeat_interleave(self.groups, dim=1)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        causal_mask = torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device).triu(1)
        scores = scores.masked_fill(causal_mask, torch.finfo(scores.dtype).min)
        weights = F.softmax(scores.float(), dim=-1).to(dtype=q.dtype)
        weights = F.dropout(weights, p=self.dropout, training=self.training)
        output = torch.matmul(weights, v)
        output = output.transpose(1, 2).contiguous().view(batch, seq_len, -1)
        return self.o_proj(output)


class SwiGLU(nn.Module):
    def __init__(self, config: CarbonConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(config.d_model, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.d_model, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class TransformerBlock(nn.Module):
    def __init__(self, config: CarbonConfig) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(config.d_model)
        self.attention = CausalSelfAttention(config)
        self.ffn_norm = RMSNorm(config.d_model)
        self.mlp = SwiGLU(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.attn_norm(x))
        return x + self.mlp(self.ffn_norm(x))


class CarbonTiny(nn.Module):
    """Randomly initialized byte-level causal LM; no pretrained weights are loaded."""

    def __init__(self, config: CarbonConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embeddings = nn.Embedding(config.vocab_size, config.d_model)
        self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        self.final_norm = RMSNorm(config.d_model)
        self.dropout = nn.Dropout(float(config.dropout))
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if input_ids.ndim != 2 or input_ids.size(1) < 1:
            raise ValueError("input_ids must have shape [batch, non-empty sequence]")
        if input_ids.dtype != torch.long:
            raise ValueError("input_ids must use torch.long token IDs")
        if input_ids.size(1) > self.config.max_seq_len:
            raise ValueError("input sequence exceeds configured max_seq_len")
        if torch.any(input_ids < 0) or torch.any(input_ids >= self.config.vocab_size):
            raise ValueError("input_ids contains token IDs outside the configured vocabulary")
        if targets is not None:
            if targets.shape != input_ids.shape:
                raise ValueError("targets must have the same shape as input_ids")
            if targets.dtype != torch.long:
                raise ValueError("targets must use torch.long token IDs")
            if torch.any(targets < 0) or torch.any(targets >= self.config.vocab_size):
                raise ValueError("targets contains token IDs outside the configured vocabulary")

        x = self.dropout(self.token_embeddings(input_ids))
        for layer in self.layers:
            x = layer(x)
        x = self.final_norm(x)
        logits = F.linear(x, self.token_embeddings.weight)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), targets.reshape(-1)
            )
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 0.8,
        top_k: int = 40,
        eos_token_id: int = EOS_TOKEN_ID,
    ) -> torch.Tensor:
        if type(max_new_tokens) is not int or max_new_tokens < 0:
            raise ValueError("max_new_tokens must be a non-negative integer")
        if type(top_k) is not int or top_k < 0:
            raise ValueError("top_k must be a non-negative integer")
        if not math.isfinite(float(temperature)) or temperature <= 0:
            raise ValueError("temperature must be finite and greater than zero")
        if type(eos_token_id) is not int or not 0 <= eos_token_id < self.config.vocab_size:
            raise ValueError("eos_token_id is outside the configured vocabulary")
        if input_ids.ndim != 2 or input_ids.size(1) < 1:
            raise ValueError("input_ids must have shape [batch, non-empty sequence]")

        was_training = self.training
        self.eval()
        output = input_ids
        try:
            for _ in range(max_new_tokens):
                context = output[:, -self.config.max_seq_len :]
                logits, _ = self(context)
                next_logits = logits[:, -1, :] / temperature
                if top_k > 0:
                    cutoff = torch.topk(
                        next_logits, min(top_k, next_logits.size(-1))
                    ).values[:, -1]
                    next_logits = next_logits.masked_fill(
                        next_logits < cutoff[:, None], float("-inf")
                    )
                probabilities = F.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probabilities, num_samples=1)
                output = torch.cat((output, next_token), dim=1)
                if torch.all(next_token == eos_token_id):
                    break
            return output
        finally:
            self.train(was_training)


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
