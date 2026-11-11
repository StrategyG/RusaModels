"""UTF-8 byte tokenizer with two reserved special token IDs."""
from __future__ import annotations

from collections.abc import Iterable

from .constants import BOS_TOKEN_ID, BYTE_VOCAB_SIZE, EOS_TOKEN_ID


class CarbonByteTokenizer:
    """A reversible UTF-8 byte tokenizer; this is not a learned subword tokenizer."""

    vocab_size = BYTE_VOCAB_SIZE
    bos_token_id = BOS_TOKEN_ID
    eos_token_id = EOS_TOKEN_ID

    def encode(
        self,
        text: str,
        *,
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> list[int]:
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        if type(add_bos) is not bool or type(add_eos) is not bool:
            raise TypeError("add_bos and add_eos must be booleans")
        tokens = list(text.encode("utf-8"))
        if add_bos:
            tokens.insert(0, self.bos_token_id)
        if add_eos:
            tokens.append(self.eos_token_id)
        return tokens

    def decode(
        self,
        token_ids: Iterable[int],
        *,
        skip_special_tokens: bool = True,
    ) -> str:
        if type(skip_special_tokens) is not bool:
            raise TypeError("skip_special_tokens must be a boolean")
        parts: list[str] = []
        buffer = bytearray()

        def flush() -> None:
            if buffer:
                parts.append(buffer.decode("utf-8", errors="replace"))
                buffer.clear()

        for token_id in token_ids:
            if type(token_id) is not int:
                raise TypeError("token IDs must be integers")
            if 0 <= token_id <= 255:
                buffer.append(token_id)
            elif token_id in {self.bos_token_id, self.eos_token_id}:
                if not skip_special_tokens:
                    flush()
                    parts.append(
                        "<|bos|>" if token_id == self.bos_token_id else "<|eos|>"
                    )
            else:
                raise ValueError(f"token ID is outside the byte vocabulary: {token_id}")
        flush()
        return "".join(parts)
