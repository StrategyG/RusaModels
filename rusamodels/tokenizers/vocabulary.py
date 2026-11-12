"""Stable special-token contract for the RusaAi research BPE tokenizer."""

SPECIAL_TOKENS = {
    "unk": "[UNK]",
    "bos": "[BOS]",
    "eos": "[EOS]",
    "pad": "[PAD]",
}


def resolve_special_token_ids(tokenizer) -> dict[str, int]:
    """Resolve special token IDs and require a stable, collision-free vocabulary."""
    token_ids: dict[str, int] = {}
    for name, token in SPECIAL_TOKENS.items():
        token_id = tokenizer.token_to_id(token)
        if type(token_id) is not int or token_id < 0:
            raise ValueError(f"tokenizer is missing required special token {token}")
        token_ids[name] = token_id
    if len(set(token_ids.values())) != len(token_ids):
        raise ValueError("special token IDs must be unique")
    expected_order = {"unk": 0, "bos": 1, "eos": 2, "pad": 3}
    if token_ids != expected_order:
        raise ValueError(
            f"special token IDs do not match the versioned order: {token_ids}"
        )
    return token_ids
