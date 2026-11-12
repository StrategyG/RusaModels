"""Tokenizer round-trip and domain-probe diagnostics."""
from __future__ import annotations

from typing import Any


DEFAULT_SMOKE_SAMPLES = {
    "indonesian": [
        "RusaAi mengembangkan model bahasa Indonesia dengan tata kelola sumber yang dapat diverifikasi.",
        "Pengujian ulang memastikan ketidaktergantungan data evaluasi dan keterlacakan lisensi.",
    ],
    "english": [
        "Tokenizer compatibility must be verified before training or model export.",
        "A reproducible release records the exact corpus and tokenizer checksums.",
    ],
    "code": [
        "def validate_payload(payload: dict) -> bool:\n    return payload.get('status') == 'ready'",
        "for item in records:\n    assert item['source_id'] and item['content_sha256']",
    ],
    "json": [
        '{"tool":"search_knowledge","arguments":{"query":"hak akses","limit":5}}',
        '{"schema_version":"1.0.0","blocked":false,"actions":[]}',
    ],
}


def audit_tokenizer(
    tokenizer: Any,
    special_token_ids: dict[str, int],
    samples: dict[str, list[str]] | None = None,
) -> dict[str, Any]:
    """Check exact text round-trips and report tokenization diagnostics by domain."""
    if not isinstance(special_token_ids, dict) or set(special_token_ids) != {
        "unk", "bos", "eos", "pad"
    }:
        raise ValueError("special_token_ids must contain unk, bos, eos, and pad")
    ids = list(special_token_ids.values())
    if any(type(value) is not int or value < 0 for value in ids) or len(set(ids)) != 4:
        raise ValueError("special token IDs must be four unique non-negative integers")
    probe_sets = DEFAULT_SMOKE_SAMPLES if samples is None else samples
    if not isinstance(probe_sets, dict) or not probe_sets:
        raise ValueError("tokenizer audit requires at least one sample category")
    vocab_size = int(tokenizer.get_vocab_size(with_added_tokens=True))
    unknown_id = special_token_ids["unk"]
    report: dict[str, Any] = {}
    for category, texts in probe_sets.items():
        if not isinstance(category, str) or not category.strip():
            raise ValueError("tokenizer audit category names must be non-empty strings")
        if not isinstance(texts, list) or not texts:
            raise ValueError(f"tokenizer audit category {category} requires samples")
        characters = 0
        token_count = 0
        unknown_count = 0
        for text in texts:
            if not isinstance(text, str) or not text:
                raise ValueError(f"tokenizer audit sample in {category} must be non-empty text")
            encoded = tokenizer.encode(text, add_special_tokens=False)
            token_ids = list(encoded.ids)
            if any(type(token_id) is not int or not 0 <= token_id < vocab_size for token_id in token_ids):
                raise ValueError(f"tokenizer emitted an out-of-vocabulary ID in {category}")
            decoded = tokenizer.decode(token_ids, skip_special_tokens=True)
            if decoded != text:
                raise ValueError(f"tokenizer failed exact UTF-8 round-trip in {category}")
            characters += len(text)
            token_count += len(token_ids)
            unknown_count += sum(token_id == unknown_id for token_id in token_ids)
        if unknown_count:
            raise ValueError(f"tokenizer emitted unknown tokens in {category}")
        report[category] = {
            "samples": len(texts),
            "characters": characters,
            "tokens": token_count,
            "unknown_tokens": unknown_count,
            "tokens_per_character": round(token_count / max(characters, 1), 6),
        }
    return {"round_trip_passed": True, "vocab_size": vocab_size, "categories": report}
