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


def validate_chat_template(
    tokenizer: Any,
    messages: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    """Validate a foundation tokenizer's chat template and generation-prompt path."""
    template = getattr(tokenizer, "chat_template", None)
    if not template:
        raise ValueError("foundation tokenizer must define a non-empty chat_template")
    sample_messages = (
        [
            {"role": "system", "content": "You are RusaAi, a local-first technical assistant."},
            {"role": "user", "content": "Jelaskan cara memvalidasi JSON API response."},
        ]
        if messages is None
        else messages
    )
    if (
        not isinstance(sample_messages, list)
        or not sample_messages
        or any(
            not isinstance(message, dict)
            or set(message) != {"role", "content"}
            or not isinstance(message["role"], str)
            or message["role"] not in {"system", "user", "assistant"}
            or not isinstance(message["content"], str)
            or not message["content"].strip()
            for message in sample_messages
        )
        or not any(message["role"] == "user" for message in sample_messages)
    ):
        raise ValueError("chat-template validation messages must contain valid system/user/assistant turns")
    try:
        vocabulary_size = len(tokenizer)
        eos_token_id = tokenizer.eos_token_id
        token_ids = tokenizer.apply_chat_template(
            sample_messages,
            tokenize=True,
            add_generation_prompt=True,
        )
        if hasattr(token_ids, "tolist"):
            token_ids = token_ids.tolist()
        if token_ids and isinstance(token_ids[0], list):
            if len(token_ids) != 1:
                raise ValueError("chat template returned more than one encoded conversation")
            token_ids = token_ids[0]
        decoded = tokenizer.decode(token_ids, skip_special_tokens=False)
    except Exception as exc:
        raise ValueError(f"foundation tokenizer chat-template execution failed: {exc}") from exc
    if type(vocabulary_size) is not int or vocabulary_size < 1:
        raise ValueError("foundation tokenizer vocabulary size is invalid")
    if (
        not isinstance(token_ids, (list, tuple))
        or not token_ids
        or any(type(token_id) is not int or not 0 <= token_id < vocabulary_size for token_id in token_ids)
    ):
        raise ValueError("chat template emitted empty or out-of-vocabulary token IDs")
    if type(eos_token_id) is not int or not 0 <= eos_token_id < vocabulary_size:
        raise ValueError("foundation tokenizer must define a valid eos_token_id")
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is not None and (
        type(pad_token_id) is not int or not 0 <= pad_token_id < vocabulary_size
    ):
        raise ValueError("foundation tokenizer pad_token_id is invalid")
    if not isinstance(decoded, str) or not decoded.strip():
        raise ValueError("chat template decode returned empty text")
    user_content = next(
        message["content"] for message in sample_messages if message["role"] == "user"
    )
    if user_content not in decoded:
        raise ValueError("chat template decode did not preserve the user message content")
    return {
        "chat_template_present": True,
        "vocab_size": vocabulary_size,
        "eos_token_id": eos_token_id,
        "pad_token_id": pad_token_id,
        "prompt_tokens": len(token_ids),
        "decoded_characters": len(decoded),
        "generation_prompt_passed": True,
    }
