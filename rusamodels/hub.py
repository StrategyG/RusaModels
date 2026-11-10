"""Public Hugging Face Hub discovery; model licenses remain upstream-controlled."""
from huggingface_hub import HfApi


def search_models(query: str, limit: int = 20) -> list[dict]:
    models = HfApi().list_models(
        search=query,
        limit=limit,
        sort="downloads",
        direction=-1,
        cardData=True,
    )
    results = []
    for model in models:
        card = getattr(model, "card_data", None)
        if card is None:
            card = getattr(model, "cardData", None)
        if isinstance(card, dict):
            license_id = card.get("license")
        else:
            license_id = getattr(card, "license", None)
        results.append({
            "repo_id": getattr(model, "modelId", None) or getattr(model, "id", None),
            "downloads": getattr(model, "downloads", None),
            "likes": getattr(model, "likes", None),
            "pipeline_tag": getattr(model, "pipeline_tag", None),
            "license_id": license_id,
            "gated": getattr(model, "gated", False),
            "tags": (getattr(model, "tags", None) or [])[:50],
        })
    return results
