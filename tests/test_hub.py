from types import SimpleNamespace

from rusamodels import hub


def test_hub_search_returns_upstream_license_and_model_metadata(monkeypatch):
    class FakeApi:
        def list_models(self, **kwargs):
            assert kwargs["search"] == "Qwen"
            assert kwargs["limit"] == 5
            assert kwargs["sort"] == "downloads"
            return [
                SimpleNamespace(
                    modelId="Qwen/demo",
                    downloads=1234,
                    likes=42,
                    pipeline_tag="text-generation",
                    card_data=SimpleNamespace(license="apache-2.0"),
                    gated=False,
                    tags=["transformers", "text-generation"],
                )
            ]

    monkeypatch.setattr(hub, "HfApi", FakeApi)
    result = hub.search_models("Qwen", 5)
    assert result[0]["repo_id"] == "Qwen/demo"
    assert result[0]["license_id"] == "apache-2.0"
    assert result[0]["downloads"] == 1234
