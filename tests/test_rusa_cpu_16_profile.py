from pathlib import Path
import json

import yaml

ROOT = Path(__file__).resolve().parents[1]


class UniqueKeyLoader(yaml.SafeLoader):
    pass


def _construct_unique_mapping(loader, node, deep=False):
    mapping = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise yaml.constructor.ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                f"duplicate key {key!r}",
                key_node.start_mark,
            )
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


def test_rusa_cpu_16_profile_matches_rusaai_architecture_v2():
    profile = yaml.safe_load(
        (ROOT / "configs" / "runtime" / "rusa-cpu-16.yaml").read_text(encoding="utf-8")
    )
    model = json.loads(
        (ROOT / "models" / "rusaai-edge" / "manifests" / "0.1.0-design.json").read_text(
            encoding="utf-8"
        )
    )

    assert profile["profile"]["id"] == "rusa-cpu-16"
    assert profile["profile"]["version"] == "2.0.0"
    assert profile["profile"]["source_architecture"] == "RusaAi Architecture v2.0"
    assert profile["profile"]["status"] == "architecture_target_not_yet_deployable"
    assert profile["platform"]["operating_system"] == "Windows 11 64-bit"
    assert profile["platform"]["physical_memory_gb"] == 16
    assert profile["platform"]["cpu"]["sku_detection"] == "required"

    assert profile["model"]["id"] == "rusaai-edge-1.54b"
    assert profile["model"]["identity_alignment"]["architecture_reference_model_id"] == "rusaai-edge-1.54b"
    assert profile["model"]["identity_alignment"]["training_engine_pre_audit_product_model_id"] == "rusaai-edge-derived"
    assert profile["model"]["identity_alignment"]["runtime_model_id_decision"] == "PENDING"
    assert profile["services"]["rusamodels_model_server"]["model_id"] is None
    assert profile["services"]["rusamodels_model_server"]["model_id_source"] == "approved_manifest_identity_alignment_runtime_model_id"
    assert profile["model"]["target_parameters"] == 1541277696
    assert profile["model"]["checkpoint"]["selection_requirement"] == "non_qwen_open_weight"
    assert profile["model"]["checkpoint"]["selection_status"] == "NOT_SELECTED"
    assert profile["model"]["tokenizer"]["status"] == "NOT_SELECTED"
    assert profile["model"]["artifact"]["status"] == "NOT_BUILT"
    assert profile["model"]["benchmark"]["status"] == "NOT_TESTED"

    assert model["model_id"] == "rusaai-edge-1.54b"
    assert model["identity_alignment"]["architecture_reference_model_id"] == "rusaai-edge-1.54b"
    assert model["identity_alignment"]["training_engine_pre_audit_product_model_id"] == "rusaai-edge-derived"
    assert model["identity_alignment"]["runtime_model_id"] is None
    assert model["identity_alignment"]["runtime_model_id_decision"] == "PENDING"
    assert model["status"] == "DESIGN_ONLY"
    assert model["checkpoint"]["selection_status"] == "NOT_SELECTED"
    assert model["checkpoint"]["non_qwen_lineage_status"] == "NOT_VERIFIED"
    assert model["checkpoint"]["architecture_relation"] == "NOT_AUDITED"
    assert model["artifact"]["status"] == "NOT_BUILT"
    assert model["architecture"]["target_parameters"] == 1541277696
    assert model["architecture"]["layers"] == 32
    assert model["architecture"]["hidden_size"] == 2048
    assert model["architecture"]["query_heads"] == 16
    assert model["architecture"]["key_value_heads"] == 4
    assert model["architecture"]["feed_forward"] == "swiglu"
    assert model["architecture"]["feed_forward_intermediate"] == 5632


def test_v2_runtime_parameters_and_control_plane_ownership():
    profile = yaml.safe_load(
        (ROOT / "configs" / "runtime" / "rusa-cpu-16.yaml").read_text(encoding="utf-8")
    )

    assert profile["runtime"]["context"]["initial_tokens"] == 4096
    assert profile["runtime"]["context"]["target_tokens"] == 8192
    assert profile["runtime"]["generation_threads_initial"] == 4
    assert profile["runtime"]["batch_threads_initial"] == 4
    assert profile["runtime"]["http_threads_initial"] == 2
    assert profile["runtime"]["batch_size_initial"] == 128
    assert profile["runtime"]["micro_batch_size_initial"] == 64
    assert profile["runtime"]["parallel_slots"] == 1
    assert profile["runtime"]["resident_model_limit"] == 1
    assert profile["runtime"]["gpu_layers"] == 0
    assert profile["orchestration"]["owner"] == "Rusaai"
    assert profile["orchestration"]["active_generation_limit"] == 1
    assert profile["orchestration"]["pending_inference_queue_limit"] == 3
    assert profile["orchestration"]["queue_owner"] == "Rusaai"
    assert profile["orchestration"]["deep_reasoning_candidates_target"] == 2
    assert profile["control_plane_reference"]["capability_profiles_share_one_model"] is True
    assert profile["control_plane_reference"]["reasoning_profiles"]["deep"]["candidates"] == 2
    assert profile["control_plane_reference"]["reasoning_profiles"]["max"]["status"] == "experimental_not_enabled_in_initial_profile"
    assert profile["resource_protection"]["minimum_free_memory_gb"] == 2.0
    assert profile["memory_estimates"]["kv_cache_q8_0_mib"]["context_4096"] == 128
    assert profile["memory_estimates"]["kv_cache_q8_0_mib"]["context_8192"] == 256


def test_native_windows_launcher_fails_closed_until_model_is_ready():
    start = (ROOT / "scripts" / "start_rusa_cpu_16.ps1").read_text(encoding="utf-8")
    verify = (ROOT / "scripts" / "verify_rusa_cpu_16.ps1").read_text(encoding="utf-8")
    env = (ROOT / ".env.rusa-cpu-16.example").read_text(encoding="utf-8")

    assert "RUSAMODELS_INFERENCE_MODEL_ID=REPLACE_AFTER_ADR_APPROVAL" in env
    assert "RUSAMODELS_CONTEXT_SIZE=4096" in env
    assert 'selection_status -ne "SELECTED"' in start
    assert 'compatibility_status -ne "VERIFIED"' in start
    assert 'non_qwen_lineage_status -ne "VERIFIED"' in start
    assert 'actual_configuration_sha256 -notmatch' in start
    assert 'architectureRelation -eq "IDENTICAL"' in start
    assert '$os.OSArchitecture -ne "64-bit"' in start
    assert 'tokenizer.status -ne "VALIDATED"' in start
    assert 'artifact.status -ne "BUILT"' in start
    assert 'status -ne "READY_FOR_DEVICE_VALIDATION"' in start
    assert 'runtime_model_id_decision -ne "APPROVED"' in start
    assert "approvedRuntimeModelId" in start
    assert "rusaai-edge-1.54b" not in start
    assert "tokenizer.revision" in start
    assert "artifact.filename" in start
    assert "Qwen/" in start
    assert '"--alias", $modelId' in start
    assert '"--ctx-size", "$contextSize"' in start
    assert '"--batch-size", "128"' in start
    assert '"--ubatch-size", "64"' in start
    assert '"--n-gpu-layers", "0"' in start
    assert '"--reasoning-budget"' not in start
    assert "function Get-RusaApprovedRuntimeModelId" in verify
    assert '"rusaai-edge-1.54b"' not in verify
    assert "model = $runtimeModelId" in verify
    assert "ModelId = $runtimeModelId" in verify
    assert "tokens_predicted_seconds_total" in verify
    assert "PromptSizeClassesTokens = @(128, 1024, 4096)" in verify
    assert "Get-LlamaTokenCount" in verify
    assert "function Get-RusaMedian" in verify
    assert "PromptBenchmarks = $promptReports" in verify
    assert "requests_processing" in verify
    assert "requests_deferred" in verify
    assert "minimumFreeMemoryGiB" in verify
    assert "IsProcessorFeaturePresent" in start
    assert "RUSA-CPU-16 services are ready" in start

    context_test = (ROOT / "scripts" / "verify_rusa_cpu_16_context.ps1").read_text(encoding="utf-8")
    assert "RUSAMODELS_CONTEXT_8K_TEST_MODE" in context_test
    assert "http://127.0.0.1:8081/tokenize" in context_test
    assert "ActualPromptTokens" in context_test
    assert "MinimumSampledFreeMemoryGiB" in context_test
    assert "while (-not $sendTask.IsCompleted)" in context_test
    assert "while (-not $bodyTask.IsCompleted)" in context_test
    assert "MemorySampleIntervalMilliseconds = 500" in context_test
    assert "totalTokens -gt 8192" in context_test
    assert "function Get-RusaApprovedRuntimeModelId" in context_test
    assert '"rusaai-edge-1.54b"' not in context_test
    assert "model = $runtimeModelId" in context_test


def test_qwen3_v1_profile_is_preserved_as_historical_reference():
    legacy = yaml.safe_load(
        (
            ROOT / "configs" / "runtime" / "archive"
            / "rusa-cpu-16-v1.0-qwen3.yaml"
        ).read_text(encoding="utf-8")
    )
    legacy_docs = (
        ROOT / "docs" / "archive" / "RUSA-CPU-16-WINDOWS-v1.0-Qwen3.md"
    ).read_text(encoding="utf-8")

    assert legacy["model"]["repository"] == "Qwen/Qwen3-4B-GGUF"
    assert "Qwen3-4B" in legacy_docs
    assert "v1.0" in legacy_docs


def test_edge_test_configuration_uses_an_explicit_runtime_model_alias():
    config_test = (ROOT / "tests" / "test_config.py").read_text(encoding="utf-8")
    assert '"inference_model_id": "rusaai-edge-test-runtime"' in config_test



def test_rusa_edge_8_profile_preserves_architecture_limits_and_device_targets():
    profile = yaml.safe_load(
        (ROOT / "configs" / "runtime" / "rusa-edge-8.yaml").read_text(encoding="utf-8")
    )

    assert profile["profile"]["id"] == "rusa-edge-8"
    assert profile["profile"]["source_architecture"] == "RusaAi Architecture v2.0"
    assert profile["profile"]["status"] == "architecture_target_not_yet_deployable"
    assert profile["platform"]["physical_memory_gb"] == 8
    assert profile["platform"]["physical_memory_only"] is True
    assert profile["platform"]["virtual_or_extended_memory_counts_as_physical"] is False
    assert profile["platform"]["reference_devices"]["windows"]["model"] == "Lenovo V14 G3 IAP"
    assert profile["platform"]["reference_devices"]["android"]["model"] == "Infinix GT 30 Pro"
    assert profile["model"]["target_parameter_range"]["maximum"] == 1500000000
    assert profile["model"]["architecture_reference"]["parameter_count"] == 1541277696
    assert profile["model"]["architecture_reference"]["fit_status"] == "not_validated_against_8gb_device"
    assert profile["model"]["identity_alignment"]["architecture_reference_model_id"] == "rusaai-edge-1.54b"
    assert profile["model"]["identity_alignment"]["training_engine_pre_audit_product_model_id"] == "rusaai-edge-derived"
    assert profile["model"]["identity_alignment"]["runtime_model_id_decision"] == "PENDING"
    assert profile["model"]["checkpoint"]["non_qwen_lineage_status"] == "NOT_VERIFIED"
    assert profile["model"]["checkpoint"]["architecture_relation"] == "NOT_AUDITED"
    assert profile["model"]["checkpoint"]["compatibility_status"] == "NOT_VERIFIED"
    assert profile["model"]["tokenizer"]["chat_template_status"] == "NOT_VALIDATED"
    assert profile["model"]["artifact"]["status"] == "NOT_BUILT"
    assert profile["model"]["training"]["status"] == "NOT_STARTED"
    assert profile["model"]["benchmark"]["status"] == "NOT_TESTED"
    assert profile["runtime"]["context_tokens"] == 4096
    assert profile["runtime"]["sessions"] == 1
    assert profile["runtime"]["active_generations"] == 1
    assert profile["runtime"]["pending_work"]["bounded"] is True
    assert profile["runtime"]["pending_work"]["capacity"] is None
    assert profile["runtime"]["pending_work"]["owner"] == "Rusaai"
    assert profile["runtime"]["gpu_layers"] == 0
    assert profile["memory_estimates"]["model_weights_q4_k_m_gb"]["minimum"] == 0.9
    assert profile["memory_estimates"]["model_weights_q4_k_m_gb"]["maximum"] == 1.1
    assert profile["memory_estimates"]["kv_cache_q8_0_mib"]["context_4096"] == 128
    assert profile["memory_policy"]["low_memory_protection"]["free_memory_below_gb"] == 2



def test_foundation_checkpoint_adr_keeps_selection_and_runtime_identity_pending():
    adr = (
        ROOT / "documentation" / "adr" / "ADR-TRAIN-001-foundation-checkpoint.md"
    ).read_text(encoding="utf-8")

    assert "Status:** DRAFT" in adr
    assert "rusaai-edge-1.54b" in adr
    assert "rusaai-edge-derived" in adr
    assert "BhinekaIntiLabs/bhineka-gpt" in adr
    assert "PENDING_AUDIT / PILOT_CANDIDATE_ONLY" in adr
    assert "No checkpoint is approved by this draft ADR." in adr



def test_rusa_cpu_16_soak_checks_process_identity_and_resource_floor():
    soak_test = (ROOT / "scripts" / "verify_rusa_cpu_16_soak.ps1").read_text(encoding="utf-8")
    assert "StableSessionMinutes = $DurationMinutes" in soak_test
    assert "RequestIntervalSeconds" in soak_test
    assert "MinimumSampledFreeMemoryGiB" in soak_test
    assert "RestartDetected = $false" in soak_test
    assert "SwapTelemetry" in soak_test
    assert "state.started_utc" in soak_test
    assert "process.StartTime" in soak_test
    assert "function Get-RusaApprovedRuntimeModelId" in soak_test
    assert '"rusaai-edge-1.54b"' not in soak_test
    assert "model = $runtimeModelId" in soak_test



def test_active_runtime_yaml_profiles_reject_duplicate_keys():
    paths = [
        ROOT / "configs" / "runtime" / "rusa-cpu-16.yaml",
        ROOT / "configs" / "runtime" / "rusa-edge-8.yaml",
        ROOT / "configs" / "runtime" / "archive" / "rusa-cpu-16-v1.0-qwen3.yaml",
    ]
    for path in paths:
        yaml.load(path.read_text(encoding="utf-8"), Loader=UniqueKeyLoader)
