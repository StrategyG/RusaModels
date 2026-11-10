from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "training" / "agent-sft-t4.yaml"


def test_t4_profile_uses_supported_memory_and_precision_settings():
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))

    assert config["model"]["id"] == "Qwen/Qwen3-4B-Instruct-2507"
    assert config["dataset"]["max_seq_length"] == 2048
    assert config["qlora"]["quant_type"] == "nf4"
    assert config["qlora"]["compute_dtype"] == "float16"
    assert config["lora"]["r"] == 8
    assert config["training"]["per_device_train_batch_size"] == 1
    assert config["training"]["gradient_checkpointing"] is True
    assert config["training"]["dataloader_num_workers"] == 0
    assert config["hardware_target"]["gpu"] == "NVIDIA Tesla T4"
    assert config["hardware_target"]["minimum_vram_gib"] == 14
    assert config["training"]["output_dir"].endswith("-t4")
