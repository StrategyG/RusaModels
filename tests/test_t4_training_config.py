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


def test_t4_workflow_uses_existing_requirements_and_immutable_actions():
    workflow = yaml.safe_load(
        (ROOT / ".github" / "workflows" / "agent-sft-training.yml").read_text(encoding="utf-8")
    )
    steps = workflow["jobs"]["train-and-evaluate"]["steps"]
    install_step = next(item for item in steps if item.get("name") == "Install dedicated training dependencies")

    assert (ROOT / "requirements-training.txt").is_file()
    assert "requirements-training.txt" in install_step["run"]
    assert "requirements-train.txt" not in install_step["run"]
    for step in steps:
        action = step.get("uses")
        if action:
            reference = action.rsplit("@", maxsplit=1)[-1].split("#", maxsplit=1)[0].strip()
            assert len(reference) == 40
            assert all(character in "0123456789abcdef" for character in reference)
