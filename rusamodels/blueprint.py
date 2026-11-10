from dataclasses import dataclass, field


@dataclass
class MoESpec:
    enabled: bool = True
    num_experts: int = 64
    experts_per_token: int = 2


@dataclass
class ModelSpec:
    name: str = "rusa-foundation-moe-research-blueprint"
    model_type: str = "decoder-only-transformer"
    layers: int = 96
    hidden_size: int = 12288
    attention_heads: int = 96
    kv_heads: int = 8
    context_window: int = 1_000_000
    positional_encoding: str = "rope"
    normalization: str = "rmsnorm"
    activation: str = "swiglu"
    modalities: list[str] = field(default_factory=lambda: ["text", "image"])
    moe: MoESpec = field(default_factory=MoESpec)

    def validate(self) -> None:
        if self.layers <= 0:
            raise ValueError("layers must be positive")
        if self.hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        if self.context_window <= 0:
            raise ValueError("context_window must be positive")
        if self.moe.enabled and self.moe.num_experts <= 0:
            raise ValueError("MoE expert count must be positive when enabled")


@dataclass
class TrainingSpec:
    objective: str = "next_token_prediction"
    precision: str = "bf16"
    post_training: list[str] = field(
        default_factory=lambda: [
            "supervised_fine_tuning",
            "rlhf_rlaif",
            "reasoning_rl",
            "safety_alignment",
        ]
    )


@dataclass
class InferenceSpec:
    quantization: List[str] = field(default_factory=lambda: ["int8", "fp8"])
    speculative_decoding: bool = True
    tool_use: bool = True


@dataclass
class ProjectBlueprint:
    model: ModelSpec = field(default_factory=ModelSpec)
    training: TrainingSpec = field(default_factory=TrainingSpec)
    inference: InferenceSpec = field(default_factory=InferenceSpec)
    status: str = "PLANNED"

    def validate(self) -> None:
        self.model.validate()


if __name__ == "__main__":
    blueprint = ProjectBlueprint()
    blueprint.validate()
    print(blueprint)
