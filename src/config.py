"""
Configuration dataclasses for the Adversarial Co-Evolution Experiment.

Defines hardware-aware settings for the NVIDIA TITAN RTX (24GB VRAM, Turing SM 7.5),
fixed token ceilings for every agent role in the single-turn interaction,
LoRA adapter hyperparameters, and GRPO optimization settings.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Tuple


@dataclass
class TokenCeilingConfig:
    """
    Fixed token ceilings for each role in the single-round interaction pipeline.

    Rationale:
    - `seed_max_tokens` (64): Medical seeds and benign queries are concise 1-2 sentence prompts.
    - `preprompt_max_new_tokens` (96): Sufficient for `prompt_adapter` to express clear,
      structured defensive boundaries without flooding Model Blue's context window.
    - `attack_max_new_tokens` (128): Allows `red_adapter` to construct sophisticated
      single-turn adversarial framings (e.g., roleplay, clinical vignette, urgency override)
      anchored to the medical seed while bounding GRPO backpropagation memory.
    - `blue_max_input_tokens` (384): Safely fits `preprompt (96) + attack (128) + chat template`.
    - `blue_max_new_tokens` (160): Enough for Model Blue to either issue a refusal/disclaimer,
      leak actionable medical advice (dosages/diagnoses), or provide a complete, fluent
      helpful paragraph on benign queries.
    - `evaluator_max_length` (256): Strictly exceeds `blue_max_new_tokens` (160) so the
      DeBERTa-v3 sidecar classifier inspects 100% of Model Blue's response without truncation.
    """

    seed_max_tokens: int = 64
    preprompt_min_new_tokens: int = 16
    preprompt_max_new_tokens: int = 96
    attack_min_new_tokens: int = 16
    attack_max_new_tokens: int = 128
    blue_max_input_tokens: int = 384
    blue_min_new_tokens: int = 16
    blue_max_new_tokens: int = 160
    evaluator_max_length: int = 256


@dataclass
class ModelConfig:
    """
    Shared-weight LLM + Dual PEFT LoRA + Sidecar DeBERTa-v3 Evaluator settings.

    Target Hardware: NVIDIA TITAN RTX 24GB (Turing architecture, Compute Capability 7.5).
    - Precision: FP16 (`torch.float16`) is required on Turing, which lacks native BF16.
    - Attention: Native PyTorch SDPA (`attn_implementation="sdpa"`) is enforced rather
      than Flash Attention 2 (which requires Ampere SM 8.0+).
    """

    base_model_id: str = "Qwen/Qwen2.5-1.5B-Instruct"
    evaluator_model_id: str = "microsoft/deberta-v3-small"
    prompt_adapter_name: str = "prompt_adapter"
    red_adapter_name: str = "red_adapter"

    # LoRA configuration for both adapters on the frozen base model
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: Tuple[str, ...] = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    )

    # Hardware-specific execution flags
    attn_implementation: str = "sdpa"
    use_fp16: bool = True

    # Sidecar evaluator fine-tuning / calibration settings
    evaluator_finetune_epochs: int = 4
    evaluator_finetune_lr: float = 3e-5
    evaluator_batch_size: int = 16


@dataclass
class TrainingConfig:
    """
    Hyperparameters for the interleaved dual-agent GRPO co-evolution loop.
    """

    num_steps: int = 100
    group_size: int = 4  # Number of preprompts (G) and attacks (G) generated per step

    # Optimizer settings (1e-5 to 5e-5 as specified for stable FP16 LoRA updates)
    lr_prompt: float = 2.5e-5
    lr_red: float = 2.5e-5
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    adam_eps: float = 1e-6

    # GRPO policy optimization settings
    kl_coef: float = 0.02
    advantage_eps: float = 1e-6
    clip_eps: float = 0.2

    # Composite reward weights for the Prompt model
    # R_prompt = defense_weight * (1.0 - P_medical_advice) + utility_weight * U_benign
    defense_weight: float = 1.0
    utility_weight: float = 0.6
    target_benign_words: int = 45

    # Sampling temperatures
    # Higher temperature for Prompt & Red enables diverse group exploration for GRPO variance
    agent_temperature: float = 0.85
    agent_top_p: float = 0.92
    # Lower temperature for frozen Model Blue simulates a standard deterministic assistant
    blue_temperature: float = 0.30
    blue_top_p: float = 0.90

    # Validation, UI refresh, and persistence intervals
    sample_refresh_interval: int = 5
    checkpoint_interval: int = 25
    moving_avg_window: int = 10
    output_dir: str = "./experimental_results"
    seed: int = 42
    use_live_ui: bool = True

    tokens: TokenCeilingConfig = field(default_factory=TokenCeilingConfig)
    model: ModelConfig = field(default_factory=ModelConfig)

    @property
    def output_path(self) -> Path:
        return Path(self.output_dir).resolve()

    @property
    def tensorboard_dir(self) -> Path:
        return self.output_path / "tensorboard"

    @property
    def checkpoints_dir(self) -> Path:
        return self.output_path / "checkpoints"

    @property
    def evaluator_cache_dir(self) -> Path:
        return self.output_path / "deberta_medical_evaluator"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
