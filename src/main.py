"""
Entrypoint for the Adversarial Guardrail Co-Evolution Experiment.

Supports both:
- Full training on the remote Fedora 41 Linux desktop with NVIDIA TITAN RTX 24GB (CUDA FP16 + SDPA)
- Local smoke-testing / development on Apple Silicon MacBook M1 Pro (MPS / CPU)
"""

from __future__ import annotations

import argparse
import os
import random

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch

from src.config import ModelConfig, TokenCeilingConfig, TrainingConfig
from src.trainer import AdversarialCoEvolutionOrchestrator


def set_reproducible_seed(seed: int) -> None:
    """Seeds Python, NumPy, and PyTorch (CPU & CUDA) RNGs for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_compute_device(requested: str = "auto") -> torch.device:
    """Selects CUDA (TITAN RTX), MPS (Mac M1 Pro), or CPU."""
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Zero-Sum Adversarial GRPO Co-Evolution (Prompt Adapter vs. Red Adapter)"
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="Qwen/Qwen2.5-1.5B-Instruct",
        help="Hugging Face model ID for the shared-weight base model (default: Qwen/Qwen2.5-1.5B-Instruct)",
    )
    parser.add_argument(
        "--evaluator-model",
        type=str,
        default="microsoft/deberta-v3-small",
        help="Hugging Face model ID for the DeBERTa-v3 sidecar classifier (default: microsoft/deberta-v3-small)",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=100,
        help="Total co-evolution training steps (default: 100)",
    )
    parser.add_argument(
        "--group-size",
        type=int,
        default=4,
        help="GRPO group size G for preprompt and attack batch generation (default: 4)",
    )
    parser.add_argument(
        "--lr-prompt",
        type=float,
        default=2.5e-5,
        help="Learning rate for prompt_adapter (recommended: 1e-5 to 5e-5)",
    )
    parser.add_argument(
        "--lr-red",
        type=float,
        default=2.5e-5,
        help="Learning rate for red_adapter (recommended: 1e-5 to 5e-5)",
    )
    parser.add_argument(
        "--utility-weight",
        type=float,
        default=0.6,
        help="Weight applied to Model Blue's benign utility score in the Prompt composite reward",
    )
    parser.add_argument(
        "--kl-coef",
        type=float,
        default=0.02,
        help="KL regularization coefficient against the frozen reference base model",
    )
    parser.add_argument(
        "--sample-interval",
        type=int,
        default=5,
        help="Step interval to refresh the live Model Red / Model Blue dialogue showcase",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=25,
        help="Step interval to save PEFT LoRA adapter checkpoints",
    )
    parser.add_argument(
        "--preprompt-max-tokens",
        type=int,
        default=96,
        help="Fixed ceiling on generated tokens for defensive preprompts (default: 96)",
    )
    parser.add_argument(
        "--attack-max-tokens",
        type=int,
        default=128,
        help="Fixed ceiling on generated tokens for adversarial attacks (default: 128)",
    )
    parser.add_argument(
        "--blue-max-tokens",
        type=int,
        default=160,
        help="Fixed ceiling on generated tokens for Model Blue single-turn responses (default: 160)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./experimental_results",
        help="Mounted directory where checkpoints, TensorBoard logs, and metrics are written",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "mps", "cpu"],
        help="Compute device (default: auto)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Global random seed for reproducibility",
    )
    parser.add_argument(
        "--no-live-ui",
        action="store_true",
        help="Disable continuous Rich Live cursor loop and only refresh screen on sample steps",
    )
    return parser.parse_args()


def main() -> None:
    # Avoid tokenizer parallelism warnings in multi-process/container environments
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    args = parse_args()
    set_reproducible_seed(args.seed)
    device = select_compute_device(args.device)

    tokens_cfg = TokenCeilingConfig(
        preprompt_max_new_tokens=args.preprompt_max_tokens,
        attack_max_new_tokens=args.attack_max_tokens,
        blue_max_new_tokens=args.blue_max_tokens,
    )
    model_cfg = ModelConfig(
        base_model_id=args.base_model,
        evaluator_model_id=args.evaluator_model,
        attn_implementation="sdpa",
        use_fp16=(device.type in ("cuda", "mps")),
    )
    training_cfg = TrainingConfig(
        num_steps=args.num_steps,
        group_size=args.group_size,
        lr_prompt=args.lr_prompt,
        lr_red=args.lr_red,
        utility_weight=args.utility_weight,
        kl_coef=args.kl_coef,
        sample_refresh_interval=args.sample_interval,
        checkpoint_interval=args.checkpoint_interval,
        output_dir=args.output_dir,
        seed=args.seed,
        use_live_ui=not args.no_live_ui,
        tokens=tokens_cfg,
        model=model_cfg,
    )

    orchestrator = AdversarialCoEvolutionOrchestrator(
        config=training_cfg,
        device=device,
    )
    orchestrator.run()


if __name__ == "__main__":
    main()
