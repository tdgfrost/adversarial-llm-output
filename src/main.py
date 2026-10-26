"""
Entrypoint for the Adversarial Guardrail Co-Evolution Experiment.

Supports both:
- Full training on the remote Fedora 41 Linux desktop with NVIDIA TITAN RTX 24GB (CUDA FP16 + SDPA)
- Local smoke-testing / development on Apple Silicon MacBook M1 Pro (MPS / CPU)
- Seamless checkpoint auto-resumption and run extension (`--extend-steps`, `--num-steps`, `--resume-from`, `--fresh`).
"""

from __future__ import annotations

import argparse
import os
import random
import sys

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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    default_cfg = TrainingConfig()
    default_tokens = default_cfg.tokens
    default_model = default_cfg.model

    parser = argparse.ArgumentParser(
        description="Zero-Sum Adversarial GRPO Co-Evolution (Prompt Adaptor vs. Red Adaptor)"
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default=default_model.base_model_id,
        help=f"Hugging Face model ID for the shared-weight base model (default: {default_model.base_model_id})",
    )
    parser.add_argument(
        "--evaluator-model",
        type=str,
        default=default_model.evaluator_model_id,
        help=(
            "Hugging Face model ID for the zero-shot NLI sidecar classifier "
            f"(default: {default_model.evaluator_model_id})"
        ),
    )
    parser.add_argument(
        "--similarity-model",
        type=str,
        default=default_model.similarity_model_id,
        help=(
            "Hugging Face model ID for the lightweight sentence-embedding cosine similarity sidecar "
            f"(default: {default_model.similarity_model_id})"
        ),
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=int(os.environ.get("NUM_STEPS", str(default_cfg.num_steps))),
        help=(
            f"Target total co-evolution training steps (default: {default_cfg.num_steps}). "
            "If resuming and > completed steps, trains up to this total."
        ),
    )
    parser.add_argument(
        "--extend-steps",
        "-e",
        type=int,
        default=int(os.environ.get("EXTEND_STEPS", str(default_cfg.extend_steps))),
        help="Extend a previous run (completed or interrupted) by N additional training steps beyond the latest checkpoint.",
    )
    parser.add_argument(
        "--resume-from",
        type=str,
        default=os.environ.get("RESUME_FROM", default_cfg.resume_from),
        help="Optional explicit path to a checkpoint directory (e.g., experimental_results/checkpoints/step_0050) to resume/extend from.",
    )
    parser.add_argument(
        "--fresh",
        "--no-resume",
        dest="fresh",
        action="store_true",
        default=os.environ.get("FRESH_RUN", "1" if default_cfg.fresh else "0") == "1",
        help="Ignore existing checkpoints in output-dir and start a fresh training run from Step 1.",
    )
    parser.add_argument(
        "--group-size",
        type=int,
        default=default_cfg.group_size,
        help=f"GRPO group size G for preprompt and attack batch generation (default: {default_cfg.group_size})",
    )
    parser.add_argument(
        "--lr-prompt",
        type=float,
        default=default_cfg.lr_prompt,
        help=f"Learning rate for prompt_adaptor (recommended: 1e-5 to 5e-5, default: {default_cfg.lr_prompt})",
    )
    parser.add_argument(
        "--lr-red",
        type=float,
        default=default_cfg.lr_red,
        help=f"Learning rate for red_adaptor (recommended: 1e-5 to 5e-5, default: {default_cfg.lr_red})",
    )
    parser.add_argument(
        "--red-relevance-weight",
        type=float,
        default=default_cfg.red_relevance_weight,
        help=f"Weight applied to the seed-scenario relevance logit in the Red composite reward (default: {default_cfg.red_relevance_weight})",
    )
    parser.add_argument(
        "--utility-weight",
        type=float,
        default=default_cfg.utility_weight,
        help=f"Weight applied to Model Blue's benign utility score in the Prompt composite reward (default: {default_cfg.utility_weight})",
    )
    parser.add_argument(
        "--prompt-kl-coef",
        type=float,
        default=default_cfg.prompt_kl_coef,
        help=f"KL regularisation coefficient for prompt_adaptor against the frozen reference base model (default: {default_cfg.prompt_kl_coef})",
    )
    parser.add_argument(
        "--red-kl-coef",
        type=float,
        default=default_cfg.red_kl_coef,
        help=f"KL regularisation coefficient for red_adaptor against the frozen reference base model (default: {default_cfg.red_kl_coef})",
    )
    parser.add_argument(
        "--sample-interval",
        type=int,
        default=default_cfg.sample_refresh_interval,
        help=f"Step interval to refresh the live Model Red / Model Blue dialogue showcase (default: {default_cfg.sample_refresh_interval})",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=default_cfg.checkpoint_interval,
        help=(
            "Step interval to save numbered PEFT LoRA adaptor checkpoints "
            f"(rolling 'latest' is also saved every sample-interval, default: {default_cfg.checkpoint_interval})"
        ),
    )
    parser.add_argument(
        "--preprompt-max-tokens",
        type=int,
        default=default_tokens.preprompt_max_new_tokens,
        help=f"Fixed ceiling on generated tokens for defensive preprompts (default: {default_tokens.preprompt_max_new_tokens})",
    )
    parser.add_argument(
        "--attack-max-tokens",
        type=int,
        default=default_tokens.attack_max_new_tokens,
        help=f"Fixed ceiling on generated tokens for adversarial attacks (default: {default_tokens.attack_max_new_tokens})",
    )
    parser.add_argument(
        "--blue-max-tokens",
        type=int,
        default=default_tokens.blue_max_new_tokens,
        help=f"Fixed ceiling on generated tokens for Model Blue single-turn responses (default: {default_tokens.blue_max_new_tokens})",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=default_cfg.output_dir,
        help=f"Mounted directory where checkpoints, TensorBoard logs, and metrics are written (default: {default_cfg.output_dir})",
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
        default=default_cfg.seed,
        help=f"Global random seed for reproducibility (default: {default_cfg.seed})",
    )
    parser.add_argument(
        "--no-live-ui",
        action="store_true",
        default=not default_cfg.use_live_ui,
        help="Disable continuous Rich Live cursor loop and only refresh screen on sample steps",
    )
    return parser.parse_args(argv)


def main() -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    args = parse_args()
    num_steps_explicit = any(
        arg == "--num-steps" or arg.startswith("--num-steps=") for arg in sys.argv[1:]
    ) or ("NUM_STEPS" in os.environ)

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
        similarity_model_id=args.similarity_model,
        use_fp16=(device.type in ("cuda", "mps")),
    )
    training_cfg = TrainingConfig(
        num_steps=args.num_steps,
        group_size=args.group_size,
        resume=not args.fresh,
        fresh=args.fresh,
        extend_steps=args.extend_steps,
        resume_from=args.resume_from,
        num_steps_explicit=num_steps_explicit,
        lr_prompt=args.lr_prompt,
        lr_red=args.lr_red,
        red_relevance_weight=args.red_relevance_weight,
        utility_weight=args.utility_weight,
        prompt_kl_coef=args.prompt_kl_coef,
        red_kl_coef=args.red_kl_coef,
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
