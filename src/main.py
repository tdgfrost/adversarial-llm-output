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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Zero-Sum Adversarial GRPO Co-Evolution (Prompt Adaptor vs. Red Adaptor)"
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
        default="MoritzLaurer/deberta-v3-base-zeroshot-v2.0",
        help=(
            "Hugging Face model ID for the zero-shot NLI sidecar classifier "
            "(default: MoritzLaurer/deberta-v3-base-zeroshot-v2.0)"
        ),
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=int(os.environ.get("NUM_STEPS", "100")),
        help="Target total co-evolution training steps (default: 100). If resuming and > completed steps, trains up to this total.",
    )
    parser.add_argument(
        "--extend-steps",
        "-e",
        type=int,
        default=int(os.environ.get("EXTEND_STEPS", "0")),
        help="Extend a previous run (completed or interrupted) by N additional training steps beyond the latest checkpoint.",
    )
    parser.add_argument(
        "--resume-from",
        type=str,
        default=os.environ.get("RESUME_FROM", None),
        help="Optional explicit path to a checkpoint directory (e.g., experimental_results/checkpoints/step_0050) to resume/extend from.",
    )
    parser.add_argument(
        "--fresh",
        "--no-resume",
        dest="fresh",
        action="store_true",
        default=os.environ.get("FRESH_RUN", "0") == "1",
        help="Ignore existing checkpoints in output-dir and start a fresh training run from Step 1.",
    )
    parser.add_argument(
        "--group-size",
        type=int,
        default=8,
        help="GRPO group size G for preprompt and attack batch generation (default: 8)",
    )
    parser.add_argument(
        "--lr-prompt",
        type=float,
        default=2.5e-5,
        help="Learning rate for prompt_adaptor (recommended: 1e-5 to 5e-5)",
    )
    parser.add_argument(
        "--lr-red",
        type=float,
        default=2.5e-5,
        help="Learning rate for red_adaptor (recommended: 1e-5 to 5e-5)",
    )
    parser.add_argument(
        "--utility-weight",
        type=float,
        default=0.2,
        help="Weight applied to Model Blue's benign utility score in the Prompt composite reward",
    )
    parser.add_argument(
        "--prompt-kl-coef",
        type=float,
        default=0.05,
        help="KL regularisation coefficient for prompt_adaptor against the frozen reference base model",
    )
    parser.add_argument(
        "--red-kl-coef",
        type=float,
        default=0.05,
        help="KL regularisation coefficient for red_adaptor against the frozen reference base model",
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
        help="Step interval to save numbered PEFT LoRA adaptor checkpoints (rolling 'latest' is also saved every sample-interval)",
    )
    parser.add_argument(
        "--preprompt-max-tokens",
        type=int,
        default=128,
        help="Fixed ceiling on generated tokens for defensive preprompts (default: 128)",
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
        default=256,
        help="Fixed ceiling on generated tokens for Model Blue single-turn responses (default: 256)",
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
        attn_implementation="sdpa",
        evaluator_attn_implementation="eager",
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
