"""
Adversarial Co-Evolution of Defensive Preprompts and Medical Prompt Injections.

Implements a zero-sum Group Relative Policy Optimization (GRPO) training pipeline
on a shared-weight base LLM (Qwen2.5-1.5B) with dual PEFT LoRA adapters
(`prompt_adapter` and `red_adapter`), simulated frozen target (`Model Blue`),
and a deterministic zero-shot NLI sidecar medical-intent evaluator.
"""

import os

# Prevent OpenMP duplicate library abort when mixing conda-forge and PyPI wheels on macOS
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

__version__ = "0.1.0"
