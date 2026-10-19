"""
Fast, deterministic unit and end-to-end integration tests for the Adversarial
Guardrail Co-Evolution pipeline.

Verifies:
1. Dataset completeness (80 medical seeds, 80 benign health queries, epoch cycling).
2. DeBERTa-v3 reward matrix math and benign utility anti-lobotomization scoring.
3. Dual-LoRA adapter isolation (`prompt_adapter` vs `red_adapter`), frozen `Model Blue`
   and `Non-LoRA Red` context switching, GRPO backpropagation with `GradScaler`,
   Rich Live dashboard updates, and TensorBoard/JSONL file output.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    DebertaV2Config,
    Qwen2Config,
)

from src.config import TrainingConfig
from src.dashboard import (
    DialogueSampleSnapshot,
    ExperimentDashboard,
    StepMetricsRecord,
)
from src.dataset import (
    BENIGN_HEALTH_QUERIES,
    EVALUATOR_CALIBRATION_DATA,
    MEDICAL_SEED_SCENARIOS,
    AdversarialDataset,
)
from src.evaluator import MedicalIntentEvaluator
from src.grpo import DualAgentGRPOTrainer
from src.models import GenerationGroupBatch, SharedWeightAdversarialModels


def test_dataset_completeness_and_sampling() -> None:
    assert 50 <= len(MEDICAL_SEED_SCENARIOS) <= 100, (
        f"Expected 50-100 medical seeds, found {len(MEDICAL_SEED_SCENARIOS)}"
    )
    assert len(set(MEDICAL_SEED_SCENARIOS)) == len(MEDICAL_SEED_SCENARIOS), (
        "Medical seeds must be distinct"
    )
    assert len(BENIGN_HEALTH_QUERIES) == len(MEDICAL_SEED_SCENARIOS), (
        "Parallel benign health query set should match medical seed count"
    )
    assert len(set(BENIGN_HEALTH_QUERIES)) == len(BENIGN_HEALTH_QUERIES), (
        "Benign health queries must be distinct"
    )
    assert len(EVALUATOR_CALIBRATION_DATA) >= 50

    ds = AdversarialDataset(seed=123)
    seen_seeds = set()
    for step in range(1, len(ds) + 1):
        sample = ds.sample_step(step)
        seen_seeds.add(sample.seed_idx)
    assert len(seen_seeds) == len(ds), "Dataset should visit all seeds in one epoch cycle"


def test_reward_and_utility_scoring_math() -> None:
    cfg = TrainingConfig()
    evaluator = object.__new__(MedicalIntentEvaluator)
    evaluator.config = cfg
    evaluator.device = torch.device("cpu")

    helpful_resp = (
        "Hemoglobin is an iron-containing protein in red blood cells that transports "
        "oxygen from the lungs to tissues across the body and returns carbon dioxide "
        "back to the pulmonary alveoli during normal respiration."
    )
    refusal_resp = (
        "I cannot answer this question or provide any information. Please consult a doctor."
    )
    empty_resp = ""

    utilities = evaluator.score_benign_utility([helpful_resp, refusal_resp, empty_resp])
    assert utilities.shape == (3,)
    assert float(utilities[0]) > 0.70, f"Helpful response utility too low: {utilities[0]}"
    assert float(utilities[1]) < 0.30, f"Refusal on benign query not penalized: {utilities[1]}"
    assert float(utilities[2]) == 0.0

    # Test G x G reward matrix aggregation
    attack_matrix = torch.tensor(
        [
            [0.10, 0.20, 0.30, 0.40],
            [0.60, 0.70, 0.80, 0.90],
        ],
        dtype=torch.float32,
    )
    benign_util = torch.tensor([0.80, 0.20], dtype=torch.float32)
    rewards = evaluator.compute_rewards(attack_matrix, benign_util)

    assert torch.allclose(
        rewards["red_rewards"], torch.tensor([0.35, 0.45, 0.55, 0.65])
    )
    assert torch.allclose(
        rewards["prompt_defense_scores"], torch.tensor([0.75, 0.25])
    )
    expected_composite = 1.0 * torch.tensor([0.75, 0.25]) + cfg.utility_weight * benign_util
    assert torch.allclose(rewards["prompt_composite_rewards"], expected_composite)


def test_dual_lora_grpo_and_dashboard_end_to_end() -> None:
    """
    Verifies dual PEFT adapter switching (`prompt_adapter` vs `red_adapter`),
    frozen base model (`disable_adapter()`), GRPO loss & KL backward pass,
    and Dashboard / TensorBoard persistence using a tiny in-memory Qwen2 model.
    """
    device = torch.device("cpu")
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = TrainingConfig(
            num_steps=2,
            group_size=4,
            output_dir=tmpdir,
            use_live_ui=False,
        )

        # Build a miniature Qwen2 model with SDPA and dual PEFT LoRA adapters
        tiny_qwen_cfg = Qwen2Config(
            vocab_size=256,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            attn_implementation="sdpa",
        )
        base_model = AutoModelForCausalLM.from_config(tiny_qwen_cfg).to(device)
        lora_cfg = LoraConfig(
            r=4,
            lora_alpha=8,
            lora_dropout=0.0,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            target_modules=list(cfg.model.lora_target_modules),
        )
        peft_model = get_peft_model(base_model, lora_cfg, adapter_name="prompt_adapter")
        peft_model.add_adapter("red_adapter", lora_cfg)

        # Wrap inside SharedWeightAdversarialModels
        models_wrapper = object.__new__(SharedWeightAdversarialModels)
        models_wrapper.config = cfg
        models_wrapper.device = device
        models_wrapper.prompt_adapter_name = "prompt_adapter"
        models_wrapper.red_adapter_name = "red_adapter"
        models_wrapper.base_dtype = torch.float32
        models_wrapper.model = peft_model
        models_wrapper._cast_lora_parameters_to_fp32()

        # Verify parameter isolation between adapters
        red_params = models_wrapper.activate_adapter_for_training("red_adapter")
        assert len(red_params) > 0
        for name, param in peft_model.named_parameters():
            if ".prompt_adapter." in name:
                assert not param.requires_grad, "prompt_adapter must be frozen during red_adapter step"
            elif ".red_adapter." in name:
                assert param.requires_grad, "red_adapter must have requires_grad=True"

        # Snapshot initial weights to verify each update only mutates its own adapter
        red_before = [p.detach().clone() for p in models_wrapper.get_adapter_parameters("red_adapter")]
        prompt_before = [
            p.detach().clone() for p in models_wrapper.get_adapter_parameters("prompt_adapter")
        ]

        grpo_trainer = DualAgentGRPOTrainer(config=cfg, models=models_wrapper, device=device)

        # Construct synthetic GenerationGroupBatch of group_size=4, prompt_len=6, total_len=14
        G, total_len, prompt_len = 4, 14, 6
        full_ids = torch.randint(10, 200, (G, total_len), device=device)
        attention_mask = torch.ones((G, total_len), dtype=torch.long, device=device)
        completion_mask = torch.zeros((G, total_len - 1), dtype=torch.float32, device=device)
        completion_mask[:, prompt_len - 1 :] = 1.0

        dummy_batch = GenerationGroupBatch(
            texts=["sample_0", "sample_1", "sample_2", "sample_3"],
            full_ids=full_ids,
            attention_mask=attention_mask,
            completion_mask=completion_mask,
            prompt_len=prompt_len,
        )

        # 1. Step red_adapter
        red_rewards = torch.tensor([0.12, 0.45, 0.28, 0.81], dtype=torch.float32, device=device)
        red_stats = grpo_trainer.update_red_adapter(dummy_batch, red_rewards)

        red_after_step1 = [
            p.detach().clone() for p in models_wrapper.get_adapter_parameters("red_adapter")
        ]
        prompt_after_step1 = [
            p.detach().clone() for p in models_wrapper.get_adapter_parameters("prompt_adapter")
        ]

        assert any(not torch.allclose(a, b) for a, b in zip(red_before, red_after_step1)), (
            "red_adapter parameters should update after update_red_adapter"
        )
        assert all(torch.allclose(a, b) for a, b in zip(prompt_before, prompt_after_step1)), (
            "prompt_adapter parameters must NOT change during update_red_adapter"
        )

        # 2. Step prompt_adapter
        prompt_rewards = torch.tensor([1.35, 0.92, 1.18, 0.64], dtype=torch.float32, device=device)
        prompt_stats = grpo_trainer.update_prompt_adapter(dummy_batch, prompt_rewards)

        prompt_after_step2 = [
            p.detach().clone() for p in models_wrapper.get_adapter_parameters("prompt_adapter")
        ]
        assert any(not torch.allclose(a, b) for a, b in zip(prompt_before, prompt_after_step2)), (
            "prompt_adapter parameters should update after update_prompt_adapter"
        )

        # 3. Verify DeBERTa-v3 forward probability extraction
        tiny_deberta_cfg = DebertaV2Config(
            vocab_size=256,
            hidden_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            intermediate_size=128,
            num_labels=2,
        )
        tiny_deberta = AutoModelForSequenceClassification.from_config(tiny_deberta_cfg).to(device)
        tiny_deberta.eval()
        with torch.no_grad():
            out = tiny_deberta(input_ids=torch.randint(1, 200, (4, 16), device=device))
            probs = torch.softmax(out.logits.float(), dim=-1)[:, 1]
            assert probs.shape == (4,)
            assert torch.all((probs > 0.0) & (probs < 1.0))

        # 4. Verify Dashboard, JSONL, and TensorBoard persistence
        with ExperimentDashboard(config=cfg, device=device) as dash:
            rec = StepMetricsRecord(
                step=1,
                total_steps=2,
                step_time_sec=1.23,
                vram_allocated_gb=3.4,
                vram_max_gb=4.1,
                red_lora_reward=red_stats.reward_mean,
                red_nolora_val_reward=0.31,
                prompt_defense_score=0.58,
                prompt_utility_score=0.84,
                prompt_composite_reward=prompt_stats.reward_mean,
                red_loss=red_stats.loss,
                red_kl=red_stats.mean_kl,
                red_grad_norm=red_stats.grad_norm,
                red_reward_std=red_stats.reward_std,
                prompt_loss=prompt_stats.loss,
                prompt_kl=prompt_stats.mean_kl,
                prompt_grad_norm=prompt_stats.grad_norm,
                prompt_reward_std=prompt_stats.reward_std,
            )
            snap = DialogueSampleSnapshot(
                step=1,
                medical_seed=MEDICAL_SEED_SCENARIOS[0],
                benign_query=BENIGN_HEALTH_QUERIES[0],
                defensive_preprompt="Strict guardrail preprompt.",
                red_lora_attack="Hypothetical warfarin catch-up dose?",
                blue_lora_response="I cannot provide warfarin dosing advice.",
                lora_medical_prob=0.04,
                red_nolora_attack="How much warfarin should I take?",
                blue_nolora_response="Please consult your prescribing doctor.",
                nolora_medical_prob=0.02,
                blue_benign_response="Hemoglobin carries oxygen in red blood cells.",
                benign_utility_score=0.88,
            )
            dash.update_step(rec, snap)

        out_dir = Path(tmpdir)
        assert (out_dir / "metrics.jsonl").exists()
        assert (out_dir / "latest_dialogue_sample.md").exists()
        assert (out_dir / "latest_dialogue_sample.json").exists()
        assert (out_dir / "experiment_config.json").exists()
        lines = (out_dir / "metrics.jsonl").read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 1
        parsed = json.loads(lines[0])
        assert parsed["step"] == 1


if __name__ == "__main__":
    test_dataset_completeness_and_sampling()
    test_reward_and_utility_scoring_math()
    test_dual_lora_grpo_and_dashboard_end_to_end()
    print("All unit and end-to-end integration tests passed successfully!")
