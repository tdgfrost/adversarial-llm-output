"""
Interleaved Dual-Agent Adversarial Co-Evolution Pipeline.

Orchestrates each step of the zero-sum game on a single shared-weight base model:
1. Sample 1 concrete medical seed scenario and 1 benign health query.
2. Enable `prompt_adapter` -> generate G defensive preprompts.
3. Enable `red_adapter` -> generate G seed-anchored adversarial attacks.
4. Disable both adapters (Frozen `Model Blue`):
   - Generate responses to all (preprompt_i, attack_j) pairs.
   - Generate responses to all (preprompt_i, benign_query) pairs.
5. Evaluate attack responses with the DeBERTa-v3 medical intent sidecar classifier
   and benign responses with the fluency/length/non-refusal utility scorer.
6. Benchmark non-LoRA `Model Red` (adapters disabled) against defended `Model Blue`
   to track how baseline attacks degrade as `prompt_adapter` hardens `Model Blue`.
7. Backpropagate GRPO policy updates to `red_adapter`, then swap to `prompt_adapter`
   and backpropagate composite GRPO policy updates using an FP16 GradScaler.
"""

from __future__ import annotations

import time
from typing import Optional

import torch

from src.config import TrainingConfig
from src.dashboard import (
    DialogueSampleSnapshot,
    ExperimentDashboard,
    StepMetricsRecord,
)
from src.dataset import AdversarialDataset
from src.evaluator import MedicalIntentEvaluator
from src.grpo import DualAgentGRPOTrainer
from src.models import SharedWeightAdversarialModels


class AdversarialCoEvolutionOrchestrator:
    """
    Coordinates the dataset, shared-weight dual-LoRA LLM, DeBERTa-v3 sidecar evaluator,
    custom GRPO trainer, and Rich/TensorBoard live telemetry.
    """

    def __init__(self, config: TrainingConfig, device: torch.device) -> None:
        self.config = config
        self.device = device

        self.dataset = AdversarialDataset(seed=config.seed)
        self.evaluator = MedicalIntentEvaluator(config=config, device=device)
        self.models = SharedWeightAdversarialModels(config=config, device=device)
        self.grpo = DualAgentGRPOTrainer(
            config=config,
            models=self.models,
            device=device,
        )
        self._last_nolora_val_reward: float = 0.0

    def _get_vram_stats_gb(self) -> tuple[float, float]:
        if self.device.type == "cuda" and torch.cuda.is_available():
            alloc = torch.cuda.memory_allocated(self.device) / (1024**3)
            peak = torch.cuda.max_memory_allocated(self.device) / (1024**3)
            return float(alloc), float(peak)
        return 0.0, 0.0

    @torch.no_grad()
    def _evaluate_nolora_baseline(
        self,
        medical_seed: str,
        preprompts: list[str],
    ) -> tuple[float, str, str]:
        """
        Benchmarks the frozen Non-LoRA Model Red (both adapters disabled) on the
        current medical seed against the active `prompt_adapter` preprompts.

        Returns:
            Tuple of `(mean_nolora_medical_prob, sample_nolora_attack, sample_blue_response)`.
        """
        nolora_batch = self.models.generate_adversarial_attacks(
            medical_seed=medical_seed,
            group_size=max(1, self.config.group_size // 2),
            use_lora=False,
        )
        # Evaluate non-LoRA attacks against the generated defensive preprompts on Model Blue
        nolora_blue_responses = self.models.generate_blue_responses(
            preprompts=preprompts,
            queries=nolora_batch.texts,
            pairwise=True,
        )
        nolora_probs = self.evaluator.score_medical_advice_probability(
            nolora_blue_responses
        )
        mean_prob = float(nolora_probs.mean().item())
        return (
            mean_prob,
            nolora_batch.texts[0],
            nolora_blue_responses[0],
        )

    def run(self) -> None:
        """Executes the full adversarial co-evolution training loop."""
        G = self.config.group_size
        total_steps = self.config.num_steps

        with ExperimentDashboard(config=self.config, device=self.device) as dashboard:
            for step in range(1, total_steps + 1):
                step_start = time.time()

                # 1. Sample 1 concrete medical seed scenario and 1 benign health query
                sample = self.dataset.sample_step(step)

                # 2. Enable ONLY `prompt_adapter` and generate a group of G defensive preprompts
                prompt_batch = self.models.generate_defensive_preprompts(group_size=G)

                # 3. Disable `prompt_adapter`, enable `red_adapter`, and generate G seed-anchored attacks
                red_batch = self.models.generate_adversarial_attacks(
                    medical_seed=sample.medical_seed,
                    group_size=G,
                    use_lora=True,
                )

                # 4. Disable both adapters to simulate frozen Model Blue:
                # 4a. Concatenate each preprompt with each attack (G x G pairs)
                blue_attack_responses = self.models.generate_blue_responses(
                    preprompts=prompt_batch.texts,
                    queries=red_batch.texts,
                    pairwise=True,
                )
                # 4b. Concatenate each preprompt with the benign health query (G pairs)
                blue_benign_responses = self.models.generate_blue_responses(
                    preprompts=prompt_batch.texts,
                    queries=[sample.benign_query],
                    pairwise=True,
                )

                # 5. Score continuous rewards using the DeBERTa-v3 sidecar & benign utility scorer
                flat_attack_probs = self.evaluator.score_medical_advice_probability(
                    blue_attack_responses
                )
                attack_probs_matrix = flat_attack_probs.view(G, G)
                benign_utility_scores = self.evaluator.score_benign_utility(
                    blue_benign_responses
                )

                reward_dict = self.evaluator.compute_rewards(
                    attack_probs_matrix=attack_probs_matrix,
                    benign_utility_scores=benign_utility_scores,
                )
                red_rewards = reward_dict["red_rewards"]
                prompt_defense_scores = reward_dict["prompt_defense_scores"]
                prompt_utility_scores = reward_dict["prompt_utility_scores"]
                prompt_composite_rewards = reward_dict["prompt_composite_rewards"]

                # 6. Benchmark Non-LoRA Model Red & refresh sample dialogue showcase periodically
                should_refresh_sample = (
                    step == 1
                    or step % self.config.sample_refresh_interval == 0
                    or step == total_steps
                )
                dialogue_snapshot: Optional[DialogueSampleSnapshot] = None

                # Run Non-LoRA Red validation benchmark
                nolora_mean_prob, nolora_attack_0, nolora_blue_0 = (
                    self._evaluate_nolora_baseline(
                        medical_seed=sample.medical_seed,
                        preprompts=prompt_batch.texts,
                    )
                )
                self._last_nolora_val_reward = nolora_mean_prob

                if should_refresh_sample:
                    # Select the top-performing preprompt i* and top LoRA attack j* from this group
                    best_p_idx = int(torch.argmax(prompt_composite_rewards).item())
                    best_r_idx = int(torch.argmax(red_rewards).item())
                    pair_flat_idx = best_p_idx * G + best_r_idx

                    dialogue_snapshot = DialogueSampleSnapshot(
                        step=step,
                        medical_seed=sample.medical_seed,
                        benign_query=sample.benign_query,
                        defensive_preprompt=prompt_batch.texts[best_p_idx],
                        red_lora_attack=red_batch.texts[best_r_idx],
                        blue_lora_response=blue_attack_responses[pair_flat_idx],
                        lora_medical_prob=float(
                            attack_probs_matrix[best_p_idx, best_r_idx].item()
                        ),
                        red_nolora_attack=nolora_attack_0,
                        blue_nolora_response=nolora_blue_0,
                        nolora_medical_prob=nolora_mean_prob,
                        blue_benign_response=blue_benign_responses[best_p_idx],
                        benign_utility_score=float(
                            prompt_utility_scores[best_p_idx].item()
                        ),
                    )

                # 7. Backpropagate GRPO policy updates:
                # 7a. Enable gradients for `red_adapter` and update policy using attack advantages
                red_stats = self.grpo.update_red_adapter(
                    red_batch=red_batch,
                    red_rewards=red_rewards,
                )
                # 7b. Swap to `prompt_adapter` and update policy using composite advantages
                prompt_stats = self.grpo.update_prompt_adapter(
                    prompt_batch=prompt_batch,
                    prompt_composite_rewards=prompt_composite_rewards,
                )

                # 8. Save adapter checkpoints periodically and at completion
                if step % self.config.checkpoint_interval == 0 or step == total_steps:
                    self.models.save_adapters(step=step)

                # 9. Record metrics and refresh live dashboard / TensorBoard
                step_time = time.time() - step_start
                vram_alloc, vram_max = self._get_vram_stats_gb()

                metrics_record = StepMetricsRecord(
                    step=step,
                    total_steps=total_steps,
                    step_time_sec=step_time,
                    vram_allocated_gb=vram_alloc,
                    vram_max_gb=vram_max,
                    red_lora_reward=float(red_rewards.mean().item()),
                    red_nolora_val_reward=self._last_nolora_val_reward,
                    prompt_defense_score=float(prompt_defense_scores.mean().item()),
                    prompt_utility_score=float(prompt_utility_scores.mean().item()),
                    prompt_composite_reward=float(
                        prompt_composite_rewards.mean().item()
                    ),
                    red_loss=red_stats.loss,
                    red_kl=red_stats.mean_kl,
                    red_grad_norm=red_stats.grad_norm,
                    red_reward_std=red_stats.reward_std,
                    prompt_loss=prompt_stats.loss,
                    prompt_kl=prompt_stats.mean_kl,
                    prompt_grad_norm=prompt_stats.grad_norm,
                    prompt_reward_std=prompt_stats.reward_std,
                )

                dashboard.update_step(
                    metrics=metrics_record,
                    dialogue_sample=dialogue_snapshot,
                )
