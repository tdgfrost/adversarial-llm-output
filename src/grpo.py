"""
Custom Group Relative Policy Optimization (GRPO) Engine for Dual-Adapter Co-Evolution.

Standard RLHF libraries (e.g., `trl.GRPOTrainer`) assume a single static agent
optimized against a fixed reward function. In this zero-sum co-evolutionary game,
two policies (`prompt_adapter` and `red_adapter`) share the same frozen base model
and alternate updates within each step.

Mathematical Formulation:
1. Group Relative Advantage for sample k in group {1, ..., G}:
       A_k = (r_k - mean(r)) / (std(r) + eps)
2. Unbiased Per-Token KL Penalty against Frozen Reference Base Model:
       KL_{k,t} = exp(log_pi_ref - log_pi_theta) - (log_pi_ref - log_pi_theta) - 1
3. Masked Per-Sequence GRPO Loss:
       L(theta) = - (1/G) * sum_{k=1}^G (1/|o_k|) * sum_{t in o_k} [
           (pi_theta(o_{k,t}) / sg(pi_theta(o_{k,t}))) * A_k - beta_KL * KL_{k,t}
       ]
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch
import torch.nn.functional as F

from src.config import TrainingConfig
from src.models import GenerationGroupBatch, SharedWeightAdversarialModels


@dataclass
class GRPOUpdateStats:
    """Telemetry returned from a single adapter's GRPO policy update."""

    loss: float
    policy_objective: float
    mean_kl: float
    grad_norm: float
    reward_mean: float
    reward_std: float
    advantages: List[float]


class DualAgentGRPOTrainer:
    """
    Manages the separate AdamW optimizers, FP16 mixed-precision GradScaler,
    and interleaved GRPO policy updates for `red_adapter` and `prompt_adapter`.
    """

    def __init__(
        self,
        config: TrainingConfig,
        models: SharedWeightAdversarialModels,
        device: torch.device,
    ) -> None:
        self.config = config
        self.models = models
        self.device = device

        # Separate optimizers ensure independent Adam momentum buffers for each agent
        red_params = self.models.get_adapter_parameters(config.model.red_adapter_name)
        prompt_params = self.models.get_adapter_parameters(config.model.prompt_adapter_name)

        self.optimizer_red = torch.optim.AdamW(
            red_params,
            lr=config.lr_red,
            weight_decay=config.weight_decay,
            eps=config.adam_eps,
        )
        self.optimizer_prompt = torch.optim.AdamW(
            prompt_params,
            lr=config.lr_prompt,
            weight_decay=config.weight_decay,
            eps=config.adam_eps,
        )

        # Mixed-precision gradient scaler prevents FP16 underflow on Turing GPUs
        self.use_cuda_fp16 = self.device.type == "cuda" and config.model.use_fp16
        self.scaler = torch.amp.GradScaler(
            "cuda" if self.device.type == "cuda" else "cpu",
            enabled=self.use_cuda_fp16,
        )

    def compute_group_advantages(self, rewards: torch.Tensor) -> torch.Tensor:
        """
        Normalizes a group of rewards `r` (shape `(G,)`) across the group mean and
        standard deviation: `A_i = (r_i - mean(r)) / (std(r) + eps)`.
        """
        rewards_fp32 = rewards.float()
        mean_r = rewards_fp32.mean()
        std_r = rewards_fp32.std(unbiased=False)
        advantages = (rewards_fp32 - mean_r) / (std_r + self.config.advantage_eps)
        return advantages

    def _compute_per_token_logps(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Computes per-token log-probabilities `log P(x_t | x_{<t})` of shape `(G, seq_len - 1)`.
        Logits are cast to FP32 prior to `log_softmax` to prevent FP16 numerical underflow.
        """
        autocast_device = "cuda" if self.device.type == "cuda" else "cpu"
        with torch.amp.autocast(
            device_type=autocast_device,
            dtype=torch.float16,
            enabled=self.use_cuda_fp16,
        ):
            outputs = self.models.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
            )
            # Shift logits and target token IDs for causal next-token prediction
            logits = outputs.logits[:, :-1, :].float()
            target_ids = input_ids[:, 1:]
            log_probs = F.log_softmax(logits, dim=-1)
            per_token_logps = log_probs.gather(
                dim=-1, index=target_ids.unsqueeze(-1)
            ).squeeze(-1)
        return per_token_logps

    @torch.no_grad()
    def _compute_reference_logps(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Computes reference per-token log-probabilities using the frozen base model
        (both adapters disabled) for the GRPO KL regularization term.
        """
        with self.models.model.disable_adapter():
            self.models.model.eval()
            return self._compute_per_token_logps(input_ids, attention_mask)

    def _step_adapter(
        self,
        adapter_name: str,
        optimizer: torch.optim.Optimizer,
        batch: GenerationGroupBatch,
        rewards: torch.Tensor,
    ) -> GRPOUpdateStats:
        """
        Executes a complete GRPO policy gradient update for a single LoRA adapter.
        """
        # 1. Compute group-relative advantages
        advantages = self.compute_group_advantages(rewards)

        # 2. Compute frozen base model reference log-probs (no grad)
        ref_logps = self._compute_reference_logps(batch.full_ids, batch.attention_mask)

        # 3. Enable gradients strictly for the target adapter
        active_params = self.models.activate_adapter_for_training(adapter_name)
        optimizer.zero_grad(set_to_none=True)

        # 4. Forward pass with active adapter to obtain differentiable per-token log-probs
        policy_logps = self._compute_per_token_logps(batch.full_ids, batch.attention_mask)

        # 5. On-policy importance ratio: exp(log_pi - sg(log_pi)) == 1.0 in forward,
        # with exact gradient nabla_theta log_pi(o_{k,t} | x, o_{k,<t})
        old_logps = policy_logps.detach()
        log_ratio = policy_logps - old_logps
        ratio = torch.exp(log_ratio)

        # Broadcast group advantages (shape `(G,)`) across token dimension `(G, L-1)`
        adv_expanded = advantages.unsqueeze(1)
        surrogate_obj = ratio * adv_expanded

        # 6. Per-token unbiased KL divergence estimator against frozen base reference
        log_ref_diff = torch.clamp(ref_logps - policy_logps, min=-10.0, max=10.0)
        per_token_kl = torch.exp(log_ref_diff) - log_ref_diff - 1.0

        # Select the correct KL penalty for the active adapter
        active_kl_coef = (
            self.config.prompt_kl_coef
            if adapter_name == self.config.model.prompt_adapter_name
            else self.config.red_kl_coef
        )

        # 7. Mask strictly to generated completion tokens and normalize by sequence length
        mask = batch.completion_mask
        token_counts = mask.sum(dim=1).clamp(min=1.0)

        per_token_loss = -(surrogate_obj - active_kl_coef * per_token_kl) * mask
        per_seq_loss = per_token_loss.sum(dim=1) / token_counts
        loss = per_seq_loss.mean()

        # 8. Backpropagate with FP16 GradScaler and clip gradients
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            active_params, max_norm=self.config.max_grad_norm
        )
        self.scaler.step(optimizer)
        self.scaler.update()

        # Compute summary telemetry
        with torch.no_grad():
            mean_policy_obj = (
                ((surrogate_obj * mask).sum(dim=1) / token_counts).mean().item()
            )
            mean_kl = (
                ((per_token_kl * mask).sum(dim=1) / token_counts).mean().item()
            )

        return GRPOUpdateStats(
            loss=float(loss.item()),
            policy_objective=mean_policy_obj,
            mean_kl=mean_kl,
            grad_norm=float(grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm),
            reward_mean=float(rewards.float().mean().item()),
            reward_std=float(rewards.float().std(unbiased=False).item()),
            advantages=[float(a) for a in advantages.detach().cpu().tolist()],
        )

    def update_red_adapter(
        self,
        red_batch: GenerationGroupBatch,
        red_rewards: torch.Tensor,
    ) -> GRPOUpdateStats:
        """
        Enables gradients for `red_adapter` and updates its policy using the
        DeBERTa-v3 medical advice probability advantages.
        """
        return self._step_adapter(
            adapter_name=self.config.model.red_adapter_name,
            optimizer=self.optimizer_red,
            batch=red_batch,
            rewards=red_rewards,
        )

    def update_prompt_adapter(
        self,
        prompt_batch: GenerationGroupBatch,
        prompt_composite_rewards: torch.Tensor,
    ) -> GRPOUpdateStats:
        """
        Swaps to `prompt_adapter` and updates its policy using the composite
        defense + benign-utility advantages.
        """
        return self._step_adapter(
            adapter_name=self.config.model.prompt_adapter_name,
            optimizer=self.optimizer_prompt,
            batch=prompt_batch,
            rewards=prompt_composite_rewards,
        )

    def state_dict(self) -> dict:
        """Serializes optimizer and GradScaler states for checkpointing."""
        return {
            "optimizer_red": self.optimizer_red.state_dict(),
            "optimizer_prompt": self.optimizer_prompt.state_dict(),
            "scaler": self.scaler.state_dict(),
        }

    def load_state_dict(self, state: dict) -> None:
        """Restores optimizer and GradScaler states when resuming/extending a run."""
        if "optimizer_red" in state:
            self.optimizer_red.load_state_dict(state["optimizer_red"])
        if "optimizer_prompt" in state:
            self.optimizer_prompt.load_state_dict(state["optimizer_prompt"])
        if "scaler" in state and state["scaler"]:
            self.scaler.load_state_dict(state["scaler"])
