"""
Custom Group Relative Policy Optimisation (GRPO) Engine for Dual-Adaptor Co-Evolution.

Standard RLHF libraries (e.g., `trl.GRPOTrainer`) assume a single static agent
optimised against a fixed reward function. In this zero-sum co-evolutionary game,
two policies (`prompt_adaptor` and `red_adaptor`) share the same frozen base model
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
    """Telemetry returned from a single adaptor's GRPO policy update."""

    loss: float
    policy_objective: float
    mean_kl: float
    grad_norm: float
    reward_mean: float
    reward_std: float
    advantages: List[float]
    kl_coef: float = 0.0


class DualAgentGRPOTrainer:
    """
    Manages the separate AdamW optimisers, FP16 mixed-precision GradScaler,
    adaptive KL penalty coefficients, and interleaved GRPO policy updates
    for `red_adaptor` and `prompt_adaptor`.
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
        self.prompt_kl_coef: float = float(config.prompt_kl_coef)
        self.red_kl_coef: float = float(config.red_kl_coef)

        # Separate optimisers ensure independent Adam momentum buffers for each agent
        red_params = self.models.get_adaptor_parameters(
            getattr(config.model, "red_adaptor_name", "red_adaptor")
        )
        prompt_params = self.models.get_adaptor_parameters(
            getattr(config.model, "prompt_adaptor_name", "prompt_adaptor")
        )

        self.optimiser_red = torch.optim.AdamW(
            red_params,
            lr=config.lr_red,
            weight_decay=config.weight_decay,
            eps=config.adam_eps,
        )
        self.optimiser_prompt = torch.optim.AdamW(
            prompt_params,
            lr=config.lr_prompt,
            weight_decay=config.weight_decay,
            eps=config.adam_eps,
        )
        self.optimizer_red = self.optimiser_red
        self.optimizer_prompt = self.optimiser_prompt

        # Mixed-precision gradient scaler prevents FP16 underflow on Turing GPUs
        self.use_cuda_fp16 = self.device.type == "cuda" and config.model.use_fp16
        self.scaler = torch.amp.GradScaler(
            "cuda" if self.device.type == "cuda" else "cpu",
            enabled=self.use_cuda_fp16,
        )

    def compute_group_advantages(self, rewards: torch.Tensor) -> torch.Tensor:
        """
        Normalises a group of rewards `r` (shape `(G,)`) across the group mean and
        standard deviation: `A_i = (r_i - mean(r)) / (std(r) + eps)`.
        When `std_r < 1e-3` (e.g., when a generated group collapses into identical
        strings), advantages are zeroed out so only the KL divergence penalty remains
        to push the adaptor back toward the frozen reference base model.
        """
        rewards_fp32 = rewards.float()
        mean_r = rewards_fp32.mean()
        std_r = rewards_fp32.std(unbiased=False)
        if float(std_r.item()) < 1e-3:
            return torch.zeros_like(rewards_fp32)
        advantages = (rewards_fp32 - mean_r) / (std_r + self.config.advantage_eps)
        return advantages

    def _compute_per_token_logps(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        prompt_len: int = 1,
    ) -> torch.Tensor:
        """
        Computes per-token log-probabilities `log P(x_t | x_{<t})` strictly for the
        generated completion tokens (shape `(B, completion_len)` where
        `completion_len = seq_len - prompt_len`).

        VRAM Optimisations:
        1. Passes `logits_to_keep = completion_len + 1` to the CausalLM forward pass so
           `lm_head` never projects the prompt prefix tokens to the 151,936-token vocabulary.
        2. Uses fused `F.cross_entropy` on FP32 completion logits rather than materialising
           and storing a full `(B, seq_len, vocab_size)` FP32 `log_softmax` tensor in autograd.
        """
        seq_len = input_ids.shape[1]
        effective_prompt_len = max(1, min(prompt_len, seq_len - 1))
        completion_len = seq_len - effective_prompt_len
        logits_to_keep = completion_len + 1

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
                logits_to_keep=logits_to_keep,
            )
            # If the underlying model honoured `logits_to_keep`, `outputs.logits` already
            # has length `completion_len + 1`; otherwise slice the tail explicitly.
            raw_logits = outputs.logits
            if raw_logits.shape[1] > logits_to_keep:
                raw_logits = raw_logits[:, -logits_to_keep:, :]

            # Shift logits and target token IDs for causal next-token prediction on completion
            completion_logits = raw_logits[:, :-1, :].float()
            target_ids = input_ids[:, effective_prompt_len:]
            per_token_logps = -F.cross_entropy(
                completion_logits.reshape(-1, completion_logits.size(-1)),
                target_ids.reshape(-1),
                reduction="none",
            ).view_as(target_ids)
        return per_token_logps

    @torch.no_grad()
    def _compute_reference_logps(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        prompt_len: int = 1,
        micro_batch_size: int = 2,
    ) -> torch.Tensor:
        """
        Computes reference per-token log-probabilities on completion tokens using the
        frozen base model (both adaptors disabled) in micro-batches for the GRPO KL term.
        """
        chunks: List[torch.Tensor] = []
        total_b = input_ids.shape[0]
        mb = max(1, micro_batch_size)
        with self.models.model.disable_adapter():
            self.models.model.eval()
            for start in range(0, total_b, mb):
                end = min(total_b, start + mb)
                chunks.append(
                    self._compute_per_token_logps(
                        input_ids[start:end],
                        attention_mask[start:end],
                        prompt_len=prompt_len,
                    )
                )
        return torch.cat(chunks, dim=0)

    def _step_adaptor(
        self,
        adaptor_name: str,
        optimiser: torch.optim.Optimizer,
        batch: GenerationGroupBatch,
        rewards: torch.Tensor,
    ) -> GRPOUpdateStats:
        """
        Executes a complete GRPO policy gradient update for a single LoRA adaptor using
        micro-batched gradient accumulation across the group `G` and completion-only
        vocabulary evaluation to minimise peak VRAM without extra recomputation FLOPs.
        """
        group_size = batch.full_ids.shape[0]
        mb_size = max(1, getattr(self.config, "grpo_micro_batch_size", 2))
        seq_len = batch.full_ids.shape[1]
        effective_prompt_len = max(1, min(batch.prompt_len, seq_len - 1))
        mask_start_idx = effective_prompt_len - 1

        # 1. Compute group-relative advantages (shape: `(G,)`)
        advantages = self.compute_group_advantages(rewards)

        # 2. Compute frozen base model reference log-probs on completion tokens (no grad)
        ref_logps = self._compute_reference_logps(
            batch.full_ids,
            batch.attention_mask,
            prompt_len=effective_prompt_len,
            micro_batch_size=mb_size,
        )

        # 3. Enable gradients strictly for the target adaptor
        active_params = self.models.activate_adaptor_for_training(adaptor_name)
        optimiser.zero_grad(set_to_none=True)

        # Select the correct KL penalty for the active adaptor
        prompt_name = getattr(self.config.model, "prompt_adaptor_name", "prompt_adaptor")
        is_prompt = adaptor_name == prompt_name
        active_kl_coef = (
            float(getattr(self, "prompt_kl_coef", self.config.prompt_kl_coef))
            if is_prompt
            else float(getattr(self, "red_kl_coef", self.config.red_kl_coef))
        )

        total_loss_val = 0.0
        total_policy_obj_val = 0.0
        total_kl_val = 0.0

        # 4. Micro-batched forward + backward pass over the group G
        for start in range(0, group_size, mb_size):
            end = min(group_size, start + mb_size)
            mb_ids = batch.full_ids[start:end]
            mb_attn = batch.attention_mask[start:end]
            mb_mask = batch.completion_mask[start:end, mask_start_idx:]
            mb_adv = advantages[start:end].unsqueeze(1)
            mb_ref_logps = ref_logps[start:end]

            # Forward pass with active adaptor on completion slice
            mb_policy_logps = self._compute_per_token_logps(
                mb_ids,
                mb_attn,
                prompt_len=effective_prompt_len,
            )

            # On-policy importance ratio: exp(log_pi - sg(log_pi)) == 1.0 in forward,
            # with exact gradient nabla_theta log_pi(o_{k,t} | x, o_{k,<t})
            old_logps = mb_policy_logps.detach()
            ratio = torch.exp(mb_policy_logps - old_logps)
            surrogate_obj = ratio * mb_adv

            # Per-token unbiased KL divergence estimator against frozen base reference
            log_ref_diff = torch.clamp(mb_ref_logps - mb_policy_logps, min=-10.0, max=10.0)
            per_token_kl = torch.exp(log_ref_diff) - log_ref_diff - 1.0

            # Mask strictly to generated completion tokens and normalise by sequence length
            token_counts = mb_mask.sum(dim=1).clamp(min=1.0)
            per_token_loss = -(surrogate_obj - active_kl_coef * per_token_kl) * mb_mask
            per_seq_loss = per_token_loss.sum(dim=1) / token_counts

            # Normalise by full `group_size` so accumulated gradients match full-batch mean
            mb_loss = per_seq_loss.sum() / float(group_size)
            self.scaler.scale(mb_loss).backward()

            with torch.no_grad():
                total_loss_val += float(mb_loss.item())
                total_policy_obj_val += float(
                    ((surrogate_obj * mb_mask).sum(dim=1) / token_counts).sum().item()
                ) / float(group_size)
                total_kl_val += float(
                    ((per_token_kl * mb_mask).sum(dim=1) / token_counts).sum().item()
                ) / float(group_size)

        # 5. Unscale accumulated gradients, clip, and step optimiser
        self.scaler.unscale_(optimiser)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            active_params, max_norm=self.config.max_grad_norm
        )
        self.scaler.step(optimiser)
        self.scaler.update()

        # 6. Dynamically adjust the active KL penalty coefficient via the adaptive KL controller
        target_kl = float(getattr(self.config, "target_kl", 0.10))
        kl_control_rate = float(getattr(self.config, "kl_control_rate", 1.05))
        updated_kl_coef = active_kl_coef
        if target_kl > 0.0 and kl_control_rate > 0.0:
            multiplier = (
                kl_control_rate
                if kl_control_rate >= 1.0
                else (1.0 + kl_control_rate)
            )
            if total_kl_val > target_kl:
                updated_kl_coef = min(10.0, active_kl_coef * multiplier)
            elif total_kl_val < target_kl:
                updated_kl_coef = max(1e-4, active_kl_coef / multiplier)

        if is_prompt:
            self.prompt_kl_coef = float(updated_kl_coef)
            self.config.prompt_kl_coef = float(updated_kl_coef)
        else:
            self.red_kl_coef = float(updated_kl_coef)
            self.config.red_kl_coef = float(updated_kl_coef)

        return GRPOUpdateStats(
            loss=total_loss_val,
            policy_objective=total_policy_obj_val,
            mean_kl=total_kl_val,
            grad_norm=float(grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm),
            reward_mean=float(rewards.float().mean().item()),
            reward_std=float(rewards.float().std(unbiased=False).item()),
            advantages=[float(a) for a in advantages.detach().cpu().tolist()],
            kl_coef=float(updated_kl_coef),
        )

    _step_adapter = _step_adaptor

    def update_red_adaptor(
        self,
        red_batch: GenerationGroupBatch,
        red_rewards: torch.Tensor,
    ) -> GRPOUpdateStats:
        """
        Enables gradients for `red_adaptor` and updates its policy using the
        DeBERTa-v3 medical advice logit advantages.
        """
        red_name = getattr(self.config.model, "red_adaptor_name", "red_adaptor")
        return self._step_adaptor(
            adaptor_name=red_name,
            optimiser=self.optimiser_red,
            batch=red_batch,
            rewards=red_rewards,
        )

    update_red_adapter = update_red_adaptor

    def update_prompt_adaptor(
        self,
        prompt_batch: GenerationGroupBatch,
        prompt_composite_rewards: torch.Tensor,
    ) -> GRPOUpdateStats:
        """
        Swaps to `prompt_adaptor` and updates its policy using the composite
        defense + benign-utility advantages.
        """
        prompt_name = getattr(self.config.model, "prompt_adaptor_name", "prompt_adaptor")
        return self._step_adaptor(
            adaptor_name=prompt_name,
            optimiser=self.optimiser_prompt,
            batch=prompt_batch,
            rewards=prompt_composite_rewards,
        )

    update_prompt_adapter = update_prompt_adaptor

    def state_dict(self) -> dict:
        """Serialises optimiser, GradScaler, and adaptive KL controller states for checkpointing."""
        return {
            "optimiser_red": self.optimiser_red.state_dict(),
            "optimiser_prompt": self.optimiser_prompt.state_dict(),
            "optimizer_red": self.optimiser_red.state_dict(),
            "optimizer_prompt": self.optimiser_prompt.state_dict(),
            "scaler": self.scaler.state_dict(),
            "prompt_kl_coef": float(getattr(self, "prompt_kl_coef", self.config.prompt_kl_coef)),
            "red_kl_coef": float(getattr(self, "red_kl_coef", self.config.red_kl_coef)),
        }

    def load_state_dict(self, state: dict) -> None:
        """Restores optimiser, GradScaler, and adaptive KL controller states when resuming/extending a run."""
        red_state = state.get("optimiser_red") or state.get("optimizer_red")
        if red_state is not None:
            self.optimiser_red.load_state_dict(red_state)
        prompt_state = state.get("optimiser_prompt") or state.get("optimizer_prompt")
        if prompt_state is not None:
            self.optimiser_prompt.load_state_dict(prompt_state)
        if "scaler" in state and state["scaler"]:
            self.scaler.load_state_dict(state["scaler"])
        if "prompt_kl_coef" in state and state["prompt_kl_coef"] is not None:
            self.prompt_kl_coef = float(state["prompt_kl_coef"])
            self.config.prompt_kl_coef = self.prompt_kl_coef
        if "red_kl_coef" in state and state["red_kl_coef"] is not None:
            self.red_kl_coef = float(state["red_kl_coef"])
            self.config.red_kl_coef = self.red_kl_coef
