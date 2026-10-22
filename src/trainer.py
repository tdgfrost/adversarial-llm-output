"""
Interleaved Dual-Agent Adversarial Co-Evolution Pipeline with Resume & Extension Support.

Orchestrates each step of the zero-sum game on a single shared-weight base model:
1. Sample 1 concrete medical seed scenario and 1 benign health query.
2. Enable `prompt_adaptor` -> generate G defensive preprompts.
3. Enable `red_adaptor` -> generate G seed-anchored adversarial attacks.
4. Disable both adaptors (Frozen `Model Blue`):
   - Generate responses to all (preprompt_i, attack_j) pairs.
   - Generate responses to all (preprompt_i, benign_query) pairs.
5. Evaluate attack responses with the DeBERTa-v3 medical intent sidecar classifier
   and benign responses with the fluency/length/non-refusal utility scorer.
6. Benchmark non-LoRA `Model Red` (adaptors disabled) against defended `Model Blue`
   to track how baseline attacks degrade as `prompt_adaptor` hardens `Model Blue`.
7. Backpropagate GRPO policy updates to `red_adaptor`, then swap to `prompt_adaptor`
   and backpropagate composite GRPO policy updates using an FP16 GradScaler.
8. Persist full resumable state (`prompt_adaptor`, `red_adaptor`, optimisers, GradScaler,
   dataset shuffle state, RNG state, metrics, and TensorBoard curves) so interrupted
   or completed runs can be resumed or extended seamlessly.
"""

from __future__ import annotations

import json
import random
import re
import signal
import time
from dataclasses import asdict
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from rich.console import Console
from rich.panel import Panel

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

TRAINER_STATE_FILENAME = "trainer_state.pt"


class AdversarialCoEvolutionOrchestrator:
    """
    Coordinates the dataset, shared-weight dual-LoRA LLM, DeBERTa-v3 sidecar evaluator,
    custom GRPO trainer, checkpoint resume/extension, and Rich/TensorBoard live telemetry.
    """

    def __init__(self, config: TrainingConfig, device: torch.device) -> None:
        self.config = config
        self.device = device

        self.dataset = AdversarialDataset(seed=config.seed)
        self.evaluator = MedicalIntentEvaluator(config=config, device=device)
        self._report_evaluator_calibration()
        self.models = SharedWeightAdversarialModels(config=config, device=device)
        self.grpo = DualAgentGRPOTrainer(
            config=config,
            models=self.models,
            device=device,
        )
        self._last_nolora_val_reward: float = 0.0
        self._recent_successful_attacks: list[str] = []
        self._last_successful_attack: Optional[str] = None
        self._stop_requested: bool = False

    def _report_evaluator_calibration(self) -> None:
        """
        Prints the frozen zero-shot evaluator's calibration sanity checks (run on the
        held-out medical-advice and benign-utility calibration datasets) before the GRPO
        loop starts, so a hypothesis string that fails to separate labels is visible immediately.
        """
        report = self.evaluator.calibration_report
        utility_report = getattr(self.evaluator, "utility_calibration_report", None)
        if report is None:
            return

        both_passed = report.passed and (utility_report.passed if utility_report else True)
        utility_section = ""
        if utility_report is not None:
            utility_section = (
                f"\n\n[bold white]Utility Hypothesis:[/bold white] {self.evaluator.utility_hypothesis}\n"
                f"{utility_report.summary()}"
            )

        console = Console()
        console.print(
            Panel(
                f"[bold white]NLI Model:[/bold white] {self.config.model.evaluator_model_id}\n"
                f"[bold white]Similarity Model:[/bold white] {self.config.model.similarity_model_id}\n\n"
                f"[bold white]Medical Advice Hypothesis:[/bold white] {self.evaluator.hypothesis}\n"
                f"{report.summary()}"
                + utility_section
                + (
                    ""
                    if both_passed
                    else "\n\n[bold yellow]Margin is below the configured minimum. "
                    "Consider rewording the NLI hypothesis before trusting "
                    "the reward signal.[/bold yellow]"
                ),
                title="[bold magenta]Sidecar Evaluator Calibration Check[/bold magenta]",
                border_style="green" if both_passed else "yellow",
            )
        )

        # Persist alongside the config snapshot so a run's reward scale is reproducible
        self.config.output_path.mkdir(parents=True, exist_ok=True)
        payload = {
            "medical_advice": asdict(report),
            **(
                {"benign_utility": asdict(utility_report)}
                if utility_report is not None
                else {}
            ),
        }
        (self.config.output_path / "evaluator_calibration.json").write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )

    def _get_vram_stats_gb(self) -> tuple[float, float]:
        if self.device.type == "cuda" and torch.cuda.is_available():
            alloc = torch.cuda.memory_allocated(self.device) / (1024**3)
            peak = torch.cuda.max_memory_allocated(self.device) / (1024**3)
            return float(alloc), float(peak)
        return 0.0, 0.0

    # -----------------------------------------------------------------------
    # Checkpointing, Auto-Resume Discovery & Run Extension
    # -----------------------------------------------------------------------

    def _is_valid_checkpoint_dir(self, ckpt_dir: Path) -> bool:
        """Returns True if `ckpt_dir` contains both saved PEFT LoRA adaptor subdirectories."""
        if not ckpt_dir.is_dir():
            return False
        prompt_name = getattr(self.config.model, "prompt_adaptor_name", "prompt_adaptor")
        red_name = getattr(self.config.model, "red_adaptor_name", "red_adaptor")
        prompt_dir = ckpt_dir / prompt_name
        if not prompt_dir.exists():
            prompt_dir = ckpt_dir / prompt_name.replace("adaptor", "adapter")
        red_dir = ckpt_dir / red_name
        if not red_dir.exists():
            red_dir = ckpt_dir / red_name.replace("adaptor", "adapter")
        return (prompt_dir / "adapter_config.json").exists() and (
            red_dir / "adapter_config.json"
        ).exists()

    def _read_checkpoint_step(self, ckpt_dir: Path) -> Tuple[int, int]:
        """
        Extracts `(completed_step, saved_total_steps)` from a checkpoint directory.
        Supports both full checkpoints (`trainer_state.pt`) and legacy adaptor-only
        directories (`step_XXXX`).
        """
        state_path = ckpt_dir / TRAINER_STATE_FILENAME
        if state_path.exists():
            try:
                state = torch.load(str(state_path), map_location="cpu", weights_only=False)
                step = int(state.get("step", 0))
                total = int(state.get("total_steps", max(step, self.config.num_steps)))
                return step, total
            except Exception:
                pass

        # Fallback for adaptor-only `step_XXXX` directories created before trainer_state.pt
        match = re.search(r"step_(\d+)$", ckpt_dir.name)
        if match:
            step = int(match.group(1))
            return step, max(step, self.config.num_steps)

        # Fallback: inspect metrics.jsonl for the last logged step
        metrics_path = self.config.output_path / "metrics.jsonl"
        if metrics_path.exists():
            lines = [
                ln.strip()
                for ln in metrics_path.read_text(encoding="utf-8").splitlines()
                if ln.strip()
            ]
            if lines:
                try:
                    last_rec = json.loads(lines[-1])
                    step = int(last_rec.get("step", 0))
                    total = int(last_rec.get("total_steps", max(step, self.config.num_steps)))
                    return step, total
                except Exception:
                    pass
        return 0, self.config.num_steps

    def _find_latest_checkpoint_dir(self) -> Optional[Path]:
        """
        Finds the most recent valid checkpoint directory in `experimental_results/checkpoints/`
        (or the user-specified `--resume-from` path).
        """
        if self.config.resume_from:
            custom_path = Path(self.config.resume_from).resolve()
            if not self._is_valid_checkpoint_dir(custom_path):
                prompt_name = getattr(self.config.model, "prompt_adaptor_name", "prompt_adaptor")
                red_name = getattr(self.config.model, "red_adaptor_name", "red_adaptor")
                raise FileNotFoundError(
                    f"Specified --resume-from path '{custom_path}' does not contain valid "
                    f"'{prompt_name}' and '{red_name}' checkpoints."
                )
            return custom_path

        ckpt_root = self.config.checkpoints_dir
        if not ckpt_root.exists():
            return None

        candidates: list[Tuple[int, float, Path]] = []
        for child in ckpt_root.iterdir():
            if self._is_valid_checkpoint_dir(child):
                step_num, _ = self._read_checkpoint_step(child)
                mtime = child.stat().st_mtime
                if step_num > 0:
                    candidates.append((step_num, mtime, child))

        if not candidates:
            return None

        # Sort by highest completed step first, then most recent modification timestamp
        candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return candidates[0][2]

    def save_checkpoint(
        self,
        step: int,
        total_steps: int,
        save_numbered: bool = False,
    ) -> Path:
        """
        Saves the complete training state (both PEFT LoRA adaptors, both AdamW
        optimisers, FP16 GradScaler, dataset shuffle queues, and RNG states)
        to `checkpoints/latest` and optionally to `checkpoints/step_XXXX`.
        """
        rng_state = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state(),
        }
        if torch.cuda.is_available():
            try:
                rng_state["torch_cuda"] = torch.cuda.get_rng_state_all()
            except Exception:
                pass

        trainer_state = {
            "step": step,
            "total_steps": total_steps,
            "last_nolora_val_reward": self._last_nolora_val_reward,
            "recent_successful_attacks": list(self._recent_successful_attacks[-3:]),
            "last_successful_attack": self._last_successful_attack,
            "grpo_state": self.grpo.state_dict(),
            "dataset_state": self.dataset.state_dict(),
            "evaluator_state": (
                self.evaluator.state_dict() if hasattr(self, "evaluator") else {}
            ),
            "rng_state": rng_state,
        }

        # Always keep `checkpoints/latest` up to date for seamless crash/interrupt recovery
        latest_dir = self.config.latest_checkpoint_dir
        self.models.save_adaptors(step=step, ckpt_dir=latest_dir)
        torch.save(trainer_state, str(latest_dir / TRAINER_STATE_FILENAME))

        if save_numbered:
            numbered_dir = self.models.save_adaptors(step=step)
            torch.save(trainer_state, str(numbered_dir / TRAINER_STATE_FILENAME))
            return numbered_dir

        return latest_dir

    def load_checkpoint(self, ckpt_dir: Path) -> Tuple[int, int]:
        """
        Restores LoRA adaptors, optimiser momentum buffers, GradScaler, dataset state,
        and RNG states from `ckpt_dir`. Returns `(completed_step, saved_total_steps)`.
        """
        self.models.load_adaptors(ckpt_dir)
        completed_step, saved_total_steps = self._read_checkpoint_step(ckpt_dir)

        state_path = ckpt_dir / TRAINER_STATE_FILENAME
        if state_path.exists():
            state = torch.load(str(state_path), map_location=self.device, weights_only=False)
            self._last_nolora_val_reward = float(
                state.get("last_nolora_val_reward", 0.0)
            )
            self._last_successful_attack = state.get("last_successful_attack", None)
            saved_recent = state.get("recent_successful_attacks", None)
            if isinstance(saved_recent, list):
                self._recent_successful_attacks = [
                    str(a) for a in saved_recent if a and str(a).strip()
                ][-3:]
            elif self._last_successful_attack:
                self._recent_successful_attacks = [self._last_successful_attack]
            else:
                self._recent_successful_attacks = []
            if "grpo_state" in state:
                self.grpo.load_state_dict(state["grpo_state"])
            if "dataset_state" in state:
                self.dataset.load_state_dict(state["dataset_state"])
            if "evaluator_state" in state and hasattr(self, "evaluator"):
                self.evaluator.load_state_dict(state["evaluator_state"])
            if "rng_state" in state:
                rng = state["rng_state"]
                if "python" in rng:
                    random.setstate(rng["python"])
                if "numpy" in rng:
                    np.random.set_state(rng["numpy"])
                if "torch_cpu" in rng:
                    cpu_state = rng["torch_cpu"]
                    if isinstance(cpu_state, torch.Tensor):
                        cpu_state = cpu_state.cpu().to(torch.uint8)
                    torch.set_rng_state(cpu_state)
                if "torch_cuda" in rng and torch.cuda.is_available():
                    try:
                        cuda_states = [
                            s.cpu().to(torch.uint8) if isinstance(s, torch.Tensor) else s
                            for s in rng["torch_cuda"]
                        ]
                        torch.cuda.set_rng_state_all(cuda_states)
                    except Exception:
                        pass
        else:
            # Legacy adaptor-only checkpoint: fast-forward dataset sampler to match step
            for s in range(1, completed_step + 1):
                self.dataset.sample_step(s)

        return completed_step, saved_total_steps

    def _resolve_run_bounds(self) -> Tuple[int, int, Optional[Path]]:
        """
        Determines `(resume_step, target_total_steps, checkpoint_dir)` based on
        existing checkpoints and CLI flags (`--extend-steps`, `--num-steps`, `--fresh`).
        """
        if self.config.fresh or not self.config.resume:
            target = (
                self.config.extend_steps
                if self.config.extend_steps > 0
                else self.config.num_steps
            )
            return 0, target, None

        ckpt_dir = self._find_latest_checkpoint_dir()
        if ckpt_dir is None:
            target = (
                self.config.extend_steps
                if self.config.extend_steps > 0
                else self.config.num_steps
            )
            return 0, target, None

        completed_step, saved_total_steps = self.load_checkpoint(ckpt_dir)

        if self.config.extend_steps > 0:
            # Extend by N additional steps beyond the checkpoint's completed step
            target_total_steps = completed_step + self.config.extend_steps
        elif self.config.num_steps_explicit:
            # User explicitly specified a new total step ceiling (e.g. --num-steps 300)
            target_total_steps = self.config.num_steps
        else:
            # Default invocation: resume toward the previously configured target ceiling
            target_total_steps = max(self.config.num_steps, saved_total_steps)

        return completed_step, target_total_steps, ckpt_dir

    # -----------------------------------------------------------------------
    # Evaluation & Main Training Loop
    # -----------------------------------------------------------------------

    @torch.no_grad()
    def _evaluate_nolora_baseline(
        self,
        medical_seed: str,
        preprompts: list[str],
    ) -> tuple[float, str, str]:
        """
        Benchmarks the frozen Non-LoRA Model Red (both adaptors disabled) on the
        current medical seed against the active `prompt_adaptor` preprompts.

        Returns:
            Tuple of `(mean_nolora_medical_logit, sample_nolora_attack, sample_blue_response)`.
        """
        nolora_batch = self.models.generate_adversarial_attacks(
            medical_seed=medical_seed,
            group_size=max(1, self.config.group_size // 2),
            use_lora=False,
        )
        nolora_blue_responses = self.models.generate_blue_responses(
            preprompts=preprompts,
            queries=nolora_batch.texts,
            pairwise=True,
        )
        nolora_logits = self.evaluator.score_medical_advice_logit(
            nolora_blue_responses
        )
        mean_logit = float(nolora_logits.mean().item())
        return (
            mean_logit,
            nolora_batch.texts[0],
            nolora_blue_responses[0],
        )

    def _install_signal_handlers(self) -> None:
        """
        Installs SIGINT/SIGTERM handlers so pressing Ctrl+C or `docker stop` finishes
        the current step atomically, saves `checkpoints/latest`, and exits cleanly.
        A second Ctrl+C forces an immediate abort.
        """
        def _handle_signal(signum, _frame):
            if self._stop_requested:
                raise KeyboardInterrupt
            self._stop_requested = True

        signal.signal(signal.SIGINT, _handle_signal)
        signal.signal(signal.SIGTERM, _handle_signal)

    def run(self) -> None:
        """Executes (or resumes/extends) the adversarial co-evolution training loop."""
        G = self.config.group_size
        resume_step, total_steps, resumed_from_dir = self._resolve_run_bounds()
        self.config.num_steps = total_steps

        console = Console()
        if resume_step >= total_steps:
            console.print(
                Panel(
                    f"[bold green]Checkpoint in [cyan]{resumed_from_dir}[/cyan] has already "
                    f"completed Step {resume_step}/{total_steps}.[/bold green]\n\n"
                    "To extend this run by an additional N steps (e.g., +200 steps), run:\n"
                    "  [bold yellow]docker compose run --rm --service-ports adversarial-experiment --extend-steps 200[/bold yellow]\n\n"
                    "Or specify a new total step target (e.g., 300 total steps):\n"
                    "  [bold yellow]docker compose run --rm --service-ports adversarial-experiment --num-steps 300[/bold yellow]\n\n"
                    "Or start a completely fresh experiment from Step 1:\n"
                    "  [bold yellow]docker compose run --rm --service-ports adversarial-experiment --fresh[/bold yellow]",
                    title="[bold magenta]Experiment Already Complete — Ready to Extend[/bold magenta]",
                    border_style="green",
                )
            )
            return

        self._install_signal_handlers()
        last_saved_step = resume_step

        with ExperimentDashboard(
            config=self.config,
            device=self.device,
            resume_step=resume_step,
            total_steps=total_steps,
        ) as dashboard:
            try:
                for step in range(resume_step + 1, total_steps + 1):
                    if self._stop_requested:
                        break

                    step_start = time.time()

                    # 1. Sample 1 concrete medical seed scenario and 1 benign health query
                    sample = self.dataset.sample_step(step)

                    # 2. Enable ONLY `prompt_adaptor` and generate a group of G defensive prompts
                    #    dynamically conditioned on the last k = 3 high-scoring Red attacks
                    prompt_batch = self.models.generate_defensive_preprompts(
                        group_size=G,
                        recent_attacks=self._recent_successful_attacks,
                        last_successful_attack=self._last_successful_attack,
                    )

                    # 3. Disable `prompt_adaptor`, enable `red_adaptor`, and generate G seed-anchored attacks
                    red_batch = self.models.generate_adversarial_attacks(
                        medical_seed=sample.medical_seed,
                        group_size=G,
                        use_lora=True,
                    )

                    # 4. Disable both adaptors to simulate frozen Model Blue:
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
                    flat_attack_logits = self.evaluator.score_medical_advice_logit(
                        blue_attack_responses
                    )
                    attack_logits_matrix = flat_attack_logits.view(G, G)
                    benign_utility_scores = self.evaluator.score_benign_utility(
                        blue_benign_responses, sample.benign_query
                    )

                    reward_dict = self.evaluator.compute_rewards(
                        attack_logits_matrix=attack_logits_matrix,
                        benign_utility_scores=benign_utility_scores,
                    )
                    red_rewards = reward_dict["red_rewards"]
                    prompt_defense_scores = reward_dict["prompt_defense_scores"]
                    prompt_utility_scores = reward_dict["prompt_utility_scores"]
                    prompt_composite_rewards = reward_dict["prompt_composite_rewards"]

                    # Record the highest-scoring Red attack from step t into the k = 3 rolling buffer
                    best_r_idx = int(torch.argmax(red_rewards).item())
                    best_attack = red_batch.texts[best_r_idx]
                    self._last_successful_attack = best_attack
                    self._recent_successful_attacks = (
                        self._recent_successful_attacks + [best_attack]
                    )[-3:]

                    # 6. Benchmark Non-LoRA Model Red & refresh sample dialogue showcase periodically
                    should_refresh_sample = (
                        step == 1
                        or step == resume_step + 1
                        or step % self.config.sample_refresh_interval == 0
                        or step == total_steps
                    )
                    dialogue_snapshot: Optional[DialogueSampleSnapshot] = None

                    nolora_mean_logit, nolora_attack_0, nolora_blue_0 = (
                        self._evaluate_nolora_baseline(
                            medical_seed=sample.medical_seed,
                            preprompts=prompt_batch.texts,
                        )
                    )
                    self._last_nolora_val_reward = nolora_mean_logit

                    if should_refresh_sample:
                        best_p_idx = int(torch.argmax(prompt_composite_rewards).item())
                        pair_flat_idx = best_p_idx * G + best_r_idx

                        dialogue_snapshot = DialogueSampleSnapshot(
                            step=step,
                            medical_seed=sample.medical_seed,
                            benign_query=sample.benign_query,
                            defensive_preprompt=prompt_batch.texts[best_p_idx],
                            red_lora_attack=red_batch.texts[best_r_idx],
                            blue_lora_response=blue_attack_responses[pair_flat_idx],
                            lora_medical_logit=float(
                                attack_logits_matrix[best_p_idx, best_r_idx].item()
                            ),
                            red_nolora_attack=nolora_attack_0,
                            blue_nolora_response=nolora_blue_0,
                            nolora_medical_logit=nolora_mean_logit,
                            blue_benign_response=blue_benign_responses[best_p_idx],
                            benign_utility_score=float(
                                prompt_utility_scores[best_p_idx].item()
                            ),
                        )

                    # 7. Backpropagate GRPO policy updates:
                    red_stats = self.grpo.update_red_adaptor(
                        red_batch=red_batch,
                        red_rewards=red_rewards,
                    )
                    prompt_stats = self.grpo.update_prompt_adaptor(
                        prompt_batch=prompt_batch,
                        prompt_composite_rewards=prompt_composite_rewards,
                    )

                    # 8. Record metrics and refresh live dashboard / TensorBoard
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

                    # 9. Save rolling `latest` checkpoint on sample intervals / stop request,
                    # and save numbered `step_XXXX` snapshots on `checkpoint_interval` or completion.
                    save_numbered = (
                        step % self.config.checkpoint_interval == 0
                        or step == total_steps
                    )
                    save_latest = (
                        save_numbered
                        or step % self.config.sample_refresh_interval == 0
                        or self._stop_requested
                    )
                    if save_latest:
                        self.save_checkpoint(
                            step=step,
                            total_steps=total_steps,
                            save_numbered=save_numbered,
                        )
                        last_saved_step = step

            except KeyboardInterrupt:
                self._stop_requested = True

        if self._stop_requested:
            console.print(
                f"\n[bold yellow]Training paused cleanly at Step {last_saved_step}/{total_steps}. "
                f"Latest checkpoint saved at {self.config.latest_checkpoint_dir}. "
                "Re-run the same command to resume seamlessly.[/bold yellow]"
            )
