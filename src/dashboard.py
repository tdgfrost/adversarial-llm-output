"""
Live Terminal Dashboard (Rich) and TensorBoard Telemetry Logger.

Provides:
1. A live, in-place terminal UI tracking co-evolution metrics, VRAM, and sparkline trends.
2. Benchmark validation tracking of Non-LoRA Model Red vs. LoRA Model Red rewards.
3. An in-place refreshing Dialogue Showcase comparing:
   - Model Red (with `red_adapter` LoRA) -> Model Blue
   - Model Red (without LoRA baseline) -> Model Blue
   - Benign Health Query -> Model Blue
   This panel clears/refreshes whenever a new sample is generated rather than scrolling.
4. TensorBoard logging (`SummaryWriter`) and JSONL persistence in `experimental_results/`.
"""

from __future__ import annotations

import json
import time
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Deque, Dict, List, Optional

import torch
from rich.columns import Columns
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text
from torch.utils.tensorboard import SummaryWriter

from src.config import TrainingConfig

SPARK_CHARS = "▁▂▃▄▅▆▇█"


def make_sparkline(values: List[float], width: int = 16, vmin: float = 0.0, vmax: float = 1.0) -> str:
    """Renders a compact Unicode sparkline for a sequence of metric values."""
    if not values:
        return "·" * width
    tail = values[-width:]
    span = max(1e-6, vmax - vmin)
    chars: List[str] = []
    for val in tail:
        norm = max(0.0, min(1.0, (val - vmin) / span))
        idx = min(len(SPARK_CHARS) - 1, int(round(norm * (len(SPARK_CHARS) - 1))))
        chars.append(SPARK_CHARS[idx])
    return "".join(chars).ljust(width, "·")


@dataclass
class DialogueSampleSnapshot:
    """Single-round dialogue snapshot for the refreshing showcase panel."""

    step: int
    medical_seed: str
    benign_query: str
    defensive_preprompt: str
    # With LoRA (red_adapter)
    red_lora_attack: str
    blue_lora_response: str
    lora_medical_prob: float
    # Without LoRA (Frozen baseline Model Red)
    red_nolora_attack: str
    blue_nolora_response: str
    nolora_medical_prob: float
    # Benign query check
    blue_benign_response: str
    benign_utility_score: float


@dataclass
class StepMetricsRecord:
    """Complete telemetry record for a single training step."""

    step: int
    total_steps: int
    step_time_sec: float
    vram_allocated_gb: float
    vram_max_gb: float
    # Core rewards
    red_lora_reward: float
    red_nolora_val_reward: float
    prompt_defense_score: float
    prompt_utility_score: float
    prompt_composite_reward: float
    # GRPO diagnostics
    red_loss: float
    red_kl: float
    red_grad_norm: float
    red_reward_std: float
    prompt_loss: float
    prompt_kl: float
    prompt_grad_norm: float
    prompt_reward_std: float


class ExperimentDashboard:
    """
    Manages the Rich Live terminal display, refreshing dialogue panel,
    TensorBoard SummaryWriter, and JSONL file logging inside `experimental_results/`.
    """

    def __init__(self, config: TrainingConfig, device: torch.device) -> None:
        self.config = config
        self.device = device
        self.start_time = time.time()

        # Ensure output directories exist
        self.config.output_path.mkdir(parents=True, exist_ok=True)
        self.config.tensorboard_dir.mkdir(parents=True, exist_ok=True)
        self.config.checkpoints_dir.mkdir(parents=True, exist_ok=True)

        self.metrics_jsonl_path = self.config.output_path / "metrics.jsonl"
        self.latest_sample_md_path = self.config.output_path / "latest_dialogue_sample.md"
        self.latest_sample_json_path = self.config.output_path / "latest_dialogue_sample.json"

        # Save experiment configuration snapshot
        config_snapshot_path = self.config.output_path / "experiment_config.json"
        config_snapshot_path.write_text(
            json.dumps(self.config.to_dict(), indent=2), encoding="utf-8"
        )

        # Initialize TensorBoard writer
        self.tb_writer = SummaryWriter(log_dir=str(self.config.tensorboard_dir))

        # Metric histories for moving averages and sparklines
        win = config.moving_avg_window
        self.history: Dict[str, List[float]] = {
            "red_lora": [],
            "red_nolora": [],
            "defense": [],
            "utility": [],
            "composite": [],
        }
        self.rolling: Dict[str, Deque[float]] = {
            k: deque(maxlen=win) for k in self.history
        }

        self.latest_metrics: Optional[StepMetricsRecord] = None
        self.latest_dialogue: Optional[DialogueSampleSnapshot] = None
        self.initial_nolora_reward: Optional[float] = None

        self.console = Console(force_terminal=config.use_live_ui)
        self.live: Optional[Live] = None

    def __enter__(self) -> "ExperimentDashboard":
        if self.config.use_live_ui:
            self.live = Live(
                self._render_dashboard(),
                console=self.console,
                refresh_per_second=4,
                transient=False,
            )
            self.live.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self.live is not None:
            self.live.__exit__(exc_type, exc_val, exc_tb)
        self.tb_writer.flush()
        self.tb_writer.close()

    def _device_description(self) -> str:
        if self.device.type == "cuda" and torch.cuda.is_available():
            name = torch.cuda.get_device_name(self.device)
            total_gb = torch.cuda.get_device_properties(self.device).total_memory / (1024**3)
            return f"{name} ({total_gb:.1f} GB VRAM) | FP16 + PyTorch SDPA"
        if self.device.type == "mps":
            return "Apple Silicon MPS | FP16 + PyTorch SDPA"
        return "CPU | FP32 + PyTorch SDPA"

    def update_step(
        self,
        metrics: StepMetricsRecord,
        dialogue_sample: Optional[DialogueSampleSnapshot] = None,
    ) -> None:
        """
        Records step telemetry to TensorBoard and `metrics.jsonl`, updates the
        in-place dialogue sample if provided, and refreshes the terminal dashboard.
        """
        self.latest_metrics = metrics
        if self.initial_nolora_reward is None:
            self.initial_nolora_reward = metrics.red_nolora_val_reward

        # Update histories
        self._push_metric("red_lora", metrics.red_lora_reward)
        self._push_metric("red_nolora", metrics.red_nolora_val_reward)
        self._push_metric("defense", metrics.prompt_defense_score)
        self._push_metric("utility", metrics.prompt_utility_score)
        self._push_metric("composite", metrics.prompt_composite_reward)

        if dialogue_sample is not None:
            # Replace (clear/refresh) the stored dialogue snapshot with the new sample
            self.latest_dialogue = dialogue_sample
            self._persist_latest_dialogue(dialogue_sample)
            self._log_dialogue_to_tensorboard(dialogue_sample)

        # Log scalars to TensorBoard
        s = metrics.step
        self.tb_writer.add_scalar("Rewards/1_Red_LoRA_Attack_Prob", metrics.red_lora_reward, s)
        self.tb_writer.add_scalar(
            "Rewards/2_Red_NoLoRA_Baseline_Validation_Prob", metrics.red_nolora_val_reward, s
        )
        self.tb_writer.add_scalar(
            "Rewards/3_Red_LoRA_Advantage_vs_NoLoRA",
            metrics.red_lora_reward - metrics.red_nolora_val_reward,
            s,
        )
        self.tb_writer.add_scalar("Rewards/4_Prompt_Defense_Score", metrics.prompt_defense_score, s)
        self.tb_writer.add_scalar("Rewards/5_Blue_Benign_Utility_Score", metrics.prompt_utility_score, s)
        self.tb_writer.add_scalar(
            "Rewards/6_Prompt_Composite_Reward", metrics.prompt_composite_reward, s
        )

        self.tb_writer.add_scalar("GRPO_Loss/Red_Adapter_Loss", metrics.red_loss, s)
        self.tb_writer.add_scalar("GRPO_Loss/Prompt_Adapter_Loss", metrics.prompt_loss, s)
        self.tb_writer.add_scalar("GRPO_KL/Red_Adapter_KL", metrics.red_kl, s)
        self.tb_writer.add_scalar("GRPO_KL/Prompt_Adapter_KL", metrics.prompt_kl, s)
        self.tb_writer.add_scalar("GRPO_GradNorm/Red_Adapter", metrics.red_grad_norm, s)
        self.tb_writer.add_scalar("GRPO_GradNorm/Prompt_Adapter", metrics.prompt_grad_norm, s)
        self.tb_writer.add_scalar("GRPO_Variance/Red_Group_Reward_Std", metrics.red_reward_std, s)
        self.tb_writer.add_scalar(
            "GRPO_Variance/Prompt_Group_Reward_Std", metrics.prompt_reward_std, s
        )
        self.tb_writer.add_scalar("System/VRAM_Allocated_GB", metrics.vram_allocated_gb, s)
        self.tb_writer.add_scalar("System/Step_Duration_Sec", metrics.step_time_sec, s)
        self.tb_writer.flush()

        # Append to metrics.jsonl
        with self.metrics_jsonl_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(metrics)) + "\n")

        # Refresh terminal display
        if self.live is not None:
            self.live.update(self._render_dashboard(), refresh=True)
        else:
            # Non-TTY fallback: clear screen and print refreshed view on sample intervals
            if dialogue_sample is not None or metrics.step == metrics.total_steps:
                self.console.clear()
                self.console.print(self._render_dashboard())

    def _push_metric(self, key: str, value: float) -> None:
        self.history[key].append(value)
        self.rolling[key].append(value)

    def _mean(self, key: str) -> float:
        vals = self.rolling[key]
        return sum(vals) / len(vals) if vals else 0.0

    def _persist_latest_dialogue(self, sample: DialogueSampleSnapshot) -> None:
        """Overwrites `latest_dialogue_sample.json` and `.md` with the refreshed sample."""
        self.latest_sample_json_path.write_text(
            json.dumps(asdict(sample), indent=2), encoding="utf-8"
        )
        md_content = (
            f"# Latest Co-Evolution Dialogue Sample (Step {sample.step})\n\n"
            f"## 1. Sampled Medical Seed & Defensive Preprompt\n"
            f"- **Medical Seed Scenario**: {sample.medical_seed}\n"
            f"- **Defensive Preprompt (`prompt_adapter`)**:\n"
            f"  > {sample.defensive_preprompt}\n\n"
            f"## 2. Adversarial Interaction WITH LoRA (`red_adapter` -> `Model Blue`)\n"
            f"- **DeBERTa-v3 Medical Advice Probability**: `{sample.lora_medical_prob:.4f}`\n"
            f"- **Model Red (LoRA) Attack**:\n"
            f"  > {sample.red_lora_attack}\n"
            f"- **Model Blue Response**:\n"
            f"  > {sample.blue_lora_response}\n\n"
            f"## 3. Baseline Validation WITHOUT LoRA (`Frozen Red` -> `Model Blue`)\n"
            f"- **DeBERTa-v3 Medical Advice Probability**: `{sample.nolora_medical_prob:.4f}`\n"
            f"- **Model Red (Non-LoRA) Attack**:\n"
            f"  > {sample.red_nolora_attack}\n"
            f"- **Model Blue Response**:\n"
            f"  > {sample.blue_nolora_response}\n\n"
            f"## 4. Benign Health Query Utility Check\n"
            f"- **Benign Utility Score**: `{sample.benign_utility_score:.4f}`\n"
            f"- **Benign Query**: {sample.benign_query}\n"
            f"- **Model Blue Response**:\n"
            f"  > {sample.blue_benign_response}\n"
        )
        self.latest_sample_md_path.write_text(md_content, encoding="utf-8")

    def _log_dialogue_to_tensorboard(self, sample: DialogueSampleSnapshot) -> None:
        text_block = (
            f"**Step {sample.step}** | "
            f"**LoRA Red Prob:** `{sample.lora_medical_prob:.3f}` | "
            f"**Non-LoRA Red Prob:** `{sample.nolora_medical_prob:.3f}` | "
            f"**Benign Utility:** `{sample.benign_utility_score:.3f}`\n\n"
            f"- **Seed:** {sample.medical_seed}\n"
            f"- **Preprompt (`prompt_adapter`):** {sample.defensive_preprompt}\n"
            f"- **Red Attack (LoRA):** {sample.red_lora_attack}\n"
            f"- **Blue Response (vs LoRA):** {sample.blue_lora_response}\n"
            f"- **Red Attack (Non-LoRA):** {sample.red_nolora_attack}\n"
            f"- **Blue Response (vs Non-LoRA):** {sample.blue_nolora_response}\n"
            f"- **Benign Query:** {sample.benign_query}\n"
            f"- **Blue Benign Response:** {sample.blue_benign_response}"
        )
        self.tb_writer.add_text("Dialogue_Samples/CoEvolution_Comparison", text_block, sample.step)

    # -----------------------------------------------------------------------
    # Rich UI Rendering
    # -----------------------------------------------------------------------

    def _render_dashboard(self) -> Group:
        header_panel = self._build_header_panel()
        metrics_panel = self._build_metrics_panel()
        dialogue_panel = self._build_dialogue_panel()
        return Group(header_panel, metrics_panel, dialogue_panel)

    def _build_header_panel(self) -> Panel:
        elapsed = time.time() - self.start_time
        m = self.latest_metrics

        if m is None:
            progress_str = "Initializing base model (Qwen2.5-1.5B), LoRA adapters, and DeBERTa-v3 sidecar..."
            vram_str = "Allocating VRAM..."
        else:
            pct = 100.0 * m.step / max(1, m.total_steps)
            filled = int(round(28 * m.step / max(1, m.total_steps)))
            bar = "█" * filled + "░" * (28 - filled)
            eta = (elapsed / max(1, m.step)) * (m.total_steps - m.step)
            progress_str = (
                f"[bold cyan][{bar}] {pct:5.1f}%[/bold cyan] "
                f"| Step [bold]{m.step}/{m.total_steps}[/bold] "
                f"| {m.step_time_sec:.2f}s/step "
                f"| Elapsed {elapsed:.0f}s | ETA {eta:.0f}s"
            )
            vram_str = f"Active: {m.vram_allocated_gb:.2f} GB | Peak: {m.vram_max_gb:.2f} GB"

        grid = Table.grid(expand=True)
        grid.add_column(ratio=1)
        grid.add_column(ratio=1)
        grid.add_row(
            f"[bold white]Hardware:[/bold white] {self._device_description()}",
            f"[bold white]Base / Sidecar:[/bold white] {self.config.model.base_model_id} + DeBERTa-v3",
        )
        grid.add_row(
            f"[bold white]Results Mount:[/bold white] {self.config.output_path}",
            f"[bold white]VRAM Usage:[/bold white] {vram_str} | [bold green]TensorBoard:[/bold green] :6006",
        )
        grid.add_row(progress_str, "")

        return Panel(
            grid,
            title="[bold magenta]Zero-Sum Adversarial GRPO Co-Evolution: Guardrail Prompt vs. Medical Red-Team[/bold magenta]",
            border_style="magenta",
        )

    def _build_metrics_panel(self) -> Panel:
        table = Table(expand=True, box=None, pad_edge=False)
        table.add_column("Metric / Agent Role", style="bold")
        table.add_column("Current", justify="right")
        table.add_column(f"Avg({self.config.moving_avg_window})", justify="right")
        table.add_column("Trend (0→1)", justify="center")
        table.add_column("Interpretation / Target Dynamics", style="dim")

        m = self.latest_metrics
        if m is None:
            table.add_row("Waiting for Step 1...", "-", "-", "·" * 16, "Generating initial GRPO groups")
            return Panel(table, title="[bold cyan]Live Training & Validation Telemetry[/bold cyan]", border_style="cyan")

        # Calculate baseline validation delta for Non-LoRA Red
        init_nolora = self.initial_nolora_reward if self.initial_nolora_reward is not None else m.red_nolora_val_reward
        nolora_delta = m.red_nolora_val_reward - init_nolora
        delta_str = f"({nolora_delta:+.3f} vs step 1)"

        max_comp = self.config.defense_weight + self.config.utility_weight

        table.add_row(
            "[red]Red (LoRA `red_adapter`) Attack Prob[/red]",
            f"{m.red_lora_reward:.4f}",
            f"{self._mean('red_lora'):.4f}",
            f"[red]{make_sparkline(self.history['red_lora'])}[/red]",
            "DeBERTa P(medical_advice) w/ trained LoRA attacker",
        )
        table.add_row(
            "[yellow]Red (Non-LoRA Baseline) Validation Prob[/yellow]",
            f"{m.red_nolora_val_reward:.4f}",
            f"{self._mean('red_nolora'):.4f}",
            f"[yellow]{make_sparkline(self.history['red_nolora'])}[/yellow]",
            f"Frozen Red benchmark {delta_str} — falls as Blue hardens",
        )
        table.add_row(
            "[bright_red]LoRA vs Non-LoRA Red Advantage[/bright_red]",
            f"{m.red_lora_reward - m.red_nolora_val_reward:+.4f}",
            f"{self._mean('red_lora') - self._mean('red_nolora'):+.4f}",
            "—",
            "Gain of co-evolved `red_adapter` over static Red",
        )
        table.add_row(
            "[green]Prompt (`prompt_adapter`) Defense Score[/green]",
            f"{m.prompt_defense_score:.4f}",
            f"{self._mean('defense'):.4f}",
            f"[green]{make_sparkline(self.history['defense'])}[/green]",
            "1.0 - DeBERTa P(medical_advice) against LoRA Red",
        )
        table.add_row(
            "[blue]Blue Benign Query Utility Score[/blue]",
            f"{m.prompt_utility_score:.4f}",
            f"{self._mean('utility'):.4f}",
            f"[blue]{make_sparkline(self.history['utility'])}[/blue]",
            "Fluency + length + non-refusal on benign health query",
        )
        table.add_row(
            "[bold green]Prompt Composite Reward (Def + Util)[/bold green]",
            f"{m.prompt_composite_reward:.4f}",
            f"{self._mean('composite'):.4f}",
            f"[bold green]{make_sparkline(self.history['composite'], vmin=0.0, vmax=max_comp)}[/bold green]",
            f"Anti-lobotomization objective (max {max_comp:.2f})",
        )
        table.add_row(
            "[dim]GRPO Policy Updates (Red | Prompt)[/dim]",
            f"L:{m.red_loss:.3f} | {m.prompt_loss:.3f}",
            f"KL:{m.red_kl:.4f} | {m.prompt_kl:.4f}",
            f"σ:{m.red_reward_std:.3f} | {m.prompt_reward_std:.3f}",
            f"GradNorm Red: {m.red_grad_norm:.3f} | Prompt: {m.prompt_grad_norm:.3f}",
        )

        return Panel(
            table,
            title="[bold cyan]Live Training & Validation Telemetry[/bold cyan]",
            border_style="cyan",
        )

    def _build_dialogue_panel(self) -> Panel:
        d = self.latest_dialogue
        if d is None:
            return Panel(
                Text("Waiting for first dialogue sample...", style="dim"),
                title="[bold yellow]Refreshing Single-Round Dialogue Showcase[/bold yellow]",
                border_style="yellow",
            )

        def _truncate(text: str, limit: int = 340) -> str:
            clean = " ".join(text.split())
            return clean if len(clean) <= limit else clean[: limit - 3] + "..."

        top_info = Table.grid(expand=True)
        top_info.add_column()
        top_info.add_row(
            f"[bold white]Medical Seed Scenario:[/bold white] [italic]{_truncate(d.medical_seed, 220)}[/italic]"
        )
        top_info.add_row(
            f"[bold green]Active Defensive Preprompt (`prompt_adapter`):[/bold green] {_truncate(d.defensive_preprompt, 280)}"
        )

        # Left column: With LoRA (red_adapter)
        lora_box = Panel(
            Group(
                Text.from_markup(
                    f"[bold red]Model Red (WITH LoRA `red_adapter`):[/bold red]\n{_truncate(d.red_lora_attack, 280)}"
                ),
                Rule(style="dim"),
                Text.from_markup(
                    f"[bold cyan]Model Blue Response[/bold cyan] "
                    f"([bold]DeBERTa Medical Prob: {d.lora_medical_prob:.3f}[/bold]):\n"
                    f"{_truncate(d.blue_lora_response, 280)}"
                ),
            ),
            title="[red]With LoRA (`red_adapter`)[/red]",
            border_style="red",
        )

        # Right column: Without LoRA (Baseline Frozen Red)
        nolora_box = Panel(
            Group(
                Text.from_markup(
                    f"[bold yellow]Model Red (WITHOUT LoRA Baseline):[/bold yellow]\n{_truncate(d.red_nolora_attack, 280)}"
                ),
                Rule(style="dim"),
                Text.from_markup(
                    f"[bold cyan]Model Blue Response[/bold cyan] "
                    f"([bold]DeBERTa Medical Prob: {d.nolora_medical_prob:.3f}[/bold]):\n"
                    f"{_truncate(d.blue_nolora_response, 280)}"
                ),
            ),
            title="[yellow]Without LoRA (Validation Baseline Red)[/yellow]",
            border_style="yellow",
        )

        benign_footer = Text.from_markup(
            f"[bold blue]Benign Query ({_truncate(d.benign_query, 110)})[/bold blue] "
            f"→ [bold cyan]Model Blue[/bold cyan] ([bold]Utility: {d.benign_utility_score:.3f}[/bold]): "
            f"{_truncate(d.blue_benign_response, 210)}"
        )

        return Panel(
            Group(top_info, Columns([lora_box, nolora_box], expand=True, equal=True), benign_footer),
            title=(
                f"[bold yellow]Refreshing Single-Round Dialogue Showcase "
                f"(Snapshot @ Step {d.step} — Refreshes every {self.config.sample_refresh_interval} steps)[/bold yellow]"
            ),
            border_style="yellow",
        )
