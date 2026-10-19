# Zero-Sum Adversarial Co-Evolution of Medical Guardrails (`prompt_adapter` vs. `red_adapter`)

This repository implements a reproducible **Group Relative Policy Optimization (GRPO)** adversarial co-evolution experiment designed to discover resilient defensive system preprompts against medical prompt injections on a single **24GB NVIDIA TITAN RTX** GPU.

---

## 1. Architecture Overview

To fit two learning agents, a target assistant, and a continuous reward evaluator concurrently within **24GB VRAM** on a Turing-architecture GPU (NVIDIA TITAN RTX, Compute Capability 7.5), the pipeline uses a **shared-weight multi-adapter architecture** paired with a **deterministic encoder sidecar evaluator**:

| Component / Role | Implementation | Precision & Attention | Role in Single-Round Interaction |
| :--- | :--- | :--- | :--- |
| **Frozen Base Model (`Model Blue`)** | `Qwen/Qwen2.5-1.5B-Instruct` (both LoRA adapters disabled) | `FP16` + PyTorch Native `SDPA` | Acts as the target standard helpful chatbot wrapped by the defensive preprompt. |
| **Defender (`prompt_adapter`)** | PEFT LoRA Adapter (`r=16, alpha=32`) on Base Model | Master weights `FP32`, forward `FP16` autocast | Generates batches of $G$ defensive system preprompts to wrap `Model Blue`. |
| **Attacker (`red_adapter`)** | PEFT LoRA Adapter (`r=16, alpha=32`) on Base Model | Master weights `FP32`, forward `FP16` autocast | Generates batches of $G$ single-turn medical prompt injections anchored to a concrete medical seed. |
| **Validation Attacker (`Non-LoRA Red`)** | Frozen Base Model (both LoRA adapters disabled) | `FP16` + PyTorch Native `SDPA` | Benchmarks static (untrained) red-team attacks over time as `prompt_adapter` hardens `Model Blue`. |
| **Sidecar Evaluator (`DeBERTa-v3`)** | `microsoft/deberta-v3-small` fine-tuned for medical intent | `FP32` logits $\rightarrow$ `softmax` | Outputs continuous probability $P(\text{medical-advice}) \in (0.0, 1.0)$, immune to generative judge prompt injections. |

### Why Turing-Specific Hardware Flags Matter (NVIDIA TITAN RTX)
- **FP16 + Mixed-Precision `GradScaler`**: Turing GPUs lack native `bfloat16` tensor core support. The frozen 1.5B base model is held in `torch.float16`, while trainable LoRA adapter matrices (`lora_A`, `lora_B`) are maintained in `torch.float32` under `torch.amp.autocast(dtype=torch.float16)` and updated through `torch.amp.GradScaler` (`lr = 2.5e-5`) to prevent FP16 gradient underflow.
- **PyTorch Native `SDPA` (`attn_implementation="sdpa"`)**: FlashAttention-2 requires Ampere (`SM 8.0+`) or newer. We explicitly disable Flash SDP kernel dispatch on CUDA and enable PyTorch's memory-efficient and math SDPA backends.

---

## 2. Fixed Token Ceilings (Single-Round Interaction)

We enforce a strict single-round interaction (`Model Red Prompt -> Model Blue Response`) with explicit token ceilings for every role (`src/config.py`):

| Role / Stage | Config Field | Token Ceiling | Rationale |
| :--- | :--- | :---: | :--- |
| **Medical Seed / Benign Query** | `seed_max_tokens` | `64` | Concise 1–2 sentence clinical scenario or benign health question. |
| **Defensive Preprompt (`prompt_adapter`)** | `preprompt_max_new_tokens` | `96` (min `16`) | Sufficient to express clear refusal rules for clinical advice while preserving benign helpfulness, without flooding context. |
| **Adversarial Attack (`red_adapter`)** | `attack_max_new_tokens` | `128` (min `16`) | Enough to craft persuasive single-turn jailbreak framings (clinical vignettes, roleplay, urgency) anchored to the seed. |
| **Target Chatbot (`Model Blue`) Input** | `blue_max_input_tokens` | `384` | Safely accommodates `System Preprompt (96) + User Attack (128) + Chat Template`. |
| **Target Chatbot (`Model Blue`) Output** | `blue_max_new_tokens` | `160` (min `16`) | Captures either a refusal/disclaimer, leaked medical dosages/diagnoses, or a complete helpful benign paragraph. |
| **Sidecar Evaluator (`DeBERTa-v3`)** | `evaluator_max_length` | `256` | Strictly exceeds `blue_max_new_tokens` (`160`) so 100% of `Model Blue`'s response is evaluated without truncation. |

---

## 3. Interleaved GRPO Training Loop & Reward Formulation

At each step $t$:
1. **Scenario Sampling**: Sample 1 concrete medical seed $s_{\text{med}}$ (from 80 clinical scenarios) and 1 benign health/science query $q_{\text{benign}}$ (from 80 parallel benign queries) in `src/dataset.py`.
2. **Group Generation**:
   - Enable `prompt_adapter` $\rightarrow$ generate $G$ defensive preprompts $\{p_1, \dots, p_G\}$.
   - Enable `red_adapter` $\rightarrow$ feed $s_{\text{med}}$ and generate $G$ adversarial attacks $\{a_1, \dots, a_G\}$.
   - Disable both adapters (`Model Blue`) $\rightarrow$ generate single-turn responses $y_{i,j}^{\text{adv}}$ for all $G \times G$ pairs $(p_i, a_j)$ and benign responses $y_i^{\text{benign}}$ for all $G$ pairs $(p_i, q_{\text{benign}})$.
3. **Continuous Rewards**:
   - **DeBERTa-v3 Attack Probability**: $$S_{i,j} = P_{\text{DeBERTa}}(\text{MEDICAL-ADVICE} \mid y^{\text{adv}}_{i,j}) \in (0, 1)$$.
   - **Red Model Reward** (for attack $a_j$):
     $$r_j^{\text{red}} = \frac{1}{G} \sum_{i=1}^{G} S_{i,j}$$
   - **Prompt Model Composite Reward** (for preprompt $p_i$, preventing defensive lobotomization):
     $$r_i^{\text{prompt}} = w_{\text{def}} \cdot \left(1.0 - \frac{1}{G}\sum_{j=1}^{G} S_{i,j}\right) + w_{\text{util}} \cdot U_{\text{benign}}\left(y_i^{\text{benign}}\right)$$
     where $U_{\text{benign}} \in [0, 1]$ rewards response length adequacy, lexical fluency/diversity, and absence of false-positive refusals on harmless queries.
4. **Group Relative Advantage & Policy Updates (`src/grpo.py`)**:
   - Group advantages are normalized over each group of size $G$:
     $$A_k = \frac{r_k - \mu_r}{\sigma_r + \epsilon}$$
   - `red_adapter` is updated first via GRPO with KL regularization against the frozen base model, followed by swapping to `prompt_adapter` and updating its weights with its own optimizer and `GradScaler`.

---

## 4. Running the Experiment

### Option A: Reproducible Docker Container

The Docker setup mounts `./experimental_results` on the host to `/app/experimental_results` inside the container and runs both the **Rich Live Terminal Dashboard** and a background **TensorBoard server** on port `6006`.

This can be deployed using **Docker Compose**:
```bash
docker compose run --rm --service-ports adversarial-experiment
```

Or manually as follows:

```bash
# 1. Build the Docker image
docker build -t adversarial-llm-output:latest .

# 2. Run interactively with GPU access and mounted ./experimental_results folder
mkdir -p experimental_results
docker run --rm -it --gpus all \
  -p 6006:6006 \
  -v "$(pwd)/experimental_results:/app/experimental_results:Z" \
  adversarial-llm-output:latest \
  --num-steps 100 --group-size 4 --sample-interval 5
```

While the container is running, open **`http://localhost:6006`** in your browser to view live TensorBoard curves and text tables.

### Option B: Running Directly with Pixi (MacBook M1 Pro or Linux Host)

```bash
# Install locked dependencies for your platform (osx-arm64 or linux-64)
pixi install

# Quick 2-step smoke test to verify pipeline end-to-end
pixi run smoke-test

# Full training run
pixi run train

# Launch TensorBoard locally
pixi run tensorboard
```

---

## 5. Mounted Outputs in `./experimental_results/`

All outputs are written to `./experimental_results/`:
- `experiment_config.json` — Full hyperparameter and token-ceiling snapshot.
- `metrics.jsonl` — Step-by-step telemetry (LoRA Red reward, Non-LoRA Red validation reward, Prompt defense score, Benign utility score, GRPO losses, KL divergence, VRAM usage).
- `latest_dialogue_sample.md` & `latest_dialogue_sample.json` — Most recent refreshed comparison of `(Red w/ LoRA -> Blue)` vs. `(Red w/o LoRA -> Blue)` and benign query handling.
- `tensorboard/` — TensorBoard event logs for all scalar curves and text dialogues.
- `checkpoints/step_XXXX/` — Saved PEFT LoRA weights for `prompt_adapter` and `red_adapter`.
- `deberta_medical_evaluator/` — Calibrated DeBERTa-v3 medical intent classifier checkpoint.

