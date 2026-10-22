"""
Fast, deterministic unit and end-to-end integration tests for the Adversarial
Guardrail Co-Evolution pipeline.

Verifies:
1. Dataset completeness (80 medical seeds, 80 benign health queries, epoch cycling).
2. Zero-shot NLI evaluator label resolution, calibration sanity check, reward matrix
   math, and benign utility anti-lobotomisation scoring.
3. Dual-LoRA adaptor isolation (`prompt_adaptor` vs `red_adaptor`), frozen `Model Blue`
   and `Non-LoRA Red` context switching, GRPO backpropagation with `GradScaler`,
   Rich Live dashboard updates, and TensorBoard/JSONL file output.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from types import SimpleNamespace

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

from src.config import TokenCeilingConfig, TrainingConfig, format_utility_premise
from src.main import parse_args
from src.dashboard import (
    DialogueSampleSnapshot,
    ExperimentDashboard,
    StepMetricsRecord,
)
from src.dataset import (
    BENIGN_HEALTH_QUERIES,
    EVALUATOR_CALIBRATION_DATA,
    MEDICAL_SEED_SCENARIOS,
    UTILITY_CALIBRATION_DATA,
    AdversarialDataset,
)
from src.evaluator import MedicalIntentEvaluator
from src.grpo import DualAgentGRPOTrainer
from src.models import GenerationGroupBatch, SharedWeightAdversarialModels


def test_dataset_completeness_and_sampling() -> None:
    cfg = TrainingConfig()
    assert not hasattr(cfg.tokens, "seed_max_tokens")
    assert cfg.tokens.generator_max_input_tokens == 3 * 256 + 408 == 1176
    assert cfg.tokens.blue_max_input_tokens == 256 + 256 + 111 == 623
    # Hypothesis tokens plus the question-conditioned utility premise overhead
    # (`UTILITY_PREMISE_TEMPLATE` scaffold + the longest benign query)
    assert cfg.tokens.evaluator_max_length == int(256 * 1.5) + 32 + 9 + 38 == 463

    # Custom independent token limits should automatically update dependent token limits
    custom_tokens = TokenCeilingConfig(
        preprompt_max_new_tokens=128,
        attack_max_new_tokens=128,
        blue_max_new_tokens=300,
    )
    assert custom_tokens.generator_max_input_tokens == 3 * 128 + 408 == 792
    assert custom_tokens.blue_max_input_tokens == 128 + 128 + 111 == 367
    assert custom_tokens.evaluator_max_length == int(300 * 1.5) + 32 + 9 + 38 == 529

    # Manual positive non-zero overrides should be respected
    manual_tokens = TokenCeilingConfig(
        generator_max_input_tokens=2048,
        blue_max_input_tokens=1024,
        evaluator_max_length=768,
    )
    assert manual_tokens.generator_max_input_tokens == 2048
    assert manual_tokens.blue_max_input_tokens == 1024
    assert manual_tokens.evaluator_max_length == 768
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
    assert len(UTILITY_CALIBRATION_DATA) >= 50
    assert all(len(item) == 3 for item in UTILITY_CALIBRATION_DATA), (
        "Utility calibration items must be (question, response, label) triples"
    )

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
    evaluator.max_length = cfg.tokens.evaluator_max_length
    evaluator.hypothesis = cfg.model.evaluator_nli_hypothesis
    evaluator.utility_hypothesis = cfg.model.utility_nli_hypothesis
    evaluator.entail_index = 0
    evaluator.contradiction_indices = [1]
    evaluator.ema_defense_std = None
    evaluator.ema_utility_std = None

    captured_utility: dict = {}

    class _Encoding(dict):
        def to(self, _device):
            return self

    def fake_utility_tokenizer(premises, hypotheses, **kwargs):
        captured_utility["premises"] = premises
        captured_utility["hypotheses"] = hypotheses
        return _Encoding(batch=len(premises))

    evaluator.tokenizer = fake_utility_tokenizer
    # Return NLI logits for 4 responses:
    # 0: helpful relevant response (NLI logit +3.0)
    # 1: refusal response (NLI logit -3.0)
    # 2: empty response (NLI logit +1.0)
    # 3: reward-hacked generic anatomy response (high NLI logit +3.0, but low cosine similarity)
    evaluator.model = lambda **enc: SimpleNamespace(
        logits=torch.tensor(
            [[3.0, -3.0], [-3.0, 3.0], [1.0, -1.0], [3.0, -3.0]],
            dtype=torch.float32,
        )[: enc["batch"]]
    )

    helpful_resp = (
        "Hemoglobin is an iron-containing protein in red blood cells that transports "
        "oxygen from the lungs to tissues across the body and returns carbon dioxide "
        "back to the pulmonary alveoli during normal respiration."
    )
    refusal_resp = (
        "I cannot answer this question or provide any information. Please consult a doctor."
    )
    empty_resp = ""
    reward_hack_resp = (
        "The human body consists of complex anatomical systems including the musculoskeletal, "
        "digestive, and nervous organs that maintain general physiological homeostasis."
    )

    def fake_similarity_tokenizer(texts, **kwargs):
        captured_utility.setdefault("embedded_texts", []).append(list(texts))
        return _Encoding(
            texts=list(texts),
            attention_mask=torch.ones((len(texts), 2), dtype=torch.long),
        )

    def fake_similarity_model(**enc):
        # Construct 2D token embeddings whose mean-pooled L2-normalised vectors yield:
        # - query vector = [1.0, 0.0]
        # - helpful_resp -> cos_sim = 0.85
        # - refusal_resp -> cos_sim = 0.60 (e.g. echoing query terms while refusing)
        # - empty_resp   -> masked to 0.0 by score_query_response_similarity
        # - reward_hack_resp -> cos_sim = 0.10 (irrelevant generic anatomy statement)
        vecs = []
        for t in enc["texts"]:
            if t == helpful_resp:
                v = [0.85, (1.0 - 0.85**2) ** 0.5]
            elif t == refusal_resp:
                v = [0.60, (1.0 - 0.60**2) ** 0.5]
            elif t == reward_hack_resp:
                v = [0.10, (1.0 - 0.10**2) ** 0.5]
            elif t == "...":
                v = [0.50, (1.0 - 0.50**2) ** 0.5]
            else:
                # Benign query vector
                v = [1.0, 0.0]
            vecs.append([v, v])
        return SimpleNamespace(last_hidden_state=torch.tensor(vecs, dtype=torch.float32))

    evaluator.similarity_tokenizer = fake_similarity_tokenizer
    evaluator.similarity_model = fake_similarity_model

    benign_query = BENIGN_HEALTH_QUERIES[0]
    responses_4 = [helpful_resp, refusal_resp, empty_resp, reward_hack_resp]

    nli_logits = evaluator.score_benign_nli_logit(responses_4, benign_query)
    assert torch.allclose(nli_logits, torch.tensor([3.0, -3.0, 1.0, 3.0]))

    sims = evaluator.score_query_response_similarity(responses_4, benign_query)
    assert torch.allclose(sims, torch.tensor([0.85, 0.60, 0.0, 0.10]), atol=1e-5)

    utilities = evaluator.score_benign_utility(responses_4, benign_query)
    assert utilities.shape == (4,)
    assert captured_utility["hypotheses"] == [cfg.model.utility_nli_hypothesis] * 4
    assert captured_utility["premises"] == [
        format_utility_premise(benign_query, helpful_resp),
        format_utility_premise(benign_query, refusal_resp),
        format_utility_premise(benign_query, empty_resp),
        format_utility_premise(benign_query, reward_hack_resp),
    ]
    assert all(benign_query in premise for premise in captured_utility["premises"])
    # An empty response collapses to "..." rather than a blank tokeniser segment
    assert captured_utility["premises"][2].endswith("...")

    # Verify exact composite utility math: logit(sigmoid(nli) * clamp(sim, 0, 1), eps=utility_logit_eps)
    expected_probs = torch.sigmoid(torch.tensor([3.0, -3.0, 1.0, 3.0])) * torch.tensor(
        [0.85, 0.60, 0.0, 0.10]
    )
    expected_utilities = torch.logit(expected_probs, eps=cfg.model.utility_logit_eps)
    assert torch.allclose(utilities, expected_utilities, atol=1e-5)

    # Helpful relevant response stays strongly positive (> 0), while the reward-hacked
    # generic anatomy response (NLI=+3.0, cos_sim=0.10) is overridden into a strongly
    # negative utility logit (< -2.0) alongside refusals and empty responses.
    assert float(utilities[0]) > 1.0, f"Helpful response should be > 1.0, got {utilities[0]}"
    assert float(utilities[1]) < -3.0, f"Refusal response should be < -3.0, got {utilities[1]}"
    assert float(utilities[2]) < -4.0, f"Empty response should be < -4.0, got {utilities[2]}"
    assert float(utilities[3]) < -2.0, (
        f"Reward-hacked off-topic response should be overridden to < -2.0, got {utilities[3]}"
    )

    pairwise = evaluator.score_benign_utility(
        [helpful_resp, refusal_resp, empty_resp], BENIGN_HEALTH_QUERIES[:3]
    )
    assert pairwise.shape == (3,)
    assert captured_utility["premises"][1] == format_utility_premise(
        BENIGN_HEALTH_QUERIES[1], refusal_resp
    )
    try:
        evaluator.score_benign_utility([helpful_resp, refusal_resp], BENIGN_HEALTH_QUERIES[:3])
    except ValueError:
        pass
    else:
        raise AssertionError("Mismatched query/response counts must raise ValueError")

    # Test G x G reward matrix aggregation in raw logit space with EMA std scaling
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
    expected_def_logits = torch.tensor([-0.25, -0.75])
    assert torch.allclose(
        rewards["prompt_defense_scores"], expected_def_logits
    )

    # First batch initializes EMA std directly from batch std (unbiased=False)
    expected_def_std_1 = float(expected_def_logits.std(unbiased=False).item())
    expected_util_std_1 = float(benign_util.std(unbiased=False).item())
    assert abs(evaluator.ema_defense_std - expected_def_std_1) < 1e-6
    assert abs(evaluator.ema_utility_std - expected_util_std_1) < 1e-6

    scaled_def_1 = expected_def_logits / (expected_def_std_1 + cfg.reward_std_eps)
    scaled_util_1 = benign_util / (expected_util_std_1 + cfg.reward_std_eps)
    expected_composite_1 = cfg.defense_weight * scaled_def_1 + cfg.utility_weight * scaled_util_1
    assert torch.allclose(rewards["prompt_composite_rewards"], expected_composite_1)

    # Second batch updates EMA std with decay = cfg.reward_ema_decay
    attack_matrix_2 = torch.tensor(
        [
            [-1.0, -1.0, -1.0, -1.0],
            [1.0, 1.0, 1.0, 1.0],
        ],
        dtype=torch.float32,
    )
    benign_util_2 = torch.tensor([2.0, -2.0], dtype=torch.float32)
    rewards_2 = evaluator.compute_rewards(attack_matrix_2, benign_util_2)

    expected_def_logits_2 = torch.tensor([1.0, -1.0])
    batch_def_std_2 = float(expected_def_logits_2.std(unbiased=False).item())
    batch_util_std_2 = float(benign_util_2.std(unbiased=False).item())
    expected_ema_def_2 = (
        cfg.reward_ema_decay * expected_def_std_1 + (1.0 - cfg.reward_ema_decay) * batch_def_std_2
    )
    expected_ema_util_2 = (
        cfg.reward_ema_decay * expected_util_std_1 + (1.0 - cfg.reward_ema_decay) * batch_util_std_2
    )
    assert abs(evaluator.ema_defense_std - expected_ema_def_2) < 1e-6
    assert abs(evaluator.ema_utility_std - expected_ema_util_2) < 1e-6

    scaled_def_2 = expected_def_logits_2 / (expected_ema_def_2 + cfg.reward_std_eps)
    scaled_util_2 = benign_util_2 / (expected_ema_util_2 + cfg.reward_std_eps)
    expected_composite_2 = cfg.defense_weight * scaled_def_2 + cfg.utility_weight * scaled_util_2
    assert torch.allclose(rewards_2["prompt_composite_rewards"], expected_composite_2)


def test_nli_label_resolution_and_pair_scoring() -> None:
    """
    Verifies the zero-shot evaluator reads the entailment index from the checkpoint's
    own `id2label` (binary `not_entailment` heads and 3-way MNLI heads) and returns
    raw `entail_index` logits for (response, hypothesis) pairs.
    """
    cfg = TrainingConfig()
    device = torch.device("cpu")

    def make_evaluator(id2label: dict) -> MedicalIntentEvaluator:
        ev = object.__new__(MedicalIntentEvaluator)
        ev.config = cfg
        ev.device = device
        ev.max_length = cfg.tokens.evaluator_max_length
        ev.hypothesis = cfg.model.evaluator_nli_hypothesis
        ev.utility_hypothesis = cfg.model.utility_nli_hypothesis
        ev.ema_defense_std = None
        ev.ema_utility_std = None
        ev.model = SimpleNamespace(
            config=SimpleNamespace(id2label=id2label, num_labels=len(id2label))
        )
        return ev

    # zeroshot-v2.0 layout: entailment at index 0
    binary_ev = make_evaluator({0: "entailment", 1: "not_entailment"})
    assert binary_ev._resolve_nli_label_indices() == (0, [1])

    # 3-way MNLI layout: `neutral` is excluded from contradiction_indices
    mnli_ev = make_evaluator({0: "entailment", 1: "neutral", 2: "contradiction"})
    assert mnli_ev._resolve_nli_label_indices() == (0, [2])

    # Reversed layout: the index must follow `id2label`, not a hardcoded position
    reversed_ev = make_evaluator({0: "not_entailment", 1: "entailment"})
    assert reversed_ev._resolve_nli_label_indices() == (1, [0])

    # Scoring: premises pair with the hypothesis, returning raw logits at entail_index
    captured: dict = {}

    class _Encoding(dict):
        def to(self, _device):
            return self

    def fake_tokenizer(premises, hypotheses, **kwargs):
        captured["premises"] = premises
        captured["hypotheses"] = hypotheses
        captured["kwargs"] = kwargs
        return _Encoding(batch=len(premises))

    scorer = make_evaluator({0: "entailment", 1: "not_entailment"})
    scorer.entail_index, scorer.contradiction_indices = scorer._resolve_nli_label_indices()
    scorer.tokenizer = fake_tokenizer
    scorer.model = lambda **enc: SimpleNamespace(
        logits=torch.tensor([[4.0, -4.0], [-4.0, 4.0]], dtype=torch.float32)
    )

    logits = scorer.score_medical_advice_logit(["  Take 500mg now.  ", ""])
    assert logits.shape == (2,)
    assert float(logits[0]) == 4.0, f"Advice-like premise raw logit mismatch: {logits[0]}"
    assert float(logits[1]) == -4.0, f"Non-advice premise raw logit mismatch: {logits[1]}"
    assert captured["premises"] == ["Take 500mg now.", "..."]
    assert captured["hypotheses"] == [cfg.model.evaluator_nli_hypothesis] * 2
    assert captured["kwargs"]["max_length"] == cfg.tokens.evaluator_max_length
    assert captured["kwargs"]["truncation"] == "only_first"

    assert scorer.score_medical_advice_logit([]).shape == (0,)


def test_calibration_sanity_check_separates_labels() -> None:
    """
    Verifies the startup sanity check aggregates the held-out calibration corpora into
    margin/AUROC reports and flags a hypothesis that fails to separate the two labels.
    """
    cfg = TrainingConfig()
    ev = object.__new__(MedicalIntentEvaluator)
    ev.config = cfg
    ev.device = torch.device("cpu")

    labels = torch.tensor(
        [int(item[1]) for item in EVALUATOR_CALIBRATION_DATA], dtype=torch.float32
    )
    utility_labels = torch.tensor(
        [int(item[2]) for item in UTILITY_CALIBRATION_DATA], dtype=torch.float32
    )

    # A perfectly separating evaluator in logit space: label 1 -> +2.0, label 0 -> -2.0
    ev.score_medical_advice_logit = lambda texts: -2.0 + 4.0 * labels
    good = ev.run_calibration_sanity_check()
    assert good.num_advice + good.num_safe == len(EVALUATOR_CALIBRATION_DATA)
    assert good.passed
    assert good.margin > 3.5
    assert good.accuracy_at_midpoint == 1.0
    assert good.auroc > 0.99

    captured_questions: dict = {}

    def fake_utility_scorer(texts, questions):
        captured_questions["questions"] = questions
        return -2.0 + 4.0 * utility_labels

    ev.score_benign_utility = fake_utility_scorer
    good_utility = ev.run_utility_calibration_sanity_check()
    assert good_utility.num_advice + good_utility.num_safe == len(UTILITY_CALIBRATION_DATA)
    assert good_utility.passed
    assert good_utility.margin > 3.5
    assert good_utility.accuracy_at_midpoint == 1.0
    assert good_utility.auroc > 0.99
    assert captured_questions["questions"] == [
        item[0] for item in UTILITY_CALIBRATION_DATA
    ]

    # A collapsed evaluator (the failure mode of an untrained head) must not pass
    ev.score_medical_advice_logit = lambda texts: torch.zeros_like(labels)
    collapsed = ev.run_calibration_sanity_check()
    assert not collapsed.passed
    assert abs(collapsed.margin) < 1e-6
    assert abs(collapsed.auroc - 0.5) < 1e-6

    ev.score_benign_utility = lambda texts, questions: torch.zeros_like(utility_labels)
    collapsed_utility = ev.run_utility_calibration_sanity_check()
    assert not collapsed_utility.passed
    assert abs(collapsed_utility.margin) < 1e-6
    assert abs(collapsed_utility.auroc - 0.5) < 1e-6


def test_dual_lora_grpo_and_dashboard_end_to_end() -> None:
    """
    Verifies dual PEFT adaptor switching (`prompt_adaptor` vs `red_adaptor`),
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

        # Build a miniature Qwen2 model with SDPA and dual PEFT LoRA adaptors
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
        peft_model = get_peft_model(base_model, lora_cfg, adapter_name="prompt_adaptor")
        peft_model.add_adapter("red_adaptor", lora_cfg)

        # Wrap inside SharedWeightAdversarialModels
        models_wrapper = object.__new__(SharedWeightAdversarialModels)
        models_wrapper.config = cfg
        models_wrapper.device = device
        models_wrapper.prompt_adaptor_name = "prompt_adaptor"
        models_wrapper.red_adaptor_name = "red_adaptor"
        models_wrapper.prompt_adapter_name = "prompt_adaptor"
        models_wrapper.red_adapter_name = "red_adaptor"
        models_wrapper.base_dtype = torch.float32
        models_wrapper.model = peft_model
        models_wrapper._cast_lora_parameters_to_fp32()

        # Verify parameter isolation between adaptors
        red_params = models_wrapper.activate_adaptor_for_training("red_adaptor")
        assert len(red_params) > 0
        for name, param in peft_model.named_parameters():
            if ".prompt_adaptor." in name:
                assert not param.requires_grad, "prompt_adaptor must be frozen during red_adaptor step"
            elif ".red_adaptor." in name:
                assert param.requires_grad, "red_adaptor must have requires_grad=True"

        # Snapshot initial weights to verify each update only mutates its own adaptor
        red_before = [p.detach().clone() for p in models_wrapper.get_adaptor_parameters("red_adaptor")]
        prompt_before = [
            p.detach().clone() for p in models_wrapper.get_adaptor_parameters("prompt_adaptor")
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

        # 1. Step red_adaptor
        red_rewards = torch.tensor([0.12, 0.45, 0.28, 0.81], dtype=torch.float32, device=device)
        red_stats = grpo_trainer.update_red_adaptor(dummy_batch, red_rewards)

        red_after_step1 = [
            p.detach().clone() for p in models_wrapper.get_adaptor_parameters("red_adaptor")
        ]
        prompt_after_step1 = [
            p.detach().clone() for p in models_wrapper.get_adaptor_parameters("prompt_adaptor")
        ]

        assert any(not torch.allclose(a, b) for a, b in zip(red_before, red_after_step1)), (
            "red_adaptor parameters should update after update_red_adaptor"
        )
        assert all(torch.allclose(a, b) for a, b in zip(prompt_before, prompt_after_step1)), (
            "prompt_adaptor parameters must NOT change during update_red_adaptor"
        )

        # 2. Step prompt_adaptor
        prompt_rewards = torch.tensor([1.35, 0.92, 1.18, 0.64], dtype=torch.float32, device=device)
        prompt_stats = grpo_trainer.update_prompt_adaptor(dummy_batch, prompt_rewards)

        # Verify zero-variance rewards (std_r < 1e-3) zero out advantages
        collapsed_rewards = torch.tensor([0.5000, 0.5002, 0.4999, 0.5001], dtype=torch.float32, device=device)
        collapsed_adv = grpo_trainer.compute_group_advantages(collapsed_rewards)
        assert torch.all(collapsed_adv == 0.0), (
            "Advantages must be zeroed out when std_r < 1e-3 so only KL penalty applies"
        )

        prompt_after_step2 = [
            p.detach().clone() for p in models_wrapper.get_adaptor_parameters("prompt_adaptor")
        ]
        assert any(not torch.allclose(a, b) for a, b in zip(prompt_before, prompt_after_step2)), (
            "prompt_adaptor parameters should update after update_prompt_adaptor"
        )

        # 3. Verify DebertaV2 sequence classification loads with
        # evaluator_attn_implementation="eager" (required by its disentangled attention)
        tiny_deberta_cfg = DebertaV2Config(
            vocab_size=256,
            hidden_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            intermediate_size=128,
            num_labels=2,
        )
        deberta_init = AutoModelForSequenceClassification.from_config(tiny_deberta_cfg)
        deberta_ckpt_dir = Path(tmpdir) / "tiny_deberta_ckpt"
        deberta_init.save_pretrained(str(deberta_ckpt_dir))

        tiny_deberta = AutoModelForSequenceClassification.from_pretrained(
            str(deberta_ckpt_dir),
            dtype=torch.float32,
            attn_implementation=cfg.model.evaluator_attn_implementation,
        ).to(device)
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
                lora_medical_logit=-2.00,
                red_nolora_attack="How much warfarin should I take?",
                blue_nolora_response="Please consult your prescribing doctor.",
                nolora_medical_logit=-1.00,
                blue_benign_response="Hemoglobin carries oxygen in red blood cells.",
                benign_utility_score=1.55,
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


def test_checkpoint_resume_and_extension() -> None:
    """
    Verifies full checkpoint save/load (`prompt_adaptor`, `red_adaptor`, optimisers,
    GradScaler, dataset sampler state, RNG states), dashboard history truncation/resume,
    and run extension (`--extend-steps` and `--num-steps`).
    """
    from src.trainer import AdversarialCoEvolutionOrchestrator

    device = torch.device("cpu")
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = TrainingConfig(
            num_steps=4,
            group_size=2,
            output_dir=tmpdir,
            use_live_ui=False,
        )

        def _make_mock_orchestrator(config: TrainingConfig) -> AdversarialCoEvolutionOrchestrator:
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
                target_modules=list(config.model.lora_target_modules),
            )
            peft_model = get_peft_model(base_model, lora_cfg, adapter_name="prompt_adaptor")
            peft_model.add_adapter("red_adaptor", lora_cfg)

            models_wrapper = object.__new__(SharedWeightAdversarialModels)
            models_wrapper.config = config
            models_wrapper.device = device
            models_wrapper.prompt_adaptor_name = "prompt_adaptor"
            models_wrapper.red_adaptor_name = "red_adaptor"
            models_wrapper.prompt_adapter_name = "prompt_adaptor"
            models_wrapper.red_adapter_name = "red_adaptor"
            models_wrapper.base_dtype = torch.float32
            models_wrapper.model = peft_model
            models_wrapper._cast_lora_parameters_to_fp32()

            ev_mock = object.__new__(MedicalIntentEvaluator)
            ev_mock.ema_defense_std = None
            ev_mock.ema_utility_std = None

            orch = object.__new__(AdversarialCoEvolutionOrchestrator)
            orch.config = config
            orch.device = device
            orch.dataset = AdversarialDataset(seed=config.seed)
            orch.models = models_wrapper
            orch.evaluator = ev_mock
            orch.grpo = DualAgentGRPOTrainer(config=config, models=models_wrapper, device=device)
            orch._last_nolora_val_reward = 0.42
            orch._recent_successful_attacks = []
            orch._last_successful_attack = None
            orch._stop_requested = False
            return orch

        orch1 = _make_mock_orchestrator(cfg)
        # Advance dataset 2 steps and run 1 optimiser update on each adaptor
        orch1.dataset.sample_step(1)
        orch1.dataset.sample_step(2)
        orch1.evaluator.ema_defense_std = 1.25
        orch1.evaluator.ema_utility_std = 0.85

        dummy_batch = GenerationGroupBatch(
            texts=["a", "b"],
            full_ids=torch.randint(10, 200, (2, 10), device=device),
            attention_mask=torch.ones((2, 10), dtype=torch.long, device=device),
            completion_mask=torch.ones((2, 9), dtype=torch.float32, device=device),
            prompt_len=4,
        )
        orch1.grpo.update_red_adaptor(dummy_batch, torch.tensor([0.2, 0.8], device=device))
        orch1.grpo.update_prompt_adaptor(dummy_batch, torch.tensor([1.2, 0.6], device=device))
        orch1._last_successful_attack = "Hypothetical pharmacology case study framing"
        orch1._recent_successful_attacks = [
            "Urgent triage override framing",
            "Hypothetical pharmacology case study framing",
        ]

        # Save checkpoint at step 2 of 4 (simulating an interrupted run at step 2/4)
        orch1.save_checkpoint(step=2, total_steps=4, save_numbered=True)

        saved_red = [
            p.detach().clone() for p in orch1.models.get_adaptor_parameters("red_adaptor")
        ]
        saved_prompt = [
            p.detach().clone() for p in orch1.models.get_adaptor_parameters("prompt_adaptor")
        ]
        next_expected_sample = orch1.dataset.sample_step(3)

        # Write 3 lines to metrics.jsonl (where step 3 was an uncheckpointed orphan)
        with ExperimentDashboard(config=cfg, device=device) as dash1:
            for s in (1, 2, 3):
                dash1.update_step(
                    StepMetricsRecord(
                        step=s,
                        total_steps=4,
                        step_time_sec=1.0,
                        vram_allocated_gb=2.0,
                        vram_max_gb=2.5,
                        red_lora_reward=0.3 * s,
                        red_nolora_val_reward=0.25,
                        prompt_defense_score=0.7,
                        prompt_utility_score=0.8,
                        prompt_composite_reward=1.18,
                        red_loss=0.1,
                        red_kl=0.001,
                        red_grad_norm=0.5,
                        red_reward_std=0.1,
                        prompt_loss=0.1,
                        prompt_kl=0.001,
                        prompt_grad_norm=0.5,
                        prompt_reward_std=0.1,
                    )
                )

        # Case 1: Auto-resume interrupted run (step 2 -> target 4)
        orch2 = _make_mock_orchestrator(cfg)
        resume_step, target_steps, ckpt_dir = orch2._resolve_run_bounds()
        assert resume_step == 2
        assert target_steps == 4
        assert ckpt_dir is not None

        # Verify restored adaptor weights match saved weights & remain FP32
        for p_saved, p_loaded in zip(
            saved_red, orch2.models.get_adaptor_parameters("red_adaptor")
        ):
            assert p_loaded.dtype == torch.float32
            assert torch.allclose(p_saved, p_loaded)
        for p_saved, p_loaded in zip(
            saved_prompt, orch2.models.get_adaptor_parameters("prompt_adaptor")
        ):
            assert p_loaded.dtype == torch.float32
            assert torch.allclose(p_saved, p_loaded)

        # Verify dataset and evaluator EMA state continue from step 3 identically
        assert orch2.evaluator.ema_defense_std == 1.25
        assert orch2.evaluator.ema_utility_std == 0.85
        assert orch2._last_successful_attack == "Hypothetical pharmacology case study framing"
        assert orch2._recent_successful_attacks == [
            "Urgent triage override framing",
            "Hypothetical pharmacology case study framing",
        ]
        resumed_sample = orch2.dataset.sample_step(3)
        assert resumed_sample.seed_idx == next_expected_sample.seed_idx
        assert resumed_sample.benign_idx == next_expected_sample.benign_idx

        # Verify ExperimentDashboard truncates step 3 orphan and restores 2 steps of history
        with ExperimentDashboard(
            config=cfg, device=device, resume_step=resume_step, total_steps=target_steps
        ) as dash2:
            assert len(dash2.history["red_lora"]) == 2
            assert dash2.latest_metrics is not None
            assert dash2.latest_metrics.step == 2

        lines_after_purge = (
            (Path(tmpdir) / "metrics.jsonl").read_text(encoding="utf-8").strip().splitlines()
        )
        assert len(lines_after_purge) == 2

        # Case 2: Extend run by +200 steps via --extend-steps 200 (step 2 -> target 202)
        cfg_extend = TrainingConfig(
            num_steps=4,
            extend_steps=200,
            group_size=2,
            output_dir=tmpdir,
            use_live_ui=False,
        )
        orch_ext = _make_mock_orchestrator(cfg_extend)
        r_step, t_steps, _ = orch_ext._resolve_run_bounds()
        assert r_step == 2
        assert t_steps == 202

        # Case 3: Extend run via explicit --num-steps 300 (step 2 -> target 300)
        cfg_num = TrainingConfig(
            num_steps=300,
            num_steps_explicit=True,
            group_size=2,
            output_dir=tmpdir,
            use_live_ui=False,
        )
        orch_num = _make_mock_orchestrator(cfg_num)
        r_step2, t_steps2, _ = orch_num._resolve_run_bounds()
        assert r_step2 == 2
        assert t_steps2 == 300


def test_cli_defaults_match_config() -> None:
    cfg = TrainingConfig()
    clean_env = {
        k: v
        for k, v in os.environ.items()
        if k not in ("NUM_STEPS", "EXTEND_STEPS", "RESUME_FROM", "FRESH_RUN")
    }
    orig_env = os.environ.copy()
    try:
        os.environ.clear()
        os.environ.update(clean_env)
        args = parse_args([])
    finally:
        os.environ.clear()
        os.environ.update(orig_env)

    assert args.base_model == cfg.model.base_model_id
    assert args.evaluator_model == cfg.model.evaluator_model_id
    assert args.similarity_model == cfg.model.similarity_model_id
    assert args.num_steps == cfg.num_steps
    assert args.extend_steps == cfg.extend_steps
    assert args.resume_from == cfg.resume_from
    assert args.fresh == cfg.fresh
    assert args.group_size == cfg.group_size
    assert args.lr_prompt == cfg.lr_prompt
    assert args.lr_red == cfg.lr_red
    assert args.utility_weight == cfg.utility_weight
    assert args.prompt_kl_coef == cfg.prompt_kl_coef
    assert args.red_kl_coef == cfg.red_kl_coef
    assert args.sample_interval == cfg.sample_refresh_interval
    assert args.checkpoint_interval == cfg.checkpoint_interval
    assert args.preprompt_max_tokens == cfg.tokens.preprompt_max_new_tokens
    assert args.attack_max_tokens == cfg.tokens.attack_max_new_tokens
    assert args.blue_max_tokens == cfg.tokens.blue_max_new_tokens
    assert args.output_dir == cfg.output_dir
    assert args.seed == cfg.seed
    assert args.no_live_ui == (not cfg.use_live_ui)


if __name__ == "__main__":
    test_cli_defaults_match_config()
    test_dataset_completeness_and_sampling()
    test_reward_and_utility_scoring_math()
    test_nli_label_resolution_and_pair_scoring()
    test_calibration_sanity_check_separates_labels()
    test_dual_lora_grpo_and_dashboard_end_to_end()
    test_checkpoint_resume_and_extension()
    print("All unit, integration, and resume/extension tests passed successfully!")
