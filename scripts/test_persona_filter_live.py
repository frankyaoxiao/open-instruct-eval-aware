"""Minimal live test of persona filter: load model, forward pass, verify hook + filtering.

Uses REAL prompts from the training dataset so that baseline lookups are
meaningful — the projections should be close to the stored baselines.

Requires 1 GPU. No training, no backward pass, no DeepSpeed, no vLLM.

Usage:
    uv run python scripts/test_persona_filter_live.py
"""

import sys
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from open_instruct.persona_filter import PersonaFilter, PersonaFilterConfig
from open_instruct import data_types


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("FAIL: No GPU available")
        sys.exit(1)

    model_name = "allenai/Olmo-3-7B-Think-DPO"
    dataset_path = "/data/artifacts/frank/datasets/Dolci-Think-RL-7B-with-messages-hf"
    vector_path = "/home/fxiao/eval_awareness/eval_steering/vectors/OLMo3-7B-Base.pt"
    baseline_path = "/home/fxiao/eval_awareness/persona_attribution/runs/baselines_dpo/baselines_full_merged.pt"
    num_test_rollouts = 4

    # 1. Load model
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="auto", use_cache=False
    )
    model.eval()
    print(f"  Model loaded on {next(model.parameters()).device}")

    # 2. Load dataset (same one used in training)
    print(f"Loading dataset: {dataset_path}")
    ds = load_dataset(dataset_path, split="train")
    print(f"  Dataset: {len(ds)} rows")

    # 3. Initialize persona filter
    print("Initializing PersonaFilter...")
    config = PersonaFilterConfig(
        vector_path=vector_path,
        baseline_path=baseline_path,
        layer_idx=20,
        threshold=2.0,
        max_filter_rate=0.5,
    )
    pf = PersonaFilter(config, device)
    pf.register_hook(model)
    print(f"  Hook registered. Baselines loaded: {len(pf.baselines)}")

    # 4. Pick real prompts from the dataset that have baselines
    # Use indices 0..3 — these are the same positional indices the training
    # pipeline would assign via dataset_transformation.py add_column("index", range(len(ds)))
    test_indices = [0, 1, 2, 3]
    for idx in test_indices:
        assert idx in pf.baselines, f"Index {idx} not in baselines!"

    print(f"\n{'='*60}")
    print(f"Testing with {num_test_rollouts} REAL prompts from dataset")
    print(f"{'='*60}")

    # 5. Build packed sequence from real prompts + synthetic short responses
    # (We generate short responses ourselves since the real model output would
    # require vLLM. The key test is that the dataset_indices map to the right
    # baselines.)
    all_ids = []
    response_mask_vals = []
    ds_index_vals = []
    attention_mask_vals = []

    for i, idx in enumerate(test_indices):
        row = ds[idx]
        # Extract prompt from messages (same as training pipeline)
        messages = row["messages"]
        user_msg = next(m["content"] for m in messages if m["role"] == "user")

        print(f"\n  Rollout {i}: dataset idx={idx}")
        print(f"    Prompt: {user_msg[:80]}...")
        print(f"    Stored baseline (layer 20): {pf.baselines[idx]:.4f}")

        # Build prompt + short synthetic response
        prompt_msgs = [{"role": "user", "content": user_msg}]
        prompt_text = tokenizer.apply_chat_template(prompt_msgs, tokenize=False, add_generation_prompt=True)
        prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)

        # Use a generic short response so we get some response tokens to project
        response_text = "Let me think about this step by step. The answer involves careful reasoning."
        full_msgs = [
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": response_text},
        ]
        full_text = tokenizer.apply_chat_template(full_msgs, tokenize=False, add_generation_prompt=False)
        full_ids = tokenizer.encode(full_text, add_special_tokens=False)

        n_prompt = len(prompt_ids)
        n_response = len(full_ids) - n_prompt
        print(f"    Tokens: {n_prompt} prompt + {n_response} response = {len(full_ids)} total")

        all_ids.extend(full_ids)
        response_mask_vals.extend([0] * n_prompt + [i + 1] * n_response)
        ds_index_vals.extend([idx] * len(full_ids))
        attention_mask_vals.extend([i + 1] * len(full_ids))

    seq_len = len(all_ids)
    print(f"\n  Packed sequence: {seq_len} tokens total")

    # Build tensors
    query_responses = torch.tensor([all_ids], dtype=torch.long, device=device)
    attention_masks = torch.tensor([attention_mask_vals], dtype=torch.long, device=device)
    response_masks_int = torch.tensor([response_mask_vals], dtype=torch.long, device=device)
    dataset_indices = torch.tensor([ds_index_vals], dtype=torch.long, device=device)
    position_ids = torch.zeros_like(attention_masks)
    for gid in attention_masks[0].unique():
        if gid == 0:
            continue
        positions = (attention_masks[0] == gid)
        position_ids[0, positions] = torch.arange(positions.sum(), device=device)

    original_response_masks = [response_masks_int.clone()]
    bool_response_masks = [response_masks_int.bool()]

    data_bt = data_types.CollatedBatchData(
        query_responses=[query_responses],
        attention_masks=[attention_masks],
        response_masks=bool_response_masks,
        position_ids=[position_ids],
        advantages=[torch.ones(1, seq_len, device=device)],
        vllm_logprobs=[torch.zeros(1, seq_len, device=device)],
        dataset_indices=[dataset_indices],
    )

    # 6. Forward pass with hook capture
    print(f"\n{'='*60}")
    print("Forward pass with persona hook...")
    print(f"{'='*60}")
    pf.begin_capture()
    with torch.no_grad():
        model(input_ids=query_responses, attention_mask=(attention_masks > 0).long(), position_ids=position_ids)
    captured = pf.end_capture()
    print(f"  Captured {len(captured)} projection tensor(s)")

    if len(captured) == 0:
        print("FAIL: No projections captured!")
        sys.exit(1)

    # 7. Manually verify per-rollout projections vs baselines
    print(f"\n{'='*60}")
    print("Per-rollout projection vs baseline comparison:")
    print(f"{'='*60}")
    proj_tensor = captured[0]  # (1, T)
    orig_mask = response_masks_int[0]  # (T,)

    for i, idx in enumerate(test_indices):
        rid = i + 1
        resp_positions = (orig_mask == rid)
        mean_proj = proj_tensor[0][resp_positions.cpu()].mean().item()
        baseline = pf.baselines[idx]
        diff = abs(mean_proj - baseline)
        would_filter = "FILTERED" if diff > config.threshold else "kept"

        print(f"  Rollout {i} (ds_idx={idx}):")
        print(f"    Online projection:  {mean_proj:.4f}")
        print(f"    Stored baseline:    {baseline:.4f}")
        print(f"    |diff|:             {diff:.4f}  -> {would_filter}")

    # 8. Run filter_rollouts and check metrics
    print(f"\n{'='*60}")
    print("filter_rollouts() output:")
    print(f"{'='*60}")
    metrics = pf.filter_rollouts(data_bt, original_response_masks, captured)
    for k, v in sorted(metrics.items()):
        print(f"  {k}: {v}")

    # 9. Verify
    total = metrics.get("persona/total_rollouts", 0)
    baseline_miss = metrics.get("persona/baseline_miss_rate", 1.0)

    if total == 0:
        print("\nFAIL: No rollouts processed")
        sys.exit(1)
    if baseline_miss > 0:
        print(f"\nFAIL: Baseline miss rate {baseline_miss:.2%} — index mismatch!")
        sys.exit(1)

    print(f"\nPASS: {int(total)} rollouts, 0% baseline misses, projections compared to correct prompts.")

    pf.remove_hook()


if __name__ == "__main__":
    main()
