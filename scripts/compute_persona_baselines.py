"""Compute per-prompt persona vector baseline projections from pre-computed rollouts.

This script processes a dataset containing pre-computed rollouts (in the `outputs`
column) through the DPO/base model, extracting hidden state projections onto a
persona direction vector. The result is a dict mapping dataset row indices to average
projection scalars, used during GRPO training for rollout filtering.

Usage:
    uv run python scripts/compute_persona_baselines.py \
        --model_name_or_path allenai/Olmo-3-7B-Think-DPO \
        --dataset_path /data/artifacts/frank/datasets/Dolci-Think-RL-7B-with-messages-hf \
        --persona_vector_path /path/to/persona_vector.pt \
        --layer_idx 16 \
        --output_path /path/to/baselines.pt
"""

from __future__ import annotations

import argparse

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute persona vector baselines per prompt.")
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--persona_vector_path", type=str, required=True)
    parser.add_argument("--layer_idx", type=int, default=16)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--max_length", type=int, default=4096, help="Max tokens per prompt+output sequence")
    parser.add_argument("--limit", type=int, default=None, help="Process only this many prompts (for testing)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Load persona vector
    raw_vector = torch.load(args.persona_vector_path, map_location="cpu", weights_only=True)
    direction = raw_vector.to(torch.bfloat16)
    direction = direction / direction.norm()

    # Load model and tokenizer
    print(f"Loading model: {args.model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        use_cache=False,
    )
    model.eval()

    # Move direction to model device
    device = next(model.parameters()).device
    direction = direction.to(device)

    # Register hook on target layer
    captured_hidden: list[torch.Tensor] = []

    def hook_fn(module: torch.nn.Module, input: tuple, output: tuple | torch.Tensor) -> None:
        hidden = output[0] if isinstance(output, tuple) else output
        captured_hidden.append(hidden.detach())

    target_layer = model.model.layers[args.layer_idx]
    hook_handle = target_layer.register_forward_hook(hook_fn)

    # Load dataset
    print(f"Loading dataset: {args.dataset_path}")
    ds = load_dataset(args.dataset_path, split="train")

    # Verify required columns exist
    required_cols = {"input_ids_prompt", "outputs"}
    missing = required_cols - set(ds.column_names)
    if missing:
        raise ValueError(
            f"Dataset missing required columns: {missing}. "
            f"Available: {ds.column_names}"
        )

    # Baselines keyed by dataset row index (int -> float)
    baselines: dict[int, float] = {}
    skipped = 0
    processed = 0

    limit = args.limit if args.limit is not None else len(ds)

    for idx in tqdm(range(min(limit, len(ds))), desc="Computing baselines"):
        row = ds[idx]

        # Skip rows without outputs
        outputs = row.get("outputs")
        if not outputs or len(outputs) == 0:
            skipped += 1
            continue

        # Use input_ids_prompt directly from the dataset — this matches what
        # the training pipeline puts into packed sequences.
        prompt_ids = row["input_ids_prompt"]

        # Process each rollout
        rollout_projections: list[float] = []

        for output_text in outputs:
            if not output_text or len(output_text.strip()) == 0:
                continue

            # Tokenize output text and concatenate with prompt token IDs
            output_ids = tokenizer.encode(output_text, add_special_tokens=False)
            full_ids = prompt_ids + output_ids
            if len(full_ids) > args.max_length:
                full_ids = full_ids[: args.max_length]
                output_ids = full_ids[len(prompt_ids) :]

            if len(output_ids) == 0:
                continue

            input_ids = torch.tensor([full_ids], dtype=torch.long, device=device)
            attention_mask = torch.ones_like(input_ids)

            # Forward pass (hook captures hidden states)
            captured_hidden.clear()
            with torch.no_grad():
                model(input_ids=input_ids, attention_mask=attention_mask)

            if not captured_hidden:
                continue

            hidden_states = captured_hidden[0]  # (1, T, D)

            # Mean projection over response tokens only
            response_start = len(prompt_ids)
            response_hidden = hidden_states[0, response_start:]  # (T_resp, D)
            if response_hidden.shape[0] == 0:
                continue

            projection = (response_hidden @ direction).mean().item()
            rollout_projections.append(projection)

        if rollout_projections:
            baselines[idx] = sum(rollout_projections) / len(rollout_projections)
            processed += 1

    hook_handle.remove()

    print(f"Processed: {processed}, Skipped (no outputs): {skipped}")
    print(f"Saving baselines to: {args.output_path}")
    torch.save(baselines, args.output_path)
    print("Done.")


if __name__ == "__main__":
    main()
