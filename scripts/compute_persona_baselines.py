"""Compute per-prompt persona vector baseline projections from pre-computed rollouts.

This script processes a dataset containing pre-computed rollouts (in the `outputs`
column) through the DPO/base model, extracting hidden state projections onto persona
direction vectors at ALL layers. The result is a dict mapping dataset row indices to
per-layer average projection scalars, used during GRPO training for rollout filtering.

Projections are computed over all response tokens in float32.

Supports sharding for parallel execution across multiple GPUs via SLURM array jobs.
Each shard writes its own output file; use --aggregate to merge shard outputs into
a single baselines.pt file.

Usage (single GPU):
    uv run python scripts/compute_persona_baselines.py \
        --model_name_or_path allenai/Olmo-3-7B-Think-DPO \
        --dataset_path /data/artifacts/frank/datasets/Dolci-Think-RL-7B-with-messages-hf-no-ifeval-complete \
        --persona_vector_path /home/fxiao/eval_awareness/eval_steering/vectors/OLMo3-7B-Base.pt \
        --output_dir /path/to/baselines_run

Usage (sharded, called by SLURM):
    uv run python scripts/compute_persona_baselines.py \
        --model_name_or_path allenai/Olmo-3-7B-Think-DPO \
        --dataset_path /data/artifacts/frank/datasets/Dolci-Think-RL-7B-with-messages-hf-no-ifeval-complete \
        --persona_vector_path /home/fxiao/eval_awareness/eval_steering/vectors/OLMo3-7B-Base.pt \
        --output_dir /path/to/baselines_run \
        --shard_id 0 --num_shards 16

Usage (aggregate after all shards finish):
    uv run python scripts/compute_persona_baselines.py \
        --aggregate --output_dir /path/to/baselines_run
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute persona vector baselines per prompt.")
    parser.add_argument("--aggregate", action="store_true", help="Aggregate shard outputs into baselines.pt")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for shards and final result")

    # Compute args (not needed for --aggregate)
    parser.add_argument("--model_name_or_path", type=str, default=None)
    parser.add_argument("--dataset_path", type=str, default=None)
    parser.add_argument("--persona_vector_path", type=str, default=None)
    parser.add_argument("--max_length", type=int, default=4096, help="Max tokens per prompt+output sequence")
    parser.add_argument("--shard_id", type=int, default=None, help="Shard index (0-indexed). If None, process all.")
    parser.add_argument("--num_shards", type=int, default=1, help="Total number of shards")
    parser.add_argument("--index_start", type=int, default=None, help="Only process indices >= this value")
    parser.add_argument("--index_end", type=int, default=None, help="Only process indices < this value")
    return parser.parse_args()


def aggregate(output_dir: str) -> None:
    """Merge per-shard JSONL files into a single baselines.pt."""
    shard_dir = Path(output_dir) / "shards"
    shard_files = sorted(shard_dir.glob("shard_*.jsonl"))
    if not shard_files:
        raise FileNotFoundError(f"No shard files found in {shard_dir}")

    print(f"Found {len(shard_files)} shard files")

    baselines: dict[int, dict[int, float]] = {}
    for shard_file in shard_files:
        with open(shard_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                idx = obj["idx"]
                per_layer = {int(k): v for k, v in obj["projections"].items()}
                baselines[idx] = per_layer

    output_path = Path(output_dir) / "baselines.pt"
    torch.save(baselines, output_path)
    print(f"Aggregated {len(baselines)} prompts to {output_path}")

    # Summary stats per layer
    if baselines:
        sample = next(iter(baselines.values()))
        layers = sorted(sample.keys())
        for layer_idx in layers:
            vals = [b[layer_idx] for b in baselines.values() if layer_idx in b]
            if vals:
                mean_val = sum(vals) / len(vals)
                print(f"  Layer {layer_idx}: mean={mean_val:.4f}, n={len(vals)}")


def compute(args: argparse.Namespace) -> None:
    """Run projection computation for a shard (or all data if no sharding)."""
    from datasets import load_dataset, load_from_disk
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Load persona vectors: shape (num_layers, hidden_dim)
    all_vectors = torch.load(args.persona_vector_path, map_location="cpu", weights_only=False)
    if isinstance(all_vectors, dict):
        num_layers = max(all_vectors.keys()) + 1
        all_vectors = torch.stack([all_vectors[i] for i in range(num_layers)])
    assert all_vectors.dim() == 2, f"Expected (num_layers, hidden_dim), got {all_vectors.shape}"
    num_layers = all_vectors.shape[0]

    # Normalize each layer's vector in float32
    all_vectors = all_vectors.float()
    all_vectors = all_vectors / all_vectors.norm(dim=1, keepdim=True)
    print(f"Persona vectors: {num_layers} layers, dim {all_vectors.shape[1]}")

    # Load model and tokenizer
    print(f"Loading model: {args.model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        use_cache=False,
    )
    model.eval()

    device = next(model.parameters()).device
    all_vectors = all_vectors.to(device)

    # Register hooks on ALL layers
    captured_hidden: dict[int, torch.Tensor] = {}

    def make_hook(layer_idx: int):
        def hook_fn(module: torch.nn.Module, input: tuple, output: tuple | torch.Tensor) -> None:
            hidden = output[0] if isinstance(output, tuple) else output
            captured_hidden[layer_idx] = hidden.detach()
        return hook_fn

    hooks = []
    for layer_idx in range(num_layers):
        handle = model.model.layers[layer_idx].register_forward_hook(make_hook(layer_idx))
        hooks.append(handle)

    # Load dataset
    print(f"Loading dataset: {args.dataset_path}")
    try:
        ds = load_from_disk(args.dataset_path)
    except FileNotFoundError:
        ds = load_dataset(args.dataset_path, split="train")

    required_cols = {"outputs"}
    missing = required_cols - set(ds.column_names)
    if missing:
        raise ValueError(f"Dataset missing required columns: {missing}. Available: {ds.column_names}")

    # Determine this shard's index range
    idx_start = args.index_start if args.index_start is not None else 0
    idx_end = args.index_end if args.index_end is not None else len(ds)
    all_indices = list(range(idx_start, idx_end))
    if args.shard_id is not None:
        shard_size = len(all_indices) // args.num_shards
        start = args.shard_id * shard_size
        end = start + shard_size if args.shard_id < args.num_shards - 1 else len(all_indices)
        shard_indices = all_indices[start:end]
        print(f"Shard {args.shard_id}/{args.num_shards}: indices {start}-{end-1} ({len(shard_indices)} rows)")
    else:
        shard_indices = all_indices

    # Output path
    shard_dir = Path(args.output_dir) / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    if args.shard_id is not None:
        output_path = shard_dir / f"shard_{args.shard_id:04d}.jsonl"
    else:
        output_path = shard_dir / "shard_0000.jsonl"

    # Load completed indices for resumption
    done_idxs: set[int] = set()
    if output_path.exists():
        with open(output_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    done_idxs.add(obj["idx"])
                except json.JSONDecodeError:
                    pass
        if done_idxs:
            print(f"Resuming: {len(done_idxs)} samples already completed")

    remaining = [idx for idx in shard_indices if idx not in done_idxs]
    print(f"Remaining: {len(remaining)} samples")

    skipped = 0
    processed = 0
    out_file = open(output_path, "a")

    for idx in tqdm(remaining, desc=f"Shard {args.shard_id or 0}"):
        row = ds[idx]

        # Skip rows without outputs
        outputs = row.get("outputs")
        if not outputs or len(outputs) == 0:
            skipped += 1
            continue

        # Use chat template to tokenize prompt
        prompt_text = row["prompt"]
        if prompt_text.startswith("user: "):
            user_content = prompt_text[len("user: "):]
        else:
            user_content = prompt_text

        prompt_messages = [{"role": "user", "content": user_content}]
        prompt_templated = tokenizer.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True
        )
        prompt_ids = tokenizer.encode(prompt_templated, add_special_tokens=False)

        # Accumulate per-layer projections across rollouts
        rollout_projections: dict[int, list[float]] = {l: [] for l in range(num_layers)}

        for output_text in outputs:
            if not output_text or len(output_text.strip()) == 0:
                continue

            full_messages = [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": output_text},
            ]
            full_templated = tokenizer.apply_chat_template(
                full_messages, tokenize=False, add_generation_prompt=False
            )
            full_ids = tokenizer.encode(full_templated, add_special_tokens=False)

            if len(full_ids) > args.max_length:
                full_ids = full_ids[: args.max_length]

            num_response_tokens = len(full_ids) - len(prompt_ids)
            if num_response_tokens == 0:
                continue

            input_ids = torch.tensor([full_ids], dtype=torch.long, device=device)
            attention_mask = torch.ones_like(input_ids)

            captured_hidden.clear()
            with torch.no_grad():
                model(input_ids=input_ids, attention_mask=attention_mask)

            if not captured_hidden:
                continue

            response_start = len(prompt_ids)

            for layer_idx in range(num_layers):
                if layer_idx not in captured_hidden:
                    continue
                hidden_states = captured_hidden[layer_idx]  # (1, T, D)
                response_hidden = hidden_states[0, response_start:]  # (T_resp, D)
                if response_hidden.shape[0] == 0:
                    continue
                projection = (response_hidden.float() @ all_vectors[layer_idx]).mean().item()
                rollout_projections[layer_idx].append(projection)

        # Average across rollouts for each layer
        per_layer_baselines: dict[int, float] = {}
        for layer_idx in range(num_layers):
            projs = rollout_projections[layer_idx]
            if projs:
                per_layer_baselines[layer_idx] = sum(projs) / len(projs)

        if per_layer_baselines:
            row_out = {"idx": idx, "projections": per_layer_baselines}
            out_file.write(json.dumps(row_out) + "\n")
            processed += 1
            if processed % 100 == 0:
                out_file.flush()

    out_file.flush()
    out_file.close()

    # Remove hooks
    for handle in hooks:
        handle.remove()

    print(f"Processed: {processed}, Skipped (no outputs): {skipped}")
    print(f"Output: {output_path}")


def main() -> None:
    args = parse_args()

    if args.aggregate:
        aggregate(args.output_dir)
    else:
        if not args.model_name_or_path or not args.dataset_path or not args.persona_vector_path:
            raise ValueError("--model_name_or_path, --dataset_path, and --persona_vector_path are required for compute mode")
        compute(args)


if __name__ == "__main__":
    main()
