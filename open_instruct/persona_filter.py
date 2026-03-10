"""Persona vector-based rollout filtering for GRPO training.

Filters rollouts whose activation projections onto a persona direction drift
too far in the negative direction from a pre-computed baseline.

The persona vector is deployment_mean - evaluation_mean, so more negative
projections = more eval-aware. We filter rollouts where
(projection - baseline) < -threshold, i.e. rollouts that are drifting
toward eval-awareness relative to the reference model baseline.

Projections are computed over all response tokens in float32, matching the
offline baseline computation pipeline.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import torch

from open_instruct import data_types, logger_utils

logger = logger_utils.setup_logger(__name__)


@dataclass
class PersonaFilterConfig:
    vector_path: str
    baseline_path: str
    layer_idx: int = 20
    threshold: float = 2.0
    max_filter_rate: float = 0.5


class PersonaFilter:
    """Filters rollouts based on activation projections onto a persona vector.

    Registers a forward hook on a specific layer of the reference model to
    capture hidden states during the already-happening ref model forward pass.
    Projects hidden states onto a persona direction and compares to pre-computed
    per-prompt baselines keyed by dataset row index.

    The persona vector file contains all layers (shape [num_layers, D]).
    Only the layer specified by layer_idx is used at runtime.
    Baselines are stored as {dataset_row_idx: {layer_idx: float}}.
    """

    def __init__(self, config: PersonaFilterConfig, device: torch.device, log_dir: str | None = None) -> None:
        # Load all-layer vectors and select the target layer
        all_vectors = torch.load(config.vector_path, map_location="cpu", weights_only=False)
        if isinstance(all_vectors, dict):
            num_layers = max(all_vectors.keys()) + 1
            all_vectors = torch.stack([all_vectors[i] for i in range(num_layers)])
        assert all_vectors.dim() == 2, f"Expected (num_layers, hidden_dim), got {all_vectors.shape}"

        # Normalize in float32 and select target layer
        all_vectors = all_vectors.float()
        all_vectors = all_vectors / all_vectors.norm(dim=1, keepdim=True)
        self.direction = all_vectors[config.layer_idx].to(device)  # (D,) float32

        # Load baseline dict: {dataset_row_index: {layer_idx: float}}
        raw_baselines = torch.load(config.baseline_path, map_location="cpu", weights_only=True)
        # Extract only the layer we care about
        self.baselines: dict[int, float] = {}
        for idx, layer_dict in raw_baselines.items():
            if config.layer_idx in layer_dict:
                self.baselines[idx] = layer_dict[config.layer_idx]

        self.layer_idx = config.layer_idx
        self.threshold = config.threshold
        self.max_filter_rate = config.max_filter_rate
        self.device = device

        self._captured_projections: list[torch.Tensor] | None = None
        self._hook_handle: torch.utils.hooks.RemovableHook | None = None
        self._log_dir: Path | None = Path(log_dir) if log_dir is not None else None

        logger.info(
            f"PersonaFilter initialized: layer={self.layer_idx}, "
            f"threshold={self.threshold}, max_filter_rate={self.max_filter_rate}, "
            f"baselines={len(self.baselines)} prompts"
        )

    def register_hook(self, ref_model: torch.nn.Module) -> None:
        """Register a forward hook on the target layer of the ref model."""
        # Navigate through DeepSpeed wrapping: module.model.layers[idx]
        unwrapped = ref_model
        if hasattr(unwrapped, "module"):
            unwrapped = unwrapped.module
        target_layer = unwrapped.model.layers[self.layer_idx]
        self._hook_handle = target_layer.register_forward_hook(self._hook_fn)
        logger.info(f"Registered persona hook on layer {self.layer_idx}")

    def remove_hook(self) -> None:
        """Remove the forward hook."""
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None

    def _hook_fn(
        self, module: torch.nn.Module, input: tuple[torch.Tensor, ...], output: tuple[torch.Tensor, ...] | torch.Tensor
    ) -> None:
        """Forward hook that captures per-token projections onto the persona direction.

        Projects in float32 to match offline baseline computation.
        """
        # output is typically a tuple; output[0] is hidden states (B, T, D)
        hidden_states = output[0] if isinstance(output, tuple) else output
        # Project in float32: (B, T, D) @ (D,) -> (B, T)
        projections = (hidden_states.detach().float() @ self.direction).cpu()
        if self._captured_projections is not None:
            self._captured_projections.append(projections)

    def begin_capture(self) -> None:
        """Start accumulating per-sample projection tensors."""
        self._captured_projections = []

    def end_capture(self) -> list[torch.Tensor]:
        """Stop capturing and return the accumulated projections."""
        result = self._captured_projections if self._captured_projections is not None else []
        self._captured_projections = None
        return result

    def filter_rollouts(
        self,
        data_bt: data_types.CollatedBatchData,
        original_response_masks: list[torch.Tensor],
        captured_projections: list[torch.Tensor],
        training_step: int | None = None,
    ) -> dict[str, float]:
        """Apply persona-based filtering to rollouts in packed sequences.

        Modifies data_bt.response_masks in-place to zero out filtered rollouts.

        Args:
            data_bt: The collated batch data (response_masks already bool).
                Must have dataset_indices populated for baseline lookup.
            original_response_masks: Integer-valued response masks before bool
                conversion. Rollout i has value i+1 (1-indexed).
            captured_projections: Per-sample projection tensors from the hook,
                each of shape (1, seq_len) or (B, seq_len).
            training_step: Current training step, used for logging filtered IDs.

        Returns:
            Dict of metrics for logging.
        """
        if len(captured_projections) != len(original_response_masks):
            logger.warning(
                f"PersonaFilter: hook captured {len(captured_projections)} projections "
                f"but expected {len(original_response_masks)} samples. "
                f"Skipping filtering this step (batch_size mismatch?)."
            )
            return {"persona/skipped": 1.0}

        if data_bt.dataset_indices is None:
            logger.warning("PersonaFilter: dataset_indices not available. Skipping filtering.")
            return {"persona/skipped": 1.0}

        total_rollouts = 0
        total_filtered = 0
        total_baseline_misses = 0
        all_diffs: list[float] = []
        all_projections: list[float] = []
        filtered_ids: list[dict] = []

        for sample_idx in range(len(original_response_masks)):
            orig_mask = original_response_masks[sample_idx]  # (B, T) int, on device
            ds_indices = data_bt.dataset_indices[sample_idx]  # (B, T) long, on device

            # Get the projections for this sample
            if sample_idx >= len(captured_projections):
                continue
            proj_tensor = captured_projections[sample_idx]  # (B, T) on cpu

            # Process each row in the micro-batch (usually B=1 for packed sequences)
            for row_idx in range(orig_mask.shape[0]):
                row_orig_mask = orig_mask[row_idx]  # (T,)
                row_ds_indices = ds_indices[row_idx]  # (T,)
                row_proj = proj_tensor[row_idx] if proj_tensor.dim() > 1 else proj_tensor  # (T,)

                # Find unique rollout IDs in this packed sequence (nonzero values)
                rollout_ids = row_orig_mask.unique()
                rollout_ids = rollout_ids[rollout_ids > 0]

                # (rollout_id, diff, dataset_idx) tuples for rollouts drifting too negative
                rollouts_to_filter: list[tuple[int, float, int]] = []

                for rid in rollout_ids.tolist():
                    rid_int = int(rid)
                    total_rollouts += 1

                    # Response token positions for this rollout
                    resp_positions = row_orig_mask == rid_int
                    if resp_positions.sum() == 0:
                        continue

                    # Mean projection over all response tokens
                    mean_proj = row_proj[resp_positions.cpu()].mean().item()
                    all_projections.append(mean_proj)

                    # Get dataset index for this rollout from any response token
                    ds_idx = int(row_ds_indices[resp_positions][0].item())

                    baseline = self.baselines.get(ds_idx)
                    if baseline is None:
                        total_baseline_misses += 1
                        continue

                    diff = mean_proj - baseline
                    all_diffs.append(diff)

                    # Filter rollouts drifting toward eval-awareness (more negative)
                    if diff < self.threshold:
                        rollouts_to_filter.append((rid_int, diff, ds_idx))

                # Apply max_filter_rate safety valve — keep most negative (worst offenders)
                max_allowed = int(len(rollout_ids) * self.max_filter_rate)
                if len(rollouts_to_filter) > max_allowed:
                    rollouts_to_filter.sort(key=lambda x: x[1])
                    rollouts_to_filter = rollouts_to_filter[:max_allowed]

                # Zero out filtered rollouts in the bool response mask
                for rid_int, diff_val, ds_idx in rollouts_to_filter:
                    total_filtered += 1
                    filtered_ids.append({"dataset_idx": ds_idx, "diff": round(diff_val, 4)})
                    mask_to_clear = orig_mask[row_idx] == rid_int
                    data_bt.response_masks[sample_idx][row_idx] &= ~mask_to_clear.to(
                        data_bt.response_masks[sample_idx].device
                    )

        metrics: dict[str, float] = {
            "persona/num_filtered": float(total_filtered),
            "persona/total_rollouts": float(total_rollouts),
            "persona/filter_rate": total_filtered / max(total_rollouts, 1),
            "persona/baseline_miss_rate": total_baseline_misses / max(total_rollouts, 1),
        }
        if all_diffs:
            metrics["persona/mean_diff"] = sum(all_diffs) / len(all_diffs)
            metrics["persona/min_diff"] = min(all_diffs)
            metrics["persona/max_diff"] = max(all_diffs)
        if all_projections:
            metrics["persona/mean_projection"] = sum(all_projections) / len(all_projections)

        # Log filtered rollout IDs to jsonl file (one file per rank to avoid races)
        if filtered_ids and self._log_dir is not None:
            rank = int(os.environ.get("RANK", 0))
            log_path = self._log_dir / f"persona_filtered_rollouts_rank{rank}.jsonl"
            try:
                log_path.parent.mkdir(parents=True, exist_ok=True)
                with open(log_path, "a") as f:
                    f.write(json.dumps({"training_step": training_step, "filtered": filtered_ids}) + "\n")
            except OSError:
                logger.warning(f"Failed to write filtered rollout IDs to {log_path}")

        return metrics
