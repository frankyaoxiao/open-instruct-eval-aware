"""Tests for persona_filter.py — filter_rollouts logic and ID logging."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from open_instruct.data_types import CollatedBatchData
from open_instruct.persona_filter import PersonaFilter, PersonaFilterConfig


@pytest.fixture
def tmp_artifacts(tmp_path: Path):
    """Create minimal persona vector and baseline files."""
    hidden_dim = 16
    num_layers = 4
    layer_idx = 2

    # Persona vector: (num_layers, hidden_dim) — random unit vectors
    vectors = torch.randn(num_layers, hidden_dim)
    vectors = vectors / vectors.norm(dim=1, keepdim=True)
    vector_path = tmp_path / "vector.pt"
    torch.save(vectors, vector_path)

    # Baselines: {dataset_row_idx: {layer_idx: float}}
    # dataset indices 0, 1, 2 have baselines; 99 does not
    baselines = {
        0: {layer_idx: 0.5},
        1: {layer_idx: -0.3},
        2: {layer_idx: 0.0},
    }
    baseline_path = tmp_path / "baselines.pt"
    torch.save(baselines, baseline_path)

    return {
        "vector_path": str(vector_path),
        "baseline_path": str(baseline_path),
        "layer_idx": layer_idx,
        "hidden_dim": hidden_dim,
        "num_layers": num_layers,
        "tmp_path": tmp_path,
    }


def _make_filter(tmp_artifacts, threshold=-0.5, max_filter_rate=0.5, log_dir=None):
    config = PersonaFilterConfig(
        vector_path=tmp_artifacts["vector_path"],
        baseline_path=tmp_artifacts["baseline_path"],
        layer_idx=tmp_artifacts["layer_idx"],
        threshold=threshold,
        max_filter_rate=max_filter_rate,
    )
    return PersonaFilter(config, device=torch.device("cpu"), log_dir=log_dir)


def _make_batch(response_mask_values, dataset_idx_values, seq_len=10):
    """Build a minimal CollatedBatchData for testing.

    Args:
        response_mask_values: list of (B, T) int tensors (rollout IDs per token)
        dataset_idx_values: list of (B, T) long tensors (dataset row index per token)
    """
    n_samples = len(response_mask_values)
    return CollatedBatchData(
        query_responses=[torch.zeros(1, seq_len, dtype=torch.long) for _ in range(n_samples)],
        attention_masks=[torch.ones(1, seq_len, dtype=torch.long) for _ in range(n_samples)],
        position_ids=[torch.arange(seq_len).unsqueeze(0) for _ in range(n_samples)],
        advantages=[torch.zeros(1, seq_len) for _ in range(n_samples)],
        response_masks=[m.clone().bool() for m in response_mask_values],
        vllm_logprobs=[torch.zeros(1, seq_len) for _ in range(n_samples)],
        dataset_indices=dataset_idx_values,
    )


class TestFilterRollouts:
    def test_basic_filtering(self, tmp_artifacts):
        """Rollouts with diff < threshold get filtered."""
        pf = _make_filter(tmp_artifacts, threshold=-0.5)

        # 1 sample, 1 row, 10 tokens, 2 rollouts (IDs 1 and 2)
        # Rollout 1: tokens 0-4 (dataset_idx=0, baseline=0.5)
        # Rollout 2: tokens 5-9 (dataset_idx=1, baseline=-0.3)
        resp_mask = torch.zeros(1, 10, dtype=torch.long)
        resp_mask[0, 0:5] = 1
        resp_mask[0, 5:10] = 2

        ds_indices = torch.zeros(1, 10, dtype=torch.long)
        ds_indices[0, 0:5] = 0
        ds_indices[0, 5:10] = 1

        data_bt = _make_batch([resp_mask], [ds_indices])
        original_masks = [resp_mask.clone()]

        # Projections: rollout 1 gets mean_proj = -0.5 (diff = -0.5 - 0.5 = -1.0, < threshold)
        # Rollout 2 gets mean_proj = 0.0 (diff = 0.0 - (-0.3) = 0.3, > threshold)
        proj = torch.zeros(1, 10)
        proj[0, 0:5] = -0.5  # rollout 1 projection
        proj[0, 5:10] = 0.0  # rollout 2 projection

        # Override direction to be identity-like so projections are predictable
        pf.direction = torch.zeros(tmp_artifacts["hidden_dim"])
        # We bypass the hook — directly pass projections
        metrics = pf.filter_rollouts(data_bt, original_masks, [proj], training_step=42)

        assert metrics["persona/num_filtered"] == 1.0
        assert metrics["persona/total_rollouts"] == 2.0
        assert metrics["persona/filter_rate"] == 0.5

        # Rollout 1 should be zeroed out in response_masks
        assert data_bt.response_masks[0][0, 0:5].sum() == 0
        # Rollout 2 should be preserved
        assert data_bt.response_masks[0][0, 5:10].all()

    def test_no_filtering_when_above_threshold(self, tmp_artifacts):
        """No rollouts filtered when all diffs are above threshold."""
        pf = _make_filter(tmp_artifacts, threshold=-10.0)

        resp_mask = torch.zeros(1, 10, dtype=torch.long)
        resp_mask[0, 0:5] = 1
        resp_mask[0, 5:10] = 2

        ds_indices = torch.zeros(1, 10, dtype=torch.long)
        ds_indices[0, 0:5] = 0
        ds_indices[0, 5:10] = 1

        data_bt = _make_batch([resp_mask], [ds_indices])
        original_masks = [resp_mask.clone()]

        proj = torch.zeros(1, 10)

        metrics = pf.filter_rollouts(data_bt, original_masks, [proj])
        assert metrics["persona/num_filtered"] == 0.0
        assert data_bt.response_masks[0].all()

    def test_max_filter_rate(self, tmp_artifacts):
        """max_filter_rate caps the number of filtered rollouts."""
        # 4 rollouts, all should fail threshold, but max_filter_rate=0.25 => only 1 allowed
        pf = _make_filter(tmp_artifacts, threshold=100.0, max_filter_rate=0.25)
        # threshold=100 means diff < 100 triggers filtering (everything)

        # Extend baselines to cover dataset indices 0-3
        pf.baselines = {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0}

        seq_len = 20
        resp_mask = torch.zeros(1, seq_len, dtype=torch.long)
        resp_mask[0, 0:5] = 1
        resp_mask[0, 5:10] = 2
        resp_mask[0, 10:15] = 3
        resp_mask[0, 15:20] = 4

        ds_indices = torch.zeros(1, seq_len, dtype=torch.long)
        ds_indices[0, 0:5] = 0
        ds_indices[0, 5:10] = 1
        ds_indices[0, 10:15] = 2
        ds_indices[0, 15:20] = 3

        data_bt = _make_batch([resp_mask], [ds_indices], seq_len=seq_len)
        original_masks = [resp_mask.clone()]

        # Different projections so sorting is deterministic
        proj = torch.zeros(1, seq_len)
        proj[0, 0:5] = -3.0  # worst offender
        proj[0, 5:10] = -2.0
        proj[0, 10:15] = -1.0
        proj[0, 15:20] = 0.0

        metrics = pf.filter_rollouts(data_bt, original_masks, [proj])

        # max_allowed = int(4 * 0.25) = 1
        assert metrics["persona/num_filtered"] == 1.0
        # The worst offender (rollout 1, diff=-3.0) should be filtered
        assert data_bt.response_masks[0][0, 0:5].sum() == 0
        # Others should survive
        assert data_bt.response_masks[0][0, 5:10].all()

    def test_baseline_miss(self, tmp_artifacts):
        """Rollouts with missing baselines are not filtered."""
        pf = _make_filter(tmp_artifacts, threshold=100.0)
        # baseline exists for ds_idx 0,1,2 — not for 99

        resp_mask = torch.zeros(1, 10, dtype=torch.long)
        resp_mask[0, 0:5] = 1
        resp_mask[0, 5:10] = 2

        ds_indices = torch.zeros(1, 10, dtype=torch.long)
        ds_indices[0, 0:5] = 99  # no baseline
        ds_indices[0, 5:10] = 0  # has baseline

        data_bt = _make_batch([resp_mask], [ds_indices])
        original_masks = [resp_mask.clone()]

        proj = torch.zeros(1, 10)
        proj[0, :] = -5.0

        metrics = pf.filter_rollouts(data_bt, original_masks, [proj])

        assert metrics["persona/baseline_miss_rate"] == 0.5  # 1 out of 2 missed
        # Rollout 1 (ds_idx=99) should NOT be filtered (no baseline)
        assert data_bt.response_masks[0][0, 0:5].all()
        # Rollout 2 (ds_idx=0) should be filtered (diff = -5.0 - 0.5 = -5.5 < 100)
        assert data_bt.response_masks[0][0, 5:10].sum() == 0

    def test_skips_on_mismatched_projections(self, tmp_artifacts):
        """Returns skip metric when projection count doesn't match mask count."""
        pf = _make_filter(tmp_artifacts, threshold=-0.5)

        resp_mask = torch.zeros(1, 10, dtype=torch.long)
        resp_mask[0, :] = 1
        ds_indices = torch.zeros(1, 10, dtype=torch.long)

        data_bt = _make_batch([resp_mask], [ds_indices])
        original_masks = [resp_mask.clone()]

        # Pass 2 projections for 1 mask
        proj1 = torch.zeros(1, 10)
        proj2 = torch.zeros(1, 10)

        metrics = pf.filter_rollouts(data_bt, original_masks, [proj1, proj2])
        assert metrics.get("persona/skipped") == 1.0

    def test_skips_when_no_dataset_indices(self, tmp_artifacts):
        """Returns skip metric when dataset_indices is None."""
        pf = _make_filter(tmp_artifacts, threshold=-0.5)

        resp_mask = torch.zeros(1, 10, dtype=torch.long)
        resp_mask[0, :] = 1

        data_bt = _make_batch([resp_mask], [torch.zeros(1, 10, dtype=torch.long)])
        data_bt.dataset_indices = None
        original_masks = [resp_mask.clone()]

        proj = torch.zeros(1, 10)
        metrics = pf.filter_rollouts(data_bt, original_masks, [proj])
        assert metrics.get("persona/skipped") == 1.0


class TestFilteredIDLogging:
    def test_writes_jsonl(self, tmp_artifacts):
        """Filtered rollout IDs are written to a jsonl file."""
        log_dir = tmp_artifacts["tmp_path"] / "logs" / "run_name"
        pf = _make_filter(tmp_artifacts, threshold=-0.5, log_dir=str(log_dir))

        resp_mask = torch.zeros(1, 10, dtype=torch.long)
        resp_mask[0, 0:5] = 1
        resp_mask[0, 5:10] = 2

        ds_indices = torch.zeros(1, 10, dtype=torch.long)
        ds_indices[0, 0:5] = 0  # baseline=0.5
        ds_indices[0, 5:10] = 1  # baseline=-0.3

        data_bt = _make_batch([resp_mask], [ds_indices])
        original_masks = [resp_mask.clone()]

        # Rollout 1: proj=-2.0, diff=-2.5 < -0.5 → filtered
        # Rollout 2: proj=0.0, diff=0.3 > -0.5 → kept
        proj = torch.zeros(1, 10)
        proj[0, 0:5] = -2.0
        proj[0, 5:10] = 0.0

        pf.filter_rollouts(data_bt, original_masks, [proj], training_step=42)

        # Check the file exists and has correct content
        log_files = list(log_dir.glob("persona_filtered_rollouts_rank*.jsonl"))
        assert len(log_files) == 1

        with open(log_files[0]) as f:
            lines = f.readlines()
        assert len(lines) == 1

        record = json.loads(lines[0])
        assert record["training_step"] == 42
        assert len(record["filtered"]) == 1
        assert record["filtered"][0]["dataset_idx"] == 0
        assert record["filtered"][0]["diff"] == pytest.approx(-2.5, abs=0.01)

    def test_no_file_when_nothing_filtered(self, tmp_artifacts):
        """No jsonl file created when no rollouts are filtered."""
        log_dir = tmp_artifacts["tmp_path"] / "logs" / "empty_run"
        pf = _make_filter(tmp_artifacts, threshold=-100.0, log_dir=str(log_dir))

        resp_mask = torch.zeros(1, 10, dtype=torch.long)
        resp_mask[0, :] = 1
        ds_indices = torch.zeros(1, 10, dtype=torch.long)

        data_bt = _make_batch([resp_mask], [ds_indices])
        original_masks = [resp_mask.clone()]
        proj = torch.zeros(1, 10)

        pf.filter_rollouts(data_bt, original_masks, [proj], training_step=1)

        log_files = list(log_dir.glob("persona_filtered_rollouts_rank*.jsonl"))
        assert len(log_files) == 0

    def test_appends_across_steps(self, tmp_artifacts):
        """Multiple steps append to the same file."""
        log_dir = tmp_artifacts["tmp_path"] / "logs" / "multi_step"
        pf = _make_filter(tmp_artifacts, threshold=-0.5, max_filter_rate=1.0, log_dir=str(log_dir))

        for step in [10, 11, 12]:
            resp_mask = torch.zeros(1, 10, dtype=torch.long)
            resp_mask[0, :5] = 1
            ds_indices = torch.zeros(1, 10, dtype=torch.long)
            ds_indices[0, :5] = 0  # baseline=0.5

            data_bt = _make_batch([resp_mask], [ds_indices])
            original_masks = [resp_mask.clone()]
            proj = torch.zeros(1, 10)
            proj[0, :5] = -2.0  # diff = -2.5 < -0.5

            pf.filter_rollouts(data_bt, original_masks, [proj], training_step=step)

        log_files = list(log_dir.glob("persona_filtered_rollouts_rank*.jsonl"))
        assert len(log_files) == 1

        with open(log_files[0]) as f:
            lines = f.readlines()
        assert len(lines) == 3
        steps = [json.loads(line)["training_step"] for line in lines]
        assert steps == [10, 11, 12]

    def test_no_log_dir(self, tmp_artifacts):
        """No crash when log_dir is None."""
        pf = _make_filter(tmp_artifacts, threshold=-0.5, log_dir=None)
        assert pf._log_dir is None

        resp_mask = torch.zeros(1, 10, dtype=torch.long)
        resp_mask[0, :5] = 1
        ds_indices = torch.zeros(1, 10, dtype=torch.long)

        data_bt = _make_batch([resp_mask], [ds_indices])
        original_masks = [resp_mask.clone()]
        proj = torch.zeros(1, 10)
        proj[0, :5] = -5.0

        # Should not crash
        metrics = pf.filter_rollouts(data_bt, original_masks, [proj], training_step=1)
        assert "persona/num_filtered" in metrics


class TestMultiSampleBatch:
    def test_multiple_samples(self, tmp_artifacts):
        """Correctly handles batches with multiple packed samples."""
        pf = _make_filter(tmp_artifacts, threshold=-0.5, max_filter_rate=1.0)

        # Sample 0: 1 rollout (ds_idx=0, baseline=0.5)
        mask0 = torch.zeros(1, 10, dtype=torch.long)
        mask0[0, :5] = 1
        ds0 = torch.zeros(1, 10, dtype=torch.long)

        # Sample 1: 1 rollout (ds_idx=1, baseline=-0.3)
        mask1 = torch.zeros(1, 10, dtype=torch.long)
        mask1[0, :5] = 1
        ds1 = torch.ones(1, 10, dtype=torch.long)

        data_bt = _make_batch([mask0, mask1], [ds0, ds1])
        original_masks = [mask0.clone(), mask1.clone()]

        # Sample 0: proj=-2.0, diff=-2.5 < -0.5 → filtered
        # Sample 1: proj=0.0, diff=0.3 > -0.5 → kept
        proj0 = torch.zeros(1, 10)
        proj0[0, :5] = -2.0
        proj1 = torch.zeros(1, 10)

        metrics = pf.filter_rollouts(data_bt, original_masks, [proj0, proj1], training_step=5)

        assert metrics["persona/num_filtered"] == 1.0
        assert metrics["persona/total_rollouts"] == 2.0
        # Sample 0 rollout filtered
        assert data_bt.response_masks[0][0, :5].sum() == 0
        # Sample 1 rollout kept
        assert data_bt.response_masks[1][0, :5].all()
