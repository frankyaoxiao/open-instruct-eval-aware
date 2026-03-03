"""Tests for persona vector-based rollout filtering."""

import tempfile

import torch

from open_instruct import data_types
from open_instruct.persona_filter import PersonaFilter, PersonaFilterConfig


def _make_filter(baselines: dict[int, float], threshold: float = 2.0, hidden_dim: int = 16) -> PersonaFilter:
    """Create a PersonaFilter with a dummy persona vector and given baselines."""
    with tempfile.NamedTemporaryFile(suffix=".pt") as vec_f, tempfile.NamedTemporaryFile(suffix=".pt") as base_f:
        # Random unit vector
        vec = torch.randn(hidden_dim, dtype=torch.bfloat16)
        vec = vec / vec.norm()
        torch.save(vec, vec_f.name)
        torch.save(baselines, base_f.name)

        config = PersonaFilterConfig(
            vector_path=vec_f.name,
            baseline_path=base_f.name,
            layer_idx=0,
            threshold=threshold,
            max_filter_rate=0.5,
        )
        return PersonaFilter(config, device=torch.device("cpu"))


def _make_packed_data(
    seq_len: int,
    rollout_specs: list[tuple[int, int, int, int]],
    dataset_index_map: dict[int, int],
) -> tuple[data_types.CollatedBatchData, list[torch.Tensor]]:
    """Build a single-sample CollatedBatchData with one packed sequence.

    Args:
        seq_len: Total sequence length.
        rollout_specs: List of (rollout_id, query_start, query_end, resp_start, resp_end)
            defining token ranges for each rollout.
        dataset_index_map: Maps rollout_id -> dataset row index.

    Returns:
        (CollatedBatchData, original_response_masks) ready for filter_rollouts.
    """
    response_mask = torch.zeros(1, seq_len, dtype=torch.long)
    ds_indices = torch.zeros(1, seq_len, dtype=torch.long)

    for rid, _qs, _qe, rs, re in rollout_specs:
        response_mask[0, rs:re] = rid
        ds_indices[0, rs:re] = dataset_index_map[rid]

    original_response_masks = [response_mask.clone()]
    bool_response_masks = [response_mask.bool()]

    data_bt = data_types.CollatedBatchData(
        query_responses=[torch.zeros(1, seq_len, dtype=torch.long)],
        attention_masks=[torch.ones(1, seq_len, dtype=torch.long)],
        position_ids=[torch.arange(seq_len).unsqueeze(0)],
        advantages=[torch.zeros(1, seq_len)],
        response_masks=bool_response_masks,
        vllm_logprobs=[torch.zeros(1, seq_len)],
        dataset_indices=[ds_indices],
    )
    return data_bt, original_response_masks


class TestFilterRollouts:
    def test_no_filtering_when_within_threshold(self):
        """Rollouts within threshold should not be filtered."""
        baselines = {42: 1.0, 43: 1.0}
        pf = _make_filter(baselines, threshold=5.0)

        # 2 rollouts, both from different prompts
        # rollout_specs: (rollout_id, query_start, query_end, resp_start, resp_end)
        data_bt, orig_masks = _make_packed_data(
            seq_len=20,
            rollout_specs=[(1, 0, 5, 5, 10), (2, 10, 15, 15, 20)],
            dataset_index_map={1: 42, 2: 43},
        )

        # Projections close to baseline (1.0)
        projections = [torch.full((1, 20), 1.5)]

        metrics = pf.filter_rollouts(data_bt, orig_masks, projections)

        assert metrics["persona/total_rollouts"] == 2
        assert metrics["persona/num_filtered"] == 0
        # All response tokens should still be True
        assert data_bt.response_masks[0][0, 5:10].all()
        assert data_bt.response_masks[0][0, 15:20].all()

    def test_filters_rollout_exceeding_threshold(self):
        """Rollouts exceeding threshold should be zeroed out."""
        baselines = {42: 0.0, 43: 0.0}
        pf = _make_filter(baselines, threshold=1.0)

        data_bt, orig_masks = _make_packed_data(
            seq_len=20,
            rollout_specs=[(1, 0, 5, 5, 10), (2, 10, 15, 15, 20)],
            dataset_index_map={1: 42, 2: 43},
        )

        # Rollout 1: projection = 0.5 (within threshold)
        # Rollout 2: projection = 5.0 (exceeds threshold)
        proj = torch.zeros(1, 20)
        proj[0, 5:10] = 0.5
        proj[0, 15:20] = 5.0
        projections = [proj]

        metrics = pf.filter_rollouts(data_bt, orig_masks, projections)

        assert metrics["persona/num_filtered"] == 1
        # Rollout 1 should be kept
        assert data_bt.response_masks[0][0, 5:10].all()
        # Rollout 2 should be zeroed out
        assert not data_bt.response_masks[0][0, 15:20].any()

    def test_baseline_miss(self):
        """Rollouts without baselines should be counted as misses, not filtered."""
        baselines = {42: 0.0}  # Only baseline for prompt 42, not 99
        pf = _make_filter(baselines, threshold=1.0)

        data_bt, orig_masks = _make_packed_data(
            seq_len=20,
            rollout_specs=[(1, 0, 5, 5, 10), (2, 10, 15, 15, 20)],
            dataset_index_map={1: 42, 2: 99},  # 99 has no baseline
        )

        proj = torch.zeros(1, 20)
        proj[0, 5:10] = 5.0  # Exceeds threshold for prompt 42
        proj[0, 15:20] = 5.0  # Would exceed but no baseline
        projections = [proj]

        metrics = pf.filter_rollouts(data_bt, orig_masks, projections)

        assert metrics["persona/num_filtered"] == 1  # Only rollout 1
        assert metrics["persona/baseline_miss_rate"] == 0.5  # 1 out of 2
        assert not data_bt.response_masks[0][0, 5:10].any()  # Filtered
        assert data_bt.response_masks[0][0, 15:20].all()  # Kept (no baseline)

    def test_max_filter_rate(self):
        """Should not filter more than max_filter_rate of rollouts."""
        baselines = {42: 0.0, 43: 0.0, 44: 0.0, 45: 0.0}
        pf = _make_filter(baselines, threshold=1.0)
        pf.max_filter_rate = 0.25  # At most 1 out of 4

        data_bt, orig_masks = _make_packed_data(
            seq_len=40,
            rollout_specs=[
                (1, 0, 5, 5, 10),
                (2, 10, 15, 15, 20),
                (3, 20, 25, 25, 30),
                (4, 30, 35, 35, 40),
            ],
            dataset_index_map={1: 42, 2: 43, 3: 44, 4: 45},
        )

        # All exceed threshold, but rollout 3 has the largest deviation
        proj = torch.zeros(1, 40)
        proj[0, 5:10] = 2.0    # abs_diff = 2.0
        proj[0, 15:20] = 3.0   # abs_diff = 3.0
        proj[0, 25:30] = 10.0  # abs_diff = 10.0 (worst)
        proj[0, 35:40] = 4.0   # abs_diff = 4.0
        projections = [proj]

        metrics = pf.filter_rollouts(data_bt, orig_masks, projections)

        # Only 1 should be filtered (max_filter_rate=0.25, 4 rollouts -> max 1)
        assert metrics["persona/num_filtered"] == 1
        # The worst offender (rollout 3) should be the one filtered
        assert data_bt.response_masks[0][0, 25:30].sum() == 0

    def test_skips_when_projection_count_mismatch(self):
        """Should skip filtering when projection count doesn't match."""
        baselines = {42: 0.0}
        pf = _make_filter(baselines, threshold=1.0)

        data_bt, orig_masks = _make_packed_data(
            seq_len=10,
            rollout_specs=[(1, 0, 3, 3, 10)],
            dataset_index_map={1: 42},
        )

        # Wrong number of projections
        projections = [torch.zeros(1, 10), torch.zeros(1, 10)]

        metrics = pf.filter_rollouts(data_bt, orig_masks, projections)
        assert metrics.get("persona/skipped") == 1.0

    def test_skips_when_no_dataset_indices(self):
        """Should skip filtering when dataset_indices is None."""
        baselines = {42: 0.0}
        pf = _make_filter(baselines, threshold=1.0)

        data_bt = data_types.CollatedBatchData(
            query_responses=[torch.zeros(1, 10, dtype=torch.long)],
            attention_masks=[torch.ones(1, 10, dtype=torch.long)],
            position_ids=[torch.arange(10).unsqueeze(0)],
            advantages=[torch.zeros(1, 10)],
            response_masks=[torch.ones(1, 10, dtype=torch.bool)],
            vllm_logprobs=[torch.zeros(1, 10)],
            dataset_indices=None,
        )
        orig_masks = [torch.ones(1, 10, dtype=torch.long)]
        projections = [torch.zeros(1, 10)]

        metrics = pf.filter_rollouts(data_bt, orig_masks, projections)
        assert metrics.get("persona/skipped") == 1.0


class TestHookCapture:
    def test_hook_captures_projections(self):
        """Test that the hook correctly captures and projects hidden states."""
        baselines = {0: 0.0}
        hidden_dim = 32
        pf = _make_filter(baselines, hidden_dim=hidden_dim)

        # Create a simple model with one layer
        layer = torch.nn.Linear(hidden_dim, hidden_dim).to(torch.bfloat16)
        model_mock = torch.nn.Module()
        model_mock.model = torch.nn.Module()
        model_mock.model.layers = torch.nn.ModuleList([layer])

        pf.register_hook(model_mock)

        # Run a forward pass with capture enabled
        pf.begin_capture()
        x = torch.randn(1, 5, hidden_dim, dtype=torch.bfloat16)
        layer(x)  # This triggers the hook
        captured = pf.end_capture()

        assert len(captured) == 1
        assert captured[0].shape == (1, 5)  # (B, T) projections

        pf.remove_hook()

    def test_no_capture_when_not_started(self):
        """Hook should not accumulate when capture hasn't been started."""
        baselines = {0: 0.0}
        hidden_dim = 32
        pf = _make_filter(baselines, hidden_dim=hidden_dim)

        layer = torch.nn.Linear(hidden_dim, hidden_dim).to(torch.bfloat16)
        model_mock = torch.nn.Module()
        model_mock.model = torch.nn.Module()
        model_mock.model.layers = torch.nn.ModuleList([layer])

        pf.register_hook(model_mock)

        # Forward pass WITHOUT begin_capture
        x = torch.randn(1, 5, hidden_dim, dtype=torch.bfloat16)
        layer(x)

        # Nothing should be captured
        result = pf.end_capture()
        assert len(result) == 0

        pf.remove_hook()
