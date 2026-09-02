"""Tests for activation statistics.

These check numerical correctness against values computed independently (torch
reductions over the same data), not just "the code runs". A statistics module
that is subtly wrong is worse than no statistics module, because Phase 2 will
draw conclusions from it.
"""

from __future__ import annotations

import math

import pytest
import torch

from neurofence.activation.statistics import ActivationStats, StatsTable


def make_stats(name: str = "layer.0") -> ActivationStats:
    return ActivationStats(site_name=name, layer_index=0, capture_point="block_output")


class TestBasicMoments:
    def test_mean_and_std_match_torch(self) -> None:
        torch.manual_seed(0)
        tensor = torch.randn(4, 16, 32)
        stats = make_stats()
        stats.update(tensor)

        flat = tensor.to(torch.float64).flatten()
        assert stats.mean == pytest.approx(float(flat.mean()), abs=1e-9)
        assert stats.std == pytest.approx(float(flat.std(unbiased=False)), abs=1e-8)
        assert stats.minimum == pytest.approx(float(flat.min()))
        assert stats.maximum == pytest.approx(float(flat.max()))
        assert stats.max_abs == pytest.approx(float(flat.abs().max()))

    def test_streaming_equals_single_pass(self) -> None:
        """Chunked accumulation must equal one-shot accumulation."""
        torch.manual_seed(1)
        chunks = [torch.randn(2, 8, 16) for _ in range(5)]

        streamed = make_stats()
        for chunk in chunks:
            streamed.update(chunk)

        combined = make_stats()
        combined.update(torch.cat(chunks, dim=0))

        assert streamed.mean == pytest.approx(combined.mean, abs=1e-9)
        assert streamed.std == pytest.approx(combined.std, abs=1e-8)
        assert streamed.count == combined.count
        assert streamed.max_abs == pytest.approx(combined.max_abs)

    def test_known_values(self) -> None:
        stats = make_stats()
        stats.update(torch.tensor([[[1.0, 2.0, 3.0, 4.0]]]))
        assert stats.count == 4
        assert stats.mean == pytest.approx(2.5)
        assert stats.variance == pytest.approx(1.25)
        assert stats.std == pytest.approx(math.sqrt(1.25))
        assert stats.rms == pytest.approx(math.sqrt(7.5))
        assert stats.l2_norm == pytest.approx(math.sqrt(30.0))

    def test_constant_tensor_has_zero_variance(self) -> None:
        stats = make_stats()
        stats.update(torch.full((2, 4, 8), 3.0))
        assert stats.variance == pytest.approx(0.0, abs=1e-12)
        assert stats.variance >= 0.0  # clamped, never negative from cancellation
        assert stats.skewness is None or stats.skewness == 0.0


class TestHigherMoments:
    def test_kurtosis_of_gaussian_near_zero(self) -> None:
        torch.manual_seed(2)
        stats = make_stats()
        stats.update(torch.randn(1, 2000, 64))
        assert stats.kurtosis == pytest.approx(0.0, abs=0.2)

    def test_kurtosis_detects_heavy_tail(self) -> None:
        """A planted spike must raise kurtosis well above the Gaussian baseline."""
        torch.manual_seed(3)
        clean = torch.randn(1, 500, 32)
        spiked = clean.clone()
        spiked[0, 0, 0] = 200.0

        baseline, poisoned = make_stats(), make_stats("layer.1")
        baseline.update(clean)
        poisoned.update(spiked)

        assert poisoned.kurtosis > baseline.kurtosis * 5
        assert poisoned.outlier_ratio() > baseline.outlier_ratio() * 5

    def test_skewness_sign(self) -> None:
        stats = make_stats()
        stats.update(torch.tensor([[[0.0, 0.0, 0.0, 0.0, 10.0]]]))
        assert stats.skewness > 0


class TestChannelTracking:
    def test_top_channels_identifies_planted_channel(self) -> None:
        torch.manual_seed(4)
        tensor = torch.randn(1, 64, 16) * 0.1
        tensor[..., 7] = 50.0

        stats = make_stats()
        stats.update(tensor)
        top = stats.top_channels(3)

        assert top[0]["channel"] == 7
        assert top[0]["max_abs"] == pytest.approx(50.0, abs=1e-4)
        assert len(top) == 3

    def test_top_k_clamped_to_hidden_size(self) -> None:
        stats = make_stats()
        stats.update(torch.randn(1, 4, 3))
        assert len(stats.top_channels(10)) == 3

    def test_zero_k(self) -> None:
        stats = make_stats()
        stats.update(torch.randn(1, 4, 8))
        assert stats.top_channels(0) == []


class TestEdgeCases:
    def test_empty_tensor_ignored(self) -> None:
        stats = make_stats()
        stats.update(torch.empty(0, 8))
        assert stats.count == 0
        assert stats.mean is None

    def test_no_data_returns_none_not_nan(self) -> None:
        stats = make_stats()
        for value in (stats.mean, stats.std, stats.kurtosis, stats.l2_norm, stats.sparsity):
            assert value is None

    def test_float16_input_does_not_overflow(self) -> None:
        """Squaring large fp16 values overflows in fp16; we promote to fp64."""
        stats = make_stats()
        stats.update(torch.full((1, 4, 8), 300.0, dtype=torch.float16))
        assert stats.rms is not None
        assert math.isfinite(stats.rms)
        assert stats.rms == pytest.approx(300.0, abs=1.0)

    def test_hidden_size_mismatch_skipped(self) -> None:
        stats = make_stats()
        stats.update(torch.randn(1, 4, 16))
        before = stats.count
        stats.update(torch.randn(1, 4, 8))  # wrong hidden size
        assert stats.count == before

    def test_sparsity(self) -> None:
        stats = make_stats()
        stats.update(torch.tensor([[[0.0, 0.0, 1.0, 2.0]]]))
        assert stats.sparsity == pytest.approx(0.5)

    def test_dict_output_is_json_safe(self) -> None:
        import json

        stats = make_stats()
        stats.update(torch.randn(1, 4, 8))
        payload = json.dumps(stats.to_dict())
        assert "NaN" not in payload
        assert "Infinity" not in payload


class TestStatsTable:
    def test_get_or_create_is_idempotent(self) -> None:
        table = StatsTable()
        first = table.get_or_create("a", 0, "block_output")
        second = table.get_or_create("a", 0, "block_output")
        assert first is second
        assert len(table) == 1

    def test_rows_sorted_by_layer(self) -> None:
        table = StatsTable()
        for index in (2, 0, 1):
            entry = table.get_or_create(f"layer.{index}", index, "block_output")
            entry.update(torch.randn(1, 2, 4))
        assert [row["layer_index"] for row in table.rows()] == [0, 1, 2]
