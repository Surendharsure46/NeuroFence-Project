"""Tests for architecture discovery and hook plumbing.

Synthetic nn.Modules stand in for real architectures so the layer-discovery
logic is tested against several naming conventions without downloading four
different models.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from neurofence.activation.hooks import HookManager, extract_tensor
from neurofence.core.exceptions import ActivationCaptureError, UnsupportedArchitectureError
from neurofence.model.adapter import ModelAdapter, parse_layer_spec


class FakeBlock(nn.Module):
    def __init__(self, hidden: int = 8) -> None:
        super().__init__()
        self.self_attn = nn.Linear(hidden, hidden)
        self.mlp = nn.Linear(hidden, hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.self_attn(x))


class LlamaStyle(nn.Module):
    """model.layers — Llama, Qwen, Mistral."""

    def __init__(self, layers: int = 3) -> None:
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([FakeBlock() for _ in range(layers)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.model.layers:
            x = layer(x)
        return x


class GptStyle(nn.Module):
    """transformer.h — GPT-2, Falcon."""

    def __init__(self, layers: int = 2) -> None:
        super().__init__()
        self.transformer = nn.Module()
        self.transformer.h = nn.ModuleList([FakeBlock() for _ in range(layers)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.transformer.h:
            x = layer(x)
        return x


class UnknownStyle(nn.Module):
    """A naming convention the probe list has never seen."""

    def __init__(self, layers: int = 4) -> None:
        super().__init__()
        self.weird = nn.Module()
        self.weird.stack = nn.ModuleList([FakeBlock() for _ in range(layers)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.weird.stack:
            x = layer(x)
        return x


class TestLayerDiscovery:
    def test_llama_style(self) -> None:
        adapter = ModelAdapter(LlamaStyle(3))
        assert adapter.layers_path == "model.layers"
        assert adapter.num_layers == 3

    def test_gpt_style(self) -> None:
        adapter = ModelAdapter(GptStyle(2))
        assert adapter.layers_path == "transformer.h"
        assert adapter.num_layers == 2

    def test_unknown_architecture_falls_back(self) -> None:
        """An unfamiliar architecture must still be scannable."""
        adapter = ModelAdapter(UnknownStyle(4))
        assert adapter.num_layers == 4
        assert "stack" in adapter.layers_path

    def test_no_layers_raises(self) -> None:
        with pytest.raises(UnsupportedArchitectureError):
            ModelAdapter(nn.Linear(4, 4))

    def test_summary(self) -> None:
        summary = ModelAdapter(LlamaStyle()).summary()
        assert summary["num_layers"] == 3
        assert summary["has_attention_submodule"] is True
        assert summary["has_mlp_submodule"] is True


class TestCaptureSites:
    def test_block_output_sites(self) -> None:
        adapter = ModelAdapter(LlamaStyle(3))
        sites = adapter.get_capture_sites(["block_output"])
        assert len(sites) == 3
        assert sites[0].name == "model.layers.0"

    def test_multiple_capture_points(self) -> None:
        adapter = ModelAdapter(LlamaStyle(2))
        sites = adapter.get_capture_sites(["block_output", "attention_output", "mlp_output"])
        assert len(sites) == 6
        assert {s.capture_point for s in sites} == {
            "block_output",
            "attention_output",
            "mlp_output",
        }

    def test_layer_subset(self) -> None:
        adapter = ModelAdapter(LlamaStyle(5))
        sites = adapter.get_capture_sites(["block_output"], [0, 4])
        assert [s.index for s in sites] == [0, 4]

    def test_out_of_range_index(self) -> None:
        adapter = ModelAdapter(LlamaStyle(2))
        with pytest.raises(UnsupportedArchitectureError):
            adapter.get_capture_sites(["block_output"], [9])


class TestParseLayerSpec:
    @pytest.mark.parametrize(
        ("spec", "expected"),
        [
            ("all", [0, 1, 2, 3]),
            ("*", [0, 1, 2, 3]),
            ("0", [0]),
            ("1,3", [1, 3]),
            ("0-2", [0, 1, 2]),
            ("0-1,3", [0, 1, 3]),
            ("2-0", [0, 1, 2]),  # reversed range tolerated
            ("1,1,1", [1]),  # duplicates collapsed
        ],
    )
    def test_valid_specs(self, spec: str, expected: list[int]) -> None:
        assert parse_layer_spec(spec, 4) == expected

    def test_out_of_range(self) -> None:
        with pytest.raises(UnsupportedArchitectureError, match="out of range"):
            parse_layer_spec("0-9", 4)

    def test_garbage(self) -> None:
        with pytest.raises(UnsupportedArchitectureError):
            parse_layer_spec("abc", 4)


class TestExtractTensor:
    def test_bare_tensor(self) -> None:
        tensor = torch.randn(2, 4)
        assert extract_tensor(tensor) is tensor

    def test_tuple_output(self) -> None:
        hidden = torch.randn(1, 4, 8)
        assert extract_tensor((hidden, None, "cache")) is hidden

    def test_skips_1d_entries(self) -> None:
        hidden = torch.randn(1, 4, 8)
        assert extract_tensor((torch.randn(3), hidden)) is hidden

    def test_dict_output(self) -> None:
        hidden = torch.randn(1, 4, 8)
        assert extract_tensor({"last_hidden_state": hidden}) is hidden

    def test_none_for_unusable(self) -> None:
        assert extract_tensor("not a tensor") is None
        assert extract_tensor((None, None)) is None


class TestHookManager:
    def test_captures_every_layer(self) -> None:
        model = LlamaStyle(3)
        adapter = ModelAdapter(model)
        sites = adapter.get_capture_sites(["block_output"])
        captured: list[str] = []

        with HookManager(sites, lambda site, tensor: captured.append(site.name)):
            model(torch.randn(1, 4, 8))

        assert captured == ["model.layers.0", "model.layers.1", "model.layers.2"]

    def test_handles_removed_on_exit(self) -> None:
        model = LlamaStyle(2)
        sites = ModelAdapter(model).get_capture_sites(["block_output"])
        calls: list[str] = []

        with HookManager(sites, lambda site, tensor: calls.append(site.name)):
            model(torch.randn(1, 4, 8))
        before = len(calls)

        model(torch.randn(1, 4, 8))  # hooks should be gone
        assert len(calls) == before

    def test_handles_removed_after_exception(self) -> None:
        model = LlamaStyle(2)
        sites = ModelAdapter(model).get_capture_sites(["block_output"])
        calls: list[str] = []

        with pytest.raises(RuntimeError):
            with HookManager(sites, lambda site, tensor: calls.append(site.name)):
                raise RuntimeError("forward failed")

        model(torch.randn(1, 4, 8))
        assert calls == []

    def test_callback_error_does_not_kill_run(self) -> None:
        model = LlamaStyle(2)
        sites = ModelAdapter(model).get_capture_sites(["block_output"])

        def bad(site, tensor):
            raise ValueError("bad callback")

        with HookManager(sites, bad) as manager:
            model(torch.randn(1, 4, 8))

        assert len(manager.stats.errors) == 2
        assert manager.stats.fired == 0

    def test_empty_sites_rejected(self) -> None:
        with pytest.raises(ActivationCaptureError):
            HookManager([], lambda site, tensor: None)

    def test_stats_counts(self) -> None:
        model = LlamaStyle(3)
        sites = ModelAdapter(model).get_capture_sites(["block_output"])
        with HookManager(sites, lambda site, tensor: None) as manager:
            model(torch.randn(1, 4, 8))
        assert manager.stats.registered == 3
        assert manager.stats.fired == 3
