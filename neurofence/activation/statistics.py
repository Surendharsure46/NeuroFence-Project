"""Activation statistics.

Phase 1 computes descriptive statistics only — no detection, no thresholds, no
verdicts. The point is to produce a numerically trustworthy substrate that
Phase 2's detector can reason over.

Design notes:

* Everything is computed in float64 on CPU. Activation magnitudes in float16
  overflow when squared, and a silently-inf variance would look exactly like an
  anomaly.
* Statistics accumulate in streaming fashion (count, sums of powers) so a run
  over many prompts never holds all activations in memory.
* Per-channel maxima are tracked because the published weight-poisoning
  literature repeatedly finds backdoors expressed as a handful of extreme
  channels rather than a shift in the global mean. Recording only aggregate
  stats would average the signal away.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from ..core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ActivationStats:
    """Streaming accumulator for one capture site."""

    site_name: str
    layer_index: int
    capture_point: str

    count: int = 0  # scalar elements seen
    tokens: int = 0  # token positions seen
    hidden_size: int | None = None
    dtype: str | None = None
    shape: tuple[int, ...] | None = None

    _sum: float = 0.0
    _sum_sq: float = 0.0
    _sum_cube: float = 0.0
    _sum_quad: float = 0.0
    _sum_abs: float = 0.0
    _zero_count: int = 0

    minimum: float = math.inf
    maximum: float = -math.inf
    max_abs: float = 0.0
    _l2_sq_per_token_sum: float = 0.0

    _channel_max_abs: Any = None  # torch.Tensor[hidden_size], float64
    _channel_sum: Any = None

    invocations: int = 0

    # --- accumulation -----------------------------------------------------

    def update(self, tensor: Any) -> None:
        """Fold one captured tensor into the running statistics."""
        import torch

        flat = tensor.detach().to(torch.float64).cpu()
        if flat.numel() == 0:
            return

        if self.shape is None:
            self.shape = tuple(tensor.shape)
            self.dtype = str(tensor.dtype).replace("torch.", "")
        self.invocations += 1

        # Collapse everything except the final (hidden) dimension.
        hidden = flat.shape[-1]
        matrix = flat.reshape(-1, hidden)
        if self.hidden_size is None:
            self.hidden_size = int(hidden)
            self._channel_max_abs = torch.zeros(hidden, dtype=torch.float64)
            self._channel_sum = torch.zeros(hidden, dtype=torch.float64)
        elif self.hidden_size != hidden:
            logger.warning(
                "hidden size changed between invocations; skipping tensor",
                extra={"extra_fields": {"site": self.site_name, "expected": self.hidden_size}},
            )
            return

        absolute = matrix.abs()
        self.count += matrix.numel()
        self.tokens += matrix.shape[0]
        self._sum += float(matrix.sum())
        self._sum_sq += float((matrix**2).sum())
        self._sum_cube += float((matrix**3).sum())
        self._sum_quad += float((matrix**4).sum())
        self._sum_abs += float(absolute.sum())
        self._zero_count += int((matrix == 0).sum())

        self.minimum = min(self.minimum, float(matrix.min()))
        self.maximum = max(self.maximum, float(matrix.max()))
        self.max_abs = max(self.max_abs, float(absolute.max()))
        self._l2_sq_per_token_sum += float((matrix**2).sum(dim=-1).sqrt().sum())

        self._channel_max_abs = torch.maximum(self._channel_max_abs, absolute.max(dim=0).values)
        self._channel_sum += matrix.sum(dim=0)

    # --- derived quantities ----------------------------------------------

    @property
    def mean(self) -> float | None:
        return self._sum / self.count if self.count else None

    @property
    def variance(self) -> float | None:
        """Population variance, clamped at zero against float cancellation."""
        if self.count < 2:
            return None
        value = self._sum_sq / self.count - (self._sum / self.count) ** 2
        return max(value, 0.0)

    @property
    def std(self) -> float | None:
        variance = self.variance
        return math.sqrt(variance) if variance is not None else None

    @property
    def mean_abs(self) -> float | None:
        return self._sum_abs / self.count if self.count else None

    @property
    def l2_norm(self) -> float | None:
        """L2 norm over all captured elements."""
        return math.sqrt(self._sum_sq) if self.count else None

    @property
    def mean_token_l2(self) -> float | None:
        """Mean per-token L2 norm — comparable across sequence lengths."""
        return self._l2_sq_per_token_sum / self.tokens if self.tokens else None

    @property
    def rms(self) -> float | None:
        return math.sqrt(self._sum_sq / self.count) if self.count else None

    @property
    def skewness(self) -> float | None:
        """Fisher-Pearson skewness from raw moments."""
        std = self.std
        if not std or self.count < 3:
            return None
        mean = self._sum / self.count
        m3 = self._sum_cube / self.count - 3 * mean * (self._sum_sq / self.count) + 2 * mean**3
        return m3 / (std**3)

    @property
    def kurtosis(self) -> float | None:
        """Excess kurtosis. Heavy tails are the classic backdoor-activation tell."""
        std = self.std
        if not std or self.count < 4:
            return None
        mean = self._sum / self.count
        m4 = (
            self._sum_quad / self.count
            - 4 * mean * (self._sum_cube / self.count)
            + 6 * mean**2 * (self._sum_sq / self.count)
            - 3 * mean**4
        )
        return m4 / (std**4) - 3.0

    @property
    def sparsity(self) -> float | None:
        """Fraction of exactly-zero elements."""
        return self._zero_count / self.count if self.count else None

    def top_channels(self, k: int = 5) -> list[dict[str, float]]:
        """The k channels with the largest absolute activation seen."""
        if self._channel_max_abs is None or k <= 0:
            return []
        import torch

        k = min(k, int(self._channel_max_abs.numel()))
        values, indices = torch.topk(self._channel_max_abs, k)
        mean_per_channel = self._channel_sum / self.tokens if self.tokens else None
        results = []
        for value, index in zip(values.tolist(), indices.tolist(), strict=True):
            entry = {"channel": int(index), "max_abs": float(value)}
            if mean_per_channel is not None:
                entry["mean"] = float(mean_per_channel[index])
            results.append(entry)
        return results

    def outlier_ratio(self) -> float | None:
        """max_abs / rms — a scale-free measure of how extreme the peak is."""
        rms = self.rms
        if not rms:
            return None
        return self.max_abs / rms

    # --- serialisation ----------------------------------------------------

    def to_dict(self, top_k: int = 5) -> dict[str, Any]:
        return {
            "site_name": self.site_name,
            "layer_index": self.layer_index,
            "capture_point": self.capture_point,
            "dtype": self.dtype,
            "shape": list(self.shape) if self.shape else None,
            "hidden_size": self.hidden_size,
            "invocations": self.invocations,
            "elements": self.count,
            "tokens": self.tokens,
            "mean": _clean(self.mean),
            "std": _clean(self.std),
            "variance": _clean(self.variance),
            "min": _clean(None if self.minimum == math.inf else self.minimum),
            "max": _clean(None if self.maximum == -math.inf else self.maximum),
            "max_abs": _clean(self.max_abs),
            "mean_abs": _clean(self.mean_abs),
            "rms": _clean(self.rms),
            "l2_norm": _clean(self.l2_norm),
            "mean_token_l2": _clean(self.mean_token_l2),
            "skewness": _clean(self.skewness),
            "kurtosis": _clean(self.kurtosis),
            "sparsity": _clean(self.sparsity),
            "outlier_ratio": _clean(self.outlier_ratio()),
            "top_channels": self.top_channels(top_k),
        }


@dataclass
class StatsTable:
    """All capture sites for one run."""

    stats: dict[str, ActivationStats] = field(default_factory=dict)

    def get_or_create(
        self, site_name: str, layer_index: int, capture_point: str
    ) -> ActivationStats:
        if site_name not in self.stats:
            self.stats[site_name] = ActivationStats(
                site_name=site_name, layer_index=layer_index, capture_point=capture_point
            )
        return self.stats[site_name]

    def rows(self, top_k: int = 5) -> list[dict[str, Any]]:
        return [
            entry.to_dict(top_k)
            for entry in sorted(
                self.stats.values(), key=lambda s: (s.layer_index, s.capture_point)
            )
        ]

    def __len__(self) -> int:
        return len(self.stats)


# Columns written to CSV, in order. Nested values (top_channels) are JSON-encoded.
CSV_COLUMNS = [
    "site_name",
    "layer_index",
    "capture_point",
    "dtype",
    "hidden_size",
    "invocations",
    "elements",
    "tokens",
    "mean",
    "std",
    "variance",
    "min",
    "max",
    "max_abs",
    "mean_abs",
    "rms",
    "l2_norm",
    "mean_token_l2",
    "skewness",
    "kurtosis",
    "sparsity",
    "outlier_ratio",
]


def _clean(value: float | None) -> float | None:
    """Convert non-finite floats to None so the JSON stays strictly valid."""
    if value is None:
        return None
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return float(value)
