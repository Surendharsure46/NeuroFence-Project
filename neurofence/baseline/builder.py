"""Baseline construction.

A baseline is a per-layer, per-metric distribution built from NORMAL prompts.
Everything downstream is measured against it, so its quality caps the quality
of every conclusion the tool reaches.

The unit of observation is **one prompt**, not one tensor element. For each
prompt we reduce that layer's activations to a handful of scalars (mean,
max_abs, rms, kurtosis, ...) and the baseline is the distribution of those
scalars across prompts. This matters: pooling raw elements would give a
distribution with millions of samples and a vanishingly small standard
deviation, against which *every* input scores as wildly anomalous. The z-scores
would look impressive and mean nothing.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from ..core.logging import get_logger

logger = get_logger(__name__)

#: Scalars extracted per prompt per layer. Each answers a different question:
#: overall shift (mean), peak magnitude (max_abs), energy (rms), tail weight
#: (kurtosis), and per-token scale (mean_token_l2).
BASELINE_METRICS: tuple[str, ...] = (
    "mean",
    "max_abs",
    "rms",
    "kurtosis",
    "mean_token_l2",
)

#: Minimum prompts for a distribution worth calling a baseline. Below this the
#: percentiles are meaningless and the standard deviation is unstable.
MIN_BASELINE_SAMPLES = 20


@dataclass
class MetricDistribution:
    """Distribution of one scalar metric at one layer, across baseline prompts."""

    metric: str
    samples: list[float] = field(default_factory=list)

    # Computed in finalise()
    mean: float | None = None
    std: float | None = None
    median: float | None = None
    mad: float | None = None  # median absolute deviation (robust scale)
    p95: float | None = None
    p99: float | None = None
    minimum: float | None = None
    maximum: float | None = None

    @property
    def n(self) -> int:
        return len(self.samples)

    def add(self, value: float | None) -> None:
        if value is not None and math.isfinite(value):
            self.samples.append(float(value))

    def finalise(self) -> None:
        """Compute summary statistics. Idempotent."""
        if not self.samples:
            return
        ordered = sorted(self.samples)
        n = len(ordered)
        self.mean = sum(ordered) / n
        self.median = _percentile(ordered, 50.0)
        self.minimum = ordered[0]
        self.maximum = ordered[-1]
        if n >= 2:
            variance = sum((x - self.mean) ** 2 for x in ordered) / (n - 1)
            self.std = math.sqrt(max(variance, 0.0))
        else:
            self.std = 0.0
        deviations = sorted(abs(x - self.median) for x in ordered)
        self.mad = _percentile(deviations, 50.0)
        # Percentiles below ~20 samples are interpolation fiction; report them
        # only where there is enough data to mean something.
        if n >= MIN_BASELINE_SAMPLES:
            self.p95 = _percentile(ordered, 95.0)
            self.p99 = _percentile(ordered, 99.0)

    def to_dict(self, include_samples: bool = False) -> dict[str, Any]:
        payload = {
            "metric": self.metric,
            "n": self.n,
            "mean": self.mean,
            "std": self.std,
            "median": self.median,
            "mad": self.mad,
            "p95": self.p95,
            "p99": self.p99,
            "min": self.minimum,
            "max": self.maximum,
        }
        if include_samples:
            payload["samples"] = list(self.samples)
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MetricDistribution:
        dist = cls(metric=data["metric"], samples=list(data.get("samples", [])))
        for key, attribute in (
            ("mean", "mean"),
            ("std", "std"),
            ("median", "median"),
            ("mad", "mad"),
            ("p95", "p95"),
            ("p99", "p99"),
            ("min", "minimum"),
            ("max", "maximum"),
        ):
            setattr(dist, attribute, data.get(key))
        dist._n_override = data.get("n", len(dist.samples))
        return dist


@dataclass
class LayerBaseline:
    """All metric distributions for one capture site."""

    site_name: str
    layer_index: int
    capture_point: str
    metrics: dict[str, MetricDistribution] = field(default_factory=dict)

    def distribution(self, metric: str) -> MetricDistribution | None:
        return self.metrics.get(metric)

    def to_dict(self, include_samples: bool = False) -> dict[str, Any]:
        return {
            "site_name": self.site_name,
            "layer_index": self.layer_index,
            "capture_point": self.capture_point,
            "metrics": {
                name: dist.to_dict(include_samples) for name, dist in sorted(self.metrics.items())
            },
        }


@dataclass
class Baseline:
    """The complete baseline for one model."""

    model_name: str = ""
    model_sha256: str = ""
    layers: dict[str, LayerBaseline] = field(default_factory=dict)
    metrics: list[str] = field(default_factory=lambda: list(BASELINE_METRICS))
    prompt_count: int = 0
    prompt_ids: list[str] = field(default_factory=list)
    seed: int = 0
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat(timespec="seconds"))

    @property
    def is_usable(self) -> bool:
        """Whether the baseline has enough samples to support percentiles."""
        return self.prompt_count >= MIN_BASELINE_SAMPLES and bool(self.layers)

    @property
    def warnings(self) -> list[str]:
        issues: list[str] = []
        if self.prompt_count < MIN_BASELINE_SAMPLES:
            issues.append(
                f"baseline built from {self.prompt_count} prompts; "
                f"{MIN_BASELINE_SAMPLES}+ recommended before trusting z-scores"
            )
        for layer in self.layers.values():
            for dist in layer.metrics.values():
                if dist.std is not None and dist.std == 0.0:
                    issues.append(
                        f"{layer.site_name}/{dist.metric}: zero variance in baseline; "
                        "z-scores for this metric are undefined and will be suppressed"
                    )
        return issues

    def layer(self, site_name: str) -> LayerBaseline | None:
        return self.layers.get(site_name)

    def to_dict(self, include_samples: bool = False) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "model_sha256": self.model_sha256,
            "created_at": self.created_at,
            "seed": self.seed,
            "prompt_count": self.prompt_count,
            "prompt_ids": list(self.prompt_ids),
            "metrics": list(self.metrics),
            "is_usable": self.is_usable,
            "warnings": self.warnings,
            "layers": {
                name: layer.to_dict(include_samples) for name, layer in sorted(self.layers.items())
            },
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Baseline:
        baseline = cls(
            model_name=data.get("model_name", ""),
            model_sha256=data.get("model_sha256", ""),
            metrics=list(data.get("metrics", BASELINE_METRICS)),
            prompt_count=int(data.get("prompt_count", 0)),
            prompt_ids=list(data.get("prompt_ids", [])),
            seed=int(data.get("seed", 0)),
        )
        baseline.created_at = data.get("created_at", baseline.created_at)
        for name, layer_data in data.get("layers", {}).items():
            layer = LayerBaseline(
                site_name=layer_data["site_name"],
                layer_index=int(layer_data["layer_index"]),
                capture_point=layer_data["capture_point"],
            )
            for metric_name, dist_data in layer_data.get("metrics", {}).items():
                layer.metrics[metric_name] = MetricDistribution.from_dict(dist_data)
            baseline.layers[name] = layer
        return baseline


class BaselineBuilder:
    """Builds a :class:`Baseline` from per-prompt activation statistics."""

    def __init__(self, metrics: list[str] | None = None) -> None:
        self.metrics = list(metrics or BASELINE_METRICS)

    def build(
        self,
        prompt_rows: list[tuple[str, list[dict[str, Any]]]],
        *,
        model_name: str = "",
        model_sha256: str = "",
        seed: int = 0,
    ) -> Baseline:
        """Build a baseline from ``(prompt_id, stats_rows)`` pairs.

        ``stats_rows`` is the output of ``StatsTable.rows()`` for a single
        prompt — one row per capture site.
        """
        baseline = Baseline(
            model_name=model_name,
            model_sha256=model_sha256,
            metrics=list(self.metrics),
            seed=seed,
        )
        for prompt_id, rows in prompt_rows:
            baseline.prompt_ids.append(prompt_id)
            for row in rows:
                site = row["site_name"]
                if site not in baseline.layers:
                    baseline.layers[site] = LayerBaseline(
                        site_name=site,
                        layer_index=int(row["layer_index"]),
                        capture_point=row["capture_point"],
                    )
                layer = baseline.layers[site]
                for metric in self.metrics:
                    if metric not in layer.metrics:
                        layer.metrics[metric] = MetricDistribution(metric=metric)
                    layer.metrics[metric].add(row.get(metric))
        baseline.prompt_count = len(baseline.prompt_ids)

        for layer in baseline.layers.values():
            for dist in layer.metrics.values():
                dist.finalise()

        for warning in baseline.warnings:
            logger.warning("baseline: %s", warning)
        logger.info(
            "baseline built",
            extra={
                "extra_fields": {
                    "prompts": baseline.prompt_count,
                    "layers": len(baseline.layers),
                    "metrics": ",".join(self.metrics),
                    "usable": baseline.is_usable,
                }
            },
        )
        return baseline


def _percentile(ordered: list[float], percent: float) -> float:
    """Linear-interpolation percentile over a pre-sorted list."""
    if not ordered:
        return 0.0
    if len(ordered) == 1:
        return ordered[0]
    rank = (percent / 100.0) * (len(ordered) - 1)
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[int(rank)]
    weight = rank - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight
