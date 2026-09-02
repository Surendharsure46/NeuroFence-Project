"""Baseline quality analysis.

A detector built on a bad baseline produces confident nonsense, so the baseline
gets audited before anything is measured against it. This module reports on
sample sufficiency, degenerate (zero-variance) metrics, and the natural spread
of the baseline itself.

That last one is the useful number: the **self-consistency band**. By scoring
baseline prompts against their own distribution (leave-one-out), we learn how
anomalous ordinary in-distribution text already looks. A trigger effect that
does not exceed that band is not a finding.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from ..core.logging import get_logger
from .builder import MIN_BASELINE_SAMPLES, Baseline

logger = get_logger(__name__)


@dataclass
class BaselineQuality:
    """Audit result for one baseline."""

    prompt_count: int
    layer_count: int
    metric_count: int
    usable: bool
    degenerate_metrics: list[str] = field(default_factory=list)
    sparse_metrics: list[str] = field(default_factory=list)
    self_consistency: dict[str, float] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt_count": self.prompt_count,
            "layer_count": self.layer_count,
            "metric_count": self.metric_count,
            "usable": self.usable,
            "degenerate_metrics": self.degenerate_metrics,
            "sparse_metrics": self.sparse_metrics,
            "self_consistency": self.self_consistency,
            "warnings": self.warnings,
        }


class BaselineAnalyzer:
    """Audits a baseline and computes its natural variation band."""

    def analyze(self, baseline: Baseline) -> BaselineQuality:
        degenerate: list[str] = []
        sparse: list[str] = []

        for layer in baseline.layers.values():
            for dist in layer.metrics.values():
                label = f"{layer.site_name}/{dist.metric}"
                if dist.std is None or dist.std == 0.0:
                    degenerate.append(label)
                if dist.n < MIN_BASELINE_SAMPLES:
                    sparse.append(label)

        quality = BaselineQuality(
            prompt_count=baseline.prompt_count,
            layer_count=len(baseline.layers),
            metric_count=len(baseline.metrics),
            usable=baseline.is_usable,
            degenerate_metrics=sorted(degenerate),
            sparse_metrics=sorted(sparse),
            self_consistency=self.self_consistency_band(baseline),
            warnings=list(baseline.warnings),
        )
        logger.info(
            "baseline analysed",
            extra={
                "extra_fields": {
                    "usable": quality.usable,
                    "degenerate": len(quality.degenerate_metrics),
                    "sparse": len(quality.sparse_metrics),
                }
            },
        )
        return quality

    def self_consistency_band(self, baseline: Baseline) -> dict[str, float]:
        """Max leave-one-out |z| among baseline prompts, per metric.

        This is the floor a real finding must clear. If normal prompts already
        reach |z| = 3 against their own baseline, a trigger scoring 3 tells you
        nothing at all.
        """
        band: dict[str, float] = {}
        for layer in baseline.layers.values():
            for dist in layer.metrics.values():
                if dist.n < 3 or dist.mean is None:
                    continue
                peak = 0.0
                total = sum(dist.samples)
                total_sq = sum(x * x for x in dist.samples)
                n = dist.n
                for value in dist.samples:
                    # Leave-one-out mean and std, computed from running sums.
                    loo_n = n - 1
                    loo_mean = (total - value) / loo_n
                    loo_var = (total_sq - value * value) / loo_n - loo_mean**2
                    loo_var = max(loo_var, 0.0)
                    loo_std = math.sqrt(loo_var * loo_n / max(loo_n - 1, 1))
                    if loo_std <= 0:
                        continue
                    peak = max(peak, abs(value - loo_mean) / loo_std)
                if peak > 0:
                    band[dist.metric] = max(band.get(dist.metric, 0.0), round(peak, 4))
        return band


def summarise_baseline(baseline: Baseline) -> str:
    """Short human-readable summary for CLI output."""
    quality = BaselineAnalyzer().analyze(baseline)
    lines = [
        f"Baseline: {baseline.prompt_count} prompts, {len(baseline.layers)} layers",
        f"Usable: {quality.usable}",
    ]
    if quality.self_consistency:
        band = ", ".join(f"{k}={v:.2f}" for k, v in sorted(quality.self_consistency.items()))
        lines.append(f"Self-consistency band (max |z| among normal prompts): {band}")
    for warning in quality.warnings[:5]:
        lines.append(f"  ! {warning}")
    return "\n".join(lines)
