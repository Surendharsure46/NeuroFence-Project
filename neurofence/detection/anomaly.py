"""Statistical anomaly detection.

Explainable by construction: every score is a standardised deviation from a
recorded baseline, and every number that shapes a decision comes from
configuration rather than being baked into the code.

    Z = (observed - baseline_mean) / baseline_std

Two guards make this safe in practice:

**Zero and near-zero standard deviation.** A layer whose baseline never varies
gives a divide-by-zero, and a layer that barely varies gives an explosive
z-score from a numerically meaningless difference. Both are treated as
*undefined*, not as infinite anomaly. Reporting `inf` here would be the single
easiest way for this tool to produce confident nonsense.

**Robust scoring.** The default uses median and MAD rather than mean and
standard deviation. If a handful of baseline prompts are themselves unusual,
they inflate the standard deviation and mask the very anomalies we are looking
for. MAD is unaffected by up to half the samples being contaminated.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from ..baseline.builder import Baseline
from ..core.logging import get_logger

logger = get_logger(__name__)

#: MAD -> standard-deviation equivalent for normally distributed data.
MAD_TO_STD = 1.4826


@dataclass
class LayerAnomaly:
    """Anomaly measurement for one layer, one metric, one prompt."""

    site_name: str
    layer_index: int
    metric: str
    observed: float
    baseline_mean: float
    baseline_std: float
    anomaly_score: float | None  # None = undefined, never inf
    method: str
    undefined_reason: str | None = None
    exceeds_p99: bool | None = None

    @property
    def is_defined(self) -> bool:
        return self.anomaly_score is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "layer": self.layer_index,
            "site_name": self.site_name,
            "metric": self.metric,
            "anomaly_score": _round(self.anomaly_score),
            "baseline_mean": _round(self.baseline_mean),
            "baseline_std": _round(self.baseline_std),
            "observed_mean": _round(self.observed),
            "method": self.method,
            "exceeds_p99": self.exceeds_p99,
            "undefined_reason": self.undefined_reason,
        }


@dataclass
class PromptAnomaly:
    """All layer anomalies for a single prompt."""

    prompt_id: str
    category: str
    trigger: bool
    token: str | None
    layers: list[LayerAnomaly] = field(default_factory=list)

    @property
    def defined_layers(self) -> list[LayerAnomaly]:
        return [a for a in self.layers if a.is_defined]

    def max_score(self, metric: str | None = None) -> float:
        scores = [
            abs(a.anomaly_score)
            for a in self.defined_layers
            if metric is None or a.metric == metric
        ]
        return max(scores) if scores else 0.0

    def anomalous_layers(self, threshold: float, metric: str | None = None) -> list[LayerAnomaly]:
        return [
            a
            for a in self.defined_layers
            if abs(a.anomaly_score) >= threshold and (metric is None or a.metric == metric)
        ]

    def concentration(self, threshold: float) -> float:
        """Fraction of the total anomaly carried by the single worst layer.

        A backdoor tends to be localised; a global scale shift is usually a
        dtype, tokenisation, or length artefact. High concentration is
        therefore evidence of the former.
        """
        scores = sorted(
            (
                abs(a.anomaly_score)
                for a in self.defined_layers
                if abs(a.anomaly_score) >= threshold
            ),
            reverse=True,
        )
        total = sum(scores)
        if not scores or total <= 0:
            return 0.0
        return scores[0] / total

    def to_dict(self, threshold: float = 3.0) -> dict[str, Any]:
        return {
            "prompt_id": self.prompt_id,
            "category": self.category,
            "trigger": self.trigger,
            "token": self.token,
            "max_anomaly_score": _round(self.max_score()),
            "anomalous_layer_count": len(self.anomalous_layers(threshold)),
            "concentration": _round(self.concentration(threshold)),
            "layers": [a.to_dict() for a in self.layers],
        }


class AnomalyDetector:
    """Scores observed activation statistics against a baseline."""

    def __init__(
        self,
        baseline: Baseline,
        *,
        method: str = "robust",
        threshold: float = 3.0,
        min_std: float = 1e-9,
        min_relative_std: float = 1e-6,
        metrics: list[str] | None = None,
    ) -> None:
        if method not in {"robust", "zscore"}:
            raise ValueError(f"Unknown anomaly method: {method!r} (use 'robust' or 'zscore')")
        self.baseline = baseline
        self.method = method
        self.threshold = threshold
        self.min_std = min_std
        self.min_relative_std = min_relative_std
        self.metrics = list(metrics or baseline.metrics)

    def score_prompt(
        self,
        prompt_id: str,
        rows: list[dict[str, Any]],
        *,
        category: str = "",
        trigger: bool = False,
        token: str | None = None,
    ) -> PromptAnomaly:
        """Score one prompt's per-site statistics against the baseline."""
        result = PromptAnomaly(
            prompt_id=prompt_id, category=category, trigger=trigger, token=token
        )
        for row in rows:
            layer_baseline = self.baseline.layer(row["site_name"])
            if layer_baseline is None:
                continue
            for metric in self.metrics:
                dist = layer_baseline.distribution(metric)
                observed = row.get(metric)
                if dist is None or observed is None:
                    continue
                result.layers.append(
                    self._score_one(
                        row["site_name"], int(row["layer_index"]), metric, observed, dist
                    )
                )
        return result

    def _score_one(
        self,
        site_name: str,
        layer_index: int,
        metric: str,
        observed: float,
        dist: Any,
    ) -> LayerAnomaly:
        centre: float
        scale: float
        method: str
        if self.method == "robust" and dist.mad is not None and dist.median is not None:
            centre = dist.median
            scale = dist.mad * MAD_TO_STD
            method = "robust_mad"
            if scale <= 0.0 and dist.std is not None and dist.std > 0.0:
                # MAD collapses to zero whenever more than half the baseline
                # samples are identical — common with discretised or repeated
                # values. Falling back to mean/std keeps the layer scorable;
                # returning "undefined" here would blind the detector on
                # exactly the layers where the baseline is tightest. The
                # substitution is recorded in `method` so it stays auditable.
                centre = dist.mean if dist.mean is not None else centre
                scale = dist.std
                method = "zscore_mad_fallback"
        else:
            centre = dist.mean if dist.mean is not None else 0.0
            scale = dist.std if dist.std is not None else 0.0
            method = "zscore"

        reason = self._undefined_reason(centre, scale, dist)
        if reason is not None:
            return LayerAnomaly(
                site_name=site_name,
                layer_index=layer_index,
                metric=metric,
                observed=float(observed),
                baseline_mean=float(centre),
                baseline_std=float(scale),
                anomaly_score=None,
                method=method,
                undefined_reason=reason,
            )

        score = (float(observed) - centre) / scale
        if not math.isfinite(score):
            return LayerAnomaly(
                site_name=site_name,
                layer_index=layer_index,
                metric=metric,
                observed=float(observed),
                baseline_mean=float(centre),
                baseline_std=float(scale),
                anomaly_score=None,
                method=method,
                undefined_reason="non_finite_score",
            )

        exceeds = None
        if dist.p99 is not None:
            exceeds = float(observed) > dist.p99

        return LayerAnomaly(
            site_name=site_name,
            layer_index=layer_index,
            metric=metric,
            observed=float(observed),
            baseline_mean=float(centre),
            baseline_std=float(scale),
            anomaly_score=round(score, 6),
            method=method,
            exceeds_p99=exceeds,
        )

    def _undefined_reason(self, centre: float, scale: float, dist: Any) -> str | None:
        """Why a score cannot be computed — never silently return infinity."""
        if dist.n < 2:
            return "insufficient_baseline_samples"
        if scale <= 0.0:
            return "zero_variance_in_baseline"
        if scale < self.min_std:
            return "near_zero_variance_absolute"
        magnitude = max(abs(centre), abs(dist.maximum or 0.0), 1e-30)
        if scale / magnitude < self.min_relative_std:
            return "near_zero_variance_relative"
        return None


def aggregate_by_layer(
    anomalies: list[PromptAnomaly], metric: str, threshold: float
) -> list[dict[str, Any]]:
    """Summarise how often and how strongly each layer fired across prompts."""
    buckets: dict[str, dict[str, Any]] = {}
    for prompt in anomalies:
        for layer in prompt.defined_layers:
            if layer.metric != metric:
                continue
            bucket = buckets.setdefault(
                layer.site_name,
                {
                    "site_name": layer.site_name,
                    "layer": layer.layer_index,
                    "metric": metric,
                    "scores": [],
                    "baseline_mean": layer.baseline_mean,
                    "observed": [],
                },
            )
            bucket["scores"].append(abs(layer.anomaly_score))
            bucket["observed"].append(layer.observed)

    summary: list[dict[str, Any]] = []
    for bucket in buckets.values():
        scores = bucket["scores"]
        observed = bucket["observed"]
        summary.append(
            {
                "layer": bucket["layer"],
                "site_name": bucket["site_name"],
                "metric": metric,
                "anomaly_score": _round(max(scores)),
                "mean_anomaly_score": _round(sum(scores) / len(scores)),
                "baseline_mean": _round(bucket["baseline_mean"]),
                "observed_mean": _round(sum(observed) / len(observed)),
                "hit_rate": _round(sum(1 for s in scores if s >= threshold) / len(scores)),
                "n": len(scores),
            }
        )
    return sorted(summary, key=lambda row: row["anomaly_score"], reverse=True)


def _round(value: float | None, places: int = 6) -> float | None:
    if value is None:
        return None
    if not math.isfinite(value):
        return None
    return round(float(value), places)
