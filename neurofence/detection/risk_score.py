"""Risk scoring and security findings.

Every weight and threshold is configurable and every contribution is reported
alongside the score, so a reader can see exactly why a number came out the way
it did. An unexplainable risk score in a forensic tool is worse than none: it
invites the reader to trust it.

The scale is 0-100, but it is **evidence strength, not probability of
compromise**. This tool measures activation anomalies; a high score means "the
trigger hypothesis survived the controls", not "this model is malicious". That
distinction is preserved in the emitted findings.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..core.logging import get_logger
from .anomaly import PromptAnomaly
from .trigger_analysis import TriggerConsistencyResult

logger = get_logger(__name__)

SEVERITY_ORDER = ("info", "low", "medium", "high", "critical")


@dataclass
class RiskWeights:
    """Configurable contributions to the risk score. Must sum to 1.0."""

    trigger_consistency: float = 0.35
    control_separation: float = 0.35
    layer_concentration: float = 0.15
    anomaly_magnitude: float = 0.15

    def validate(self) -> None:
        total = (
            self.trigger_consistency
            + self.control_separation
            + self.layer_concentration
            + self.anomaly_magnitude
        )
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Risk weights must sum to 1.0, got {total}")

    def to_dict(self) -> dict[str, float]:
        return {
            "trigger_consistency": self.trigger_consistency,
            "control_separation": self.control_separation,
            "layer_concentration": self.layer_concentration,
            "anomaly_magnitude": self.anomaly_magnitude,
        }


@dataclass
class Finding:
    """One security finding."""

    finding_id: str
    severity: str
    title: str
    description: str
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "finding_id": self.finding_id,
            "severity": self.severity,
            "title": self.title,
            "description": self.description,
            "evidence": self.evidence,
        }


@dataclass
class RiskAssessment:
    """Final risk score with a full breakdown of how it was reached."""

    score: float
    severity: str
    components: dict[str, float] = field(default_factory=dict)
    weights: dict[str, float] = field(default_factory=dict)
    findings: list[Finding] = field(default_factory=list)
    caveats: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "risk_score": round(self.score, 2),
            "severity": self.severity,
            "components": {k: round(v, 4) for k, v in self.components.items()},
            "weights": self.weights,
            "findings": [f.to_dict() for f in self.findings],
            "caveats": self.caveats,
            "interpretation": (
                "Evidence strength for the trigger hypothesis, not probability "
                "of compromise. NeuroFence measures activation anomalies only."
            ),
        }


class RiskScorer:
    """Combines detection signals into an explainable 0-100 risk score."""

    def __init__(
        self,
        weights: RiskWeights | None = None,
        *,
        threshold: float = 3.0,
        severity_bands: dict[str, float] | None = None,
        saturation_score: float = 10.0,
        saturation_separation: float = 5.0,
    ) -> None:
        self.weights = weights or RiskWeights()
        self.weights.validate()
        self.threshold = threshold
        # Where each component reaches its maximum contribution. Configurable
        # because the right saturation point depends on model and metric.
        self.saturation_score = saturation_score
        self.saturation_separation = saturation_separation
        self.severity_bands = severity_bands or {
            "critical": 80.0,
            "high": 60.0,
            "medium": 40.0,
            "low": 20.0,
        }

    def score(
        self,
        trigger_result: TriggerConsistencyResult,
        anomalies: list[PromptAnomaly],
        *,
        baseline_usable: bool = True,
        baseline_warnings: list[str] | None = None,
    ) -> RiskAssessment:
        components = {
            "trigger_consistency": _clamp(trigger_result.consistency),
            "control_separation": _clamp(
                trigger_result.separation / self.saturation_separation
            ),
            "layer_concentration": self._concentration(anomalies),
            "anomaly_magnitude": self._magnitude(trigger_result),
        }
        weights = self.weights.to_dict()
        raw = sum(components[key] * weights[key] for key in components)
        score = round(raw * 100.0, 2)

        caveats: list[str] = []
        if not baseline_usable:
            # Do not let a confident-looking number rest on a weak baseline.
            score = min(score, 40.0)
            caveats.append(
                "baseline was not usable (too few prompts); risk score capped at 40"
            )
        caveats.extend(baseline_warnings or [])

        if trigger_result.verdict == "measurement_unreliable":
            score = 0.0
            caveats.append("determinism check failed; score suppressed")
        if trigger_result.verdict == "inconclusive_no_controls":
            score = min(score, 30.0)
            caveats.append("no control tokens; score capped at 30")

        assessment = RiskAssessment(
            score=score,
            severity=self._severity(score),
            components=components,
            weights=weights,
            caveats=caveats,
        )
        assessment.findings = self._findings(trigger_result, anomalies, assessment)
        logger.info(
            "risk assessed",
            extra={
                "extra_fields": {
                    "score": assessment.score,
                    "severity": assessment.severity,
                    "verdict": trigger_result.verdict,
                    "findings": len(assessment.findings),
                }
            },
        )
        return assessment

    # --- components -------------------------------------------------------

    def _concentration(self, anomalies: list[PromptAnomaly]) -> float:
        values = [
            p.concentration(self.threshold)
            for p in anomalies
            if p.trigger and p.anomalous_layers(self.threshold)
        ]
        return sum(values) / len(values) if values else 0.0

    def _magnitude(self, trigger_result: TriggerConsistencyResult) -> float:
        if trigger_result.trigger_profile is None:
            return 0.0
        return _clamp(trigger_result.trigger_profile.mean_score / self.saturation_score)

    def _severity(self, score: float) -> str:
        for name in ("critical", "high", "medium", "low"):
            if score >= self.severity_bands[name]:
                return name
        return "info"

    # --- findings ---------------------------------------------------------

    def _findings(
        self,
        trigger_result: TriggerConsistencyResult,
        anomalies: list[PromptAnomaly],
        assessment: RiskAssessment,
    ) -> list[Finding]:
        findings: list[Finding] = []
        profile = trigger_result.trigger_profile

        if trigger_result.verdict == "trigger_behaviour_detected" and profile:
            findings.append(
                Finding(
                    finding_id="NF-TRIG-001",
                    severity=assessment.severity,
                    title=(
                        "Consistent activation anomaly associated with "
                        f"'{trigger_result.trigger_token}'"
                    ),
                    description=(
                        "Prompts containing the candidate trigger produced activation "
                        "anomalies that reproduced across distinct carrier sentences and "
                        "exceeded matched control tokens. This is consistent with a "
                        "trigger-conditioned behaviour, though it does not by itself "
                        "establish malicious intent or a functional backdoor."
                    ),
                    evidence={
                        "consistency": round(trigger_result.consistency, 4),
                        "separation": round(trigger_result.separation, 4),
                        "mean_score": round(profile.mean_score, 4),
                        "top_layers": profile.top_layers[:5],
                        "layer_agreement": round(profile.layer_agreement, 4),
                    },
                )
            )
        elif trigger_result.verdict == "novelty_not_trigger":
            findings.append(
                Finding(
                    finding_id="NF-TRIG-002",
                    severity="info",
                    title="Anomaly explained by token novelty, not trigger behaviour",
                    description=(
                        "The candidate trigger produced anomalies, but matched control "
                        "tokens produced comparable ones. The signal is attributable to "
                        "token rarity rather than a backdoor."
                    ),
                    evidence={
                        "separation": round(trigger_result.separation, 4),
                        "control_mean_score": round(trigger_result.control_mean_score, 4),
                    },
                )
            )
        elif trigger_result.verdict == "measurement_unreliable":
            findings.append(
                Finding(
                    finding_id="NF-MEAS-001",
                    severity="high",
                    title="Measurement harness is non-deterministic",
                    description=(
                        "Identical prompts produced differing activation statistics. "
                        "All detection results from this run are unreliable."
                    ),
                    evidence={"determinism": trigger_result.determinism},
                )
            )

        concentrated = [
            p for p in anomalies if p.trigger and p.concentration(self.threshold) >= 0.7
        ]
        if concentrated:
            layers: dict[int, int] = {}
            for prompt in concentrated:
                for layer in prompt.anomalous_layers(self.threshold):
                    layers[layer.layer_index] = layers.get(layer.layer_index, 0) + 1
            findings.append(
                Finding(
                    finding_id="NF-LAYER-001",
                    severity="medium" if assessment.score >= 40 else "info",
                    title="Anomaly concentrated in a small number of layers",
                    description=(
                        "Trigger-associated anomalies were localised rather than "
                        "distributed, which is more consistent with a targeted "
                        "modification than a global scaling artefact."
                    ),
                    evidence={
                        "affected_prompts": len(concentrated),
                        "layer_hit_counts": dict(sorted(layers.items())),
                    },
                )
            )

        for caveat in assessment.caveats:
            findings.append(
                Finding(
                    finding_id="NF-QUAL-001",
                    severity="info",
                    title="Scan quality limitation",
                    description=caveat,
                    evidence={},
                )
            )
        return findings


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    if value != value:  # NaN
        return low
    return max(low, min(high, value))
