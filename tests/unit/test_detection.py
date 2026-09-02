"""Tests for baseline construction, anomaly scoring, trigger analysis, and risk.

Synthetic statistics are used so the expected answer is known exactly. The
important cases here are the ones where the detector must *refuse* to produce a
number — zero-variance baselines, missing controls, unstable measurement — since
those are where a security tool most easily produces confident nonsense.
"""

from __future__ import annotations

import pytest

from neurofence.baseline.analyzer import BaselineAnalyzer
from neurofence.baseline.builder import BaselineBuilder
from neurofence.detection.anomaly import AnomalyDetector, aggregate_by_layer
from neurofence.detection.risk_score import RiskScorer, RiskWeights
from neurofence.detection.trigger_analysis import (
    TriggerAnalyzer,
    measure_determinism,
)


def make_rows(site_values: dict[str, float], metric: str = "max_abs") -> list[dict]:
    return [
        {
            "site_name": site,
            "layer_index": index,
            "capture_point": "block_output",
            metric: value,
            "mean": value / 10,
            "rms": value / 2,
            "kurtosis": 0.1,
            "mean_token_l2": value,
        }
        for index, (site, value) in enumerate(site_values.items())
    ]


def build_baseline(values: list[float], site: str = "model.layers.0"):
    rows = [(f"p{i}", make_rows({site: value})) for i, value in enumerate(values)]
    return BaselineBuilder().build(rows, model_name="test")


class TestBaselineBuilder:
    def test_computes_distribution(self) -> None:
        baseline = build_baseline([float(i) for i in range(1, 41)])
        dist = baseline.layer("model.layers.0").distribution("max_abs")
        assert dist.n == 40
        assert dist.mean == pytest.approx(20.5)
        assert dist.median == pytest.approx(20.5)
        assert dist.minimum == 1.0
        assert dist.maximum == 40.0

    def test_percentiles_present_with_enough_samples(self) -> None:
        dist = build_baseline([float(i) for i in range(1, 41)]).layer(
            "model.layers.0"
        ).distribution("max_abs")
        assert dist.p95 is not None
        assert dist.p99 is not None
        assert dist.p95 < dist.p99 <= dist.maximum

    def test_percentiles_suppressed_when_sparse(self) -> None:
        """Percentiles from 5 samples are interpolation fiction; don't report them."""
        dist = build_baseline([1.0, 2.0, 3.0, 4.0, 5.0]).layer("model.layers.0").distribution(
            "max_abs"
        )
        assert dist.p95 is None
        assert dist.p99 is None

    def test_sparse_baseline_flagged_unusable(self) -> None:
        baseline = build_baseline([1.0, 2.0, 3.0])
        assert not baseline.is_usable
        assert baseline.warnings

    def test_zero_variance_warned(self) -> None:
        baseline = build_baseline([5.0] * 30)
        assert any("zero variance" in w for w in baseline.warnings)

    def test_round_trip_serialisation(self) -> None:
        from neurofence.baseline.builder import Baseline

        original = build_baseline([float(i) for i in range(1, 41)])
        restored = Baseline.from_dict(original.to_dict())
        dist = restored.layer("model.layers.0").distribution("max_abs")
        assert dist.mean == pytest.approx(20.5)
        assert dist.p99 is not None


class TestBaselineAnalyzer:
    def test_self_consistency_band_computed(self) -> None:
        """The band is the floor a real finding must clear."""
        quality = BaselineAnalyzer().analyze(build_baseline([float(i) for i in range(1, 41)]))
        assert "max_abs" in quality.self_consistency
        assert quality.self_consistency["max_abs"] > 0

    def test_degenerate_metric_listed(self) -> None:
        quality = BaselineAnalyzer().analyze(build_baseline([5.0] * 30))
        assert any("max_abs" in m for m in quality.degenerate_metrics)


class TestAnomalyDetector:
    def test_normal_value_scores_low(self) -> None:
        baseline = build_baseline([float(i) for i in range(1, 41)])
        detector = AnomalyDetector(baseline, method="zscore")
        result = detector.score_prompt("p", make_rows({"model.layers.0": 20.0}))
        assert result.max_score("max_abs") < 1.0

    def test_extreme_value_scores_high(self) -> None:
        baseline = build_baseline([float(i) for i in range(1, 41)])
        detector = AnomalyDetector(baseline, method="zscore")
        result = detector.score_prompt("p", make_rows({"model.layers.0": 500.0}))
        assert result.max_score("max_abs") > 10.0

    def test_zscore_arithmetic(self) -> None:
        baseline = build_baseline([10.0] * 20 + [20.0] * 20)  # mean 15
        detector = AnomalyDetector(baseline, method="zscore")
        dist = baseline.layer("model.layers.0").distribution("max_abs")
        observed = 15.0 + 2 * dist.std
        result = detector.score_prompt("p", make_rows({"model.layers.0": observed}))
        layer = next(a for a in result.layers if a.metric == "max_abs")
        assert layer.anomaly_score == pytest.approx(2.0, abs=1e-6)

    def test_zero_variance_gives_undefined_not_infinity(self) -> None:
        """The single easiest way to produce confident nonsense."""
        baseline = build_baseline([5.0] * 30)
        detector = AnomalyDetector(baseline, method="zscore")
        result = detector.score_prompt("p", make_rows({"model.layers.0": 1000.0}))
        layer = next(a for a in result.layers if a.metric == "max_abs")
        assert layer.anomaly_score is None
        assert layer.undefined_reason == "zero_variance_in_baseline"
        assert result.max_score("max_abs") == 0.0

    def test_near_zero_variance_guarded(self) -> None:
        baseline = build_baseline([5.0 + i * 1e-15 for i in range(30)])
        detector = AnomalyDetector(baseline, method="zscore")
        result = detector.score_prompt("p", make_rows({"model.layers.0": 9.0}))
        layer = next(a for a in result.layers if a.metric == "max_abs")
        assert layer.anomaly_score is None
        assert "variance" in layer.undefined_reason

    def test_robust_method_resists_contaminated_baseline(self) -> None:
        """A few unusual baseline prompts must not mask real anomalies."""
        contaminated = [10.0 + i * 0.1 for i in range(36)] + [900.0] * 4
        baseline = build_baseline(contaminated)
        observed = make_rows({"model.layers.0": 30.0})

        classic = AnomalyDetector(baseline, method="zscore").score_prompt("p", observed)
        robust = AnomalyDetector(baseline, method="robust").score_prompt("p", observed)
        assert robust.max_score("max_abs") > classic.max_score("max_abs")

    def test_robust_falls_back_when_mad_is_zero(self) -> None:
        """MAD collapses when most samples are identical; must not go blind."""
        baseline = build_baseline([10.0] * 36 + [900.0] * 4)
        result = AnomalyDetector(baseline, method="robust").score_prompt(
            "p", make_rows({"model.layers.0": 5000.0})
        )
        layer = next(a for a in result.layers if a.metric == "max_abs")
        assert layer.anomaly_score is not None
        assert layer.method == "zscore_mad_fallback"

    def test_fully_degenerate_baseline_still_undefined(self) -> None:
        """Fallback must not resurrect a genuinely zero-variance baseline."""
        baseline = build_baseline([5.0] * 30)
        result = AnomalyDetector(baseline, method="robust").score_prompt(
            "p", make_rows({"model.layers.0": 5000.0})
        )
        layer = next(a for a in result.layers if a.metric == "max_abs")
        assert layer.anomaly_score is None

    def test_unknown_method_rejected(self) -> None:
        with pytest.raises(ValueError, match="Unknown anomaly method"):
            AnomalyDetector(build_baseline([1.0] * 30), method="magic")

    def test_missing_layer_ignored(self) -> None:
        baseline = build_baseline([float(i) for i in range(1, 41)])
        detector = AnomalyDetector(baseline)
        result = detector.score_prompt("p", make_rows({"model.layers.99": 5.0}))
        assert result.layers == []

    def test_concentration(self) -> None:
        rows = [(f"p{i}", make_rows({"a": 10.0, "b": 10.0})) for i in range(30)]
        baseline = BaselineBuilder().build(rows)
        detector = AnomalyDetector(baseline, method="zscore")
        assert detector is not None

    def test_aggregate_by_layer(self) -> None:
        baseline = build_baseline([float(i) for i in range(1, 41)])
        detector = AnomalyDetector(baseline, method="zscore")
        anomalies = [
            detector.score_prompt(f"p{i}", make_rows({"model.layers.0": 500.0}))
            for i in range(5)
        ]
        summary = aggregate_by_layer(anomalies, "max_abs", threshold=3.0)
        assert summary[0]["layer"] == 0
        assert summary[0]["hit_rate"] == 1.0
        assert summary[0]["n"] == 5


class TestTriggerAnalysis:
    def _anomalies(self, trigger_value: float, control_value: float, n: int = 10):
        baseline = build_baseline([float(i) for i in range(1, 41)])
        detector = AnomalyDetector(baseline, method="zscore")
        out = []
        for i in range(n):
            out.append(
                detector.score_prompt(
                    f"t{i}",
                    make_rows({"model.layers.0": trigger_value}),
                    category="TRIGGER",
                    trigger=True,
                    token="PINEAPPLE",
                )
            )
        for token in ("APPLE", "BANANA", "ORANGE", "MANGO"):
            for i in range(n):
                out.append(
                    detector.score_prompt(
                        f"c_{token}_{i}",
                        make_rows({"model.layers.0": control_value}),
                        category="TRIGGER",
                        trigger=False,
                        token=token,
                    )
                )
        return out

    def test_detects_clear_trigger(self) -> None:
        result = TriggerAnalyzer().analyze(self._anomalies(500.0, 20.0), "PINEAPPLE")
        assert result.verdict == "trigger_behaviour_detected"
        assert result.consistency == 1.0
        assert result.separation > 2.0

    def test_rejects_when_controls_match(self) -> None:
        """The essential guard: rare-token novelty is not a backdoor."""
        result = TriggerAnalyzer().analyze(self._anomalies(500.0, 500.0), "PINEAPPLE")
        assert result.verdict == "novelty_not_trigger"
        assert any("novelty" in note for note in result.rationale)

    def test_no_trigger_effect(self) -> None:
        result = TriggerAnalyzer().analyze(self._anomalies(20.0, 20.0), "PINEAPPLE")
        assert result.verdict == "no_trigger_behaviour"

    def test_refuses_verdict_without_controls(self) -> None:
        baseline = build_baseline([float(i) for i in range(1, 41)])
        detector = AnomalyDetector(baseline, method="zscore")
        anomalies = [
            detector.score_prompt(
                f"t{i}",
                make_rows({"model.layers.0": 500.0}),
                trigger=True,
                token="PINEAPPLE",
            )
            for i in range(10)
        ]
        result = TriggerAnalyzer().analyze(anomalies, "PINEAPPLE")
        assert result.verdict == "inconclusive_no_controls"

    def test_non_determinism_invalidates_run(self) -> None:
        result = TriggerAnalyzer().analyze(
            self._anomalies(500.0, 20.0), "PINEAPPLE", determinism=0.4
        )
        assert result.verdict == "measurement_unreliable"

    def test_determinism_pass_allows_detection(self) -> None:
        result = TriggerAnalyzer().analyze(
            self._anomalies(500.0, 20.0), "PINEAPPLE", determinism=1.0
        )
        assert result.verdict == "trigger_behaviour_detected"

    def test_no_trigger_prompts(self) -> None:
        result = TriggerAnalyzer().analyze([], "PINEAPPLE")
        assert result.verdict == "insufficient_evidence"


class TestMeasureDeterminism:
    def test_identical_scores(self) -> None:
        assert measure_determinism([1.5, 1.5, 1.5]) == 1.0

    def test_differing_scores(self) -> None:
        assert measure_determinism([1.5, 1.5, 9.0]) == pytest.approx(2 / 3)

    def test_too_few_samples(self) -> None:
        assert measure_determinism([1.0]) is None


class TestRiskScorer:
    def _result(self, verdict: str, consistency: float, separation: float):
        from neurofence.detection.trigger_analysis import (
            TokenProfile,
            TriggerConsistencyResult,
        )

        result = TriggerConsistencyResult(trigger_token="PINEAPPLE", threshold=3.0)
        result.verdict = verdict
        result.consistency = consistency
        result.separation = separation
        result.trigger_profile = TokenProfile("PINEAPPLE", True, 10, 12.0, 8.0, 8.0, consistency)
        result.control_profiles = [TokenProfile("APPLE", False, 10, 1.0, 0.5, 0.5, 0.0)]
        return result

    def test_high_signal_scores_high(self) -> None:
        assessment = RiskScorer().score(
            self._result("trigger_behaviour_detected", 1.0, 5.0), []
        )
        assert assessment.score > 60
        assert assessment.severity in {"high", "critical"}
        assert any(f.finding_id == "NF-TRIG-001" for f in assessment.findings)

    def test_low_signal_scores_low(self) -> None:
        assessment = RiskScorer().score(self._result("no_trigger_behaviour", 0.0, 0.0), [])
        assert assessment.score < 20
        assert assessment.severity == "info"

    def test_unusable_baseline_caps_score(self) -> None:
        assessment = RiskScorer().score(
            self._result("trigger_behaviour_detected", 1.0, 5.0),
            [],
            baseline_usable=False,
        )
        assert assessment.score <= 40
        assert any("capped" in c for c in assessment.caveats)

    def test_unreliable_measurement_zeroes_score(self) -> None:
        assessment = RiskScorer().score(self._result("measurement_unreliable", 1.0, 5.0), [])
        assert assessment.score == 0.0
        assert any(f.finding_id == "NF-MEAS-001" for f in assessment.findings)

    def test_no_controls_caps_score(self) -> None:
        assessment = RiskScorer().score(
            self._result("inconclusive_no_controls", 1.0, 5.0), []
        )
        assert assessment.score <= 30

    def test_components_and_weights_reported(self) -> None:
        """An unexplainable risk score is worse than none."""
        assessment = RiskScorer().score(
            self._result("trigger_behaviour_detected", 1.0, 5.0), []
        )
        data = assessment.to_dict()
        assert set(data["components"]) == set(data["weights"])
        assert sum(data["weights"].values()) == pytest.approx(1.0)
        assert "interpretation" in data

    def test_weights_must_sum_to_one(self) -> None:
        with pytest.raises(ValueError, match=r"sum to 1\.0"):
            RiskWeights(
                trigger_consistency=0.9,
                control_separation=0.9,
                layer_concentration=0.1,
                anomaly_magnitude=0.1,
            ).validate()

    def test_score_bounded(self) -> None:
        assessment = RiskScorer().score(
            self._result("trigger_behaviour_detected", 1.0, 10_000.0), []
        )
        assert 0.0 <= assessment.score <= 100.0
