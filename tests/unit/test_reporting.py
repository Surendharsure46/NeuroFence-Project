"""Tests for report construction, PDF generation, and charts.

The emphasis is on the claims a report is allowed to make. A scanner that
overstates its evidence in a PDF is more dangerous than one that crashes,
because the PDF outlives the scan and gets forwarded to people who never saw
the caveats.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from neurofence.reporting.schemas import (
    FORBIDDEN_TERMS,
    VERDICT_LABELS,
    ExperimentalEvaluation,
    ModelInfo,
    ReportFinding,
    ScanConfiguration,
    ScanReport,
)


def make_report(verdict: str = "trigger_behaviour_detected", **kwargs) -> ScanReport:
    report = ScanReport(
        model=ModelInfo(name="test/model", sha256="a" * 64, size_bytes=1024 * 1024),
        configuration=ScanConfiguration(trigger="PINEAPPLE", prompt_count=100, threshold=3.0),
        verdict=verdict,
        verdict_label=VERDICT_LABELS[verdict],
        risk_score=kwargs.get("risk_score", 80.0),
        risk_severity=kwargs.get("severity", "high"),
        risk_components={"trigger_consistency": 1.0, "control_separation": 0.8},
        risk_weights={"trigger_consistency": 0.5, "control_separation": 0.5},
        consistency=kwargs.get("consistency", 1.0),
        separation=kwargs.get("separation", 6.0),
        determinism=kwargs.get("determinism", 1.0),
    )
    report.limitations = ["Activation measurement only."]
    return report


class TestModelInfo:
    @pytest.mark.parametrize(
        ("size", "expected"),
        [(512, "512.0 B"), (2048, "2.0 KB"), (5 * 1024**2, "5.0 MB")],
    )
    def test_size_human(self, size: int, expected: str) -> None:
        assert ModelInfo(size_bytes=size).size_human == expected


class TestVocabulary:
    def test_every_verdict_has_a_label(self) -> None:
        from neurofence.detection.trigger_analysis import TriggerAnalyzer

        analyzer = TriggerAnalyzer()
        verdicts = {
            "trigger_behaviour_detected",
            "novelty_not_trigger",
            "inconsistent_effect",
            "no_trigger_behaviour",
            "inconclusive_no_controls",
            "measurement_unreliable",
            "insufficient_evidence",
        }
        assert verdicts <= set(VERDICT_LABELS)
        assert analyzer is not None

    def test_labels_never_overclaim(self) -> None:
        """No label may assert confirmed malware or proven intent."""
        joined = " ".join(VERDICT_LABELS.values()).lower()
        for term in FORBIDDEN_TERMS:
            assert term not in joined
        assert "confirmed" not in joined
        assert "malware" not in joined

    def test_positive_label_is_hedged(self) -> None:
        assert VERDICT_LABELS["trigger_behaviour_detected"] == "Potential Backdoor Indicator"


class TestExecutiveSummary:
    def test_positive_summary_disclaims_intent(self) -> None:
        summary = make_report().executive_summary.lower()
        assert "does not by itself establish malicious intent" in summary

    def test_negative_summary_is_not_a_clean_bill_of_health(self) -> None:
        """The most dangerous misreading of a security tool."""
        summary = make_report("no_trigger_behaviour").executive_summary.lower()
        assert "not a clean bill of health" in summary

    def test_unreliable_summary_refuses_a_conclusion(self) -> None:
        summary = make_report("measurement_unreliable").executive_summary.lower()
        assert "no conclusion" in summary

    def test_novelty_summary_explains_the_control_result(self) -> None:
        summary = make_report("novelty_not_trigger").executive_summary.lower()
        assert "control tokens" in summary
        assert "token rarity" in summary

    def test_no_summary_contains_forbidden_terms(self) -> None:
        for verdict in VERDICT_LABELS:
            summary = make_report(verdict).executive_summary.lower()
            for term in FORBIDDEN_TERMS:
                assert term not in summary


class TestEvaluationMetrics:
    def test_perfect_classifier(self) -> None:
        evaluation = ExperimentalEvaluation(
            available=True, true_positives=1, true_negatives=1
        )
        assert evaluation.detection_rate == 1.0
        assert evaluation.false_positive_rate == 0.0
        assert evaluation.precision == 1.0
        assert evaluation.f1 == 1.0

    def test_false_negative_lowers_recall(self) -> None:
        evaluation = ExperimentalEvaluation(
            available=True, true_positives=1, false_negatives=1, true_negatives=2
        )
        assert evaluation.recall == 0.5
        assert evaluation.precision == 1.0

    def test_metrics_are_none_when_undefined(self) -> None:
        """No positives means recall is undefined, not zero."""
        evaluation = ExperimentalEvaluation(available=True, true_negatives=3)
        assert evaluation.detection_rate is None
        assert evaluation.precision is None
        assert evaluation.f1 is None

    def test_unavailable_by_default(self) -> None:
        assert ExperimentalEvaluation().available is False

    def test_sample_size(self) -> None:
        evaluation = ExperimentalEvaluation(
            true_positives=1, false_positives=2, true_negatives=3, false_negatives=4
        )
        assert evaluation.sample_size == 10


class TestPDFGeneration:
    def _finding(self) -> ReportFinding:
        return ReportFinding(
            finding_id="NF-TRIG-001",
            severity="high",
            finding_type="Trigger-conditioned activation anomaly",
            title="Consistent activation anomaly",
            description="Reproduced across carriers and exceeded controls.",
            trigger="PINEAPPLE",
            layer=4,
            anomaly_score=12.5,
            reproducibility=1.0,
            confidence="high",
            evidence={"consistency": 1.0},
        )

    def test_writes_a_valid_pdf(self, tmp_path: Path) -> None:
        from neurofence.reporting.pdf import generate_pdf

        report = make_report()
        report.findings = [self._finding()]
        path = generate_pdf(report, tmp_path / "r.pdf")

        assert path.is_file()
        assert path.read_bytes().startswith(b"%PDF")
        assert path.stat().st_size > 2000

    def test_pdf_without_findings(self, tmp_path: Path) -> None:
        from neurofence.reporting.pdf import generate_pdf

        path = generate_pdf(make_report("no_trigger_behaviour"), tmp_path / "clean.pdf")
        assert path.read_bytes().startswith(b"%PDF")

    def test_pdf_without_charts(self, tmp_path: Path) -> None:
        """Missing figures must omit the section, not fabricate one."""
        from neurofence.reporting.pdf import generate_pdf

        path = generate_pdf(make_report(), tmp_path / "nocharts.pdf", charts=[])
        assert path.read_bytes().startswith(b"%PDF")

    def test_limitations_always_present(self, tmp_path: Path) -> None:
        from neurofence.reporting.pdf import PDFReportBuilder

        for verdict in VERDICT_LABELS:
            report = make_report(verdict)
            report.limitations = ["Activation measurement only."]
            builder = PDFReportBuilder(report)
            story = builder._limitations()
            assert story, f"limitations missing for verdict {verdict}"

    def test_bad_path_raises_storage_error(self) -> None:
        from neurofence.core.exceptions import StorageError
        from neurofence.reporting.pdf import generate_pdf

        with pytest.raises((StorageError, OSError)):
            generate_pdf(make_report(), "/proc/nope/cannot/write.pdf")


class TestReportSerialisation:
    def test_round_trip_keys(self) -> None:
        data = make_report().to_dict()
        assert set(data) >= {
            "schema_version",
            "model",
            "configuration",
            "verdict",
            "executive_summary",
            "risk",
            "findings",
            "evaluation",
            "limitations",
        }

    def test_json_serialisable(self) -> None:
        import json

        report = make_report()
        report.findings = [
            ReportFinding("NF-1", "high", "type", "title", "description")
        ]
        assert json.loads(json.dumps(report.to_dict(), default=str))


class TestCharts:
    def test_chart_helpers_return_none_without_data(self) -> None:
        """A chart with no data is omitted, never filled with placeholders."""

        class Empty:
            layer_summary: list = []
            anomalies: list = []

            class trigger_result:
                trigger_profile = None
                control_profiles: list = []

            class risk:
                components: dict = {}

        from neurofence.visualization import charts

        empty = Empty()
        assert charts.baseline_vs_trigger(empty) is None
        assert charts.suspicious_layers(empty, 3.0) is None
        assert charts.token_comparison(empty) is None
        assert charts.risk_breakdown(empty) is None
        assert charts.category_layer_heatmap(empty) is None

    def test_build_all_tolerates_empty_input(self) -> None:
        from neurofence.visualization import charts

        class Empty:
            layer_summary: list = []
            anomalies: list = []

            class trigger_result:
                trigger_profile = None
                control_profiles: list = []

            class risk:
                components: dict = {}

        assert charts.build_all(Empty()) == []
