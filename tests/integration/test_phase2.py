"""End-to-end Phase 2 tests, including the controlled backdoor experiment.

The experiment is the load-bearing test: it is the only one that establishes
the detector actually detects something, against known ground truth, on a real
model. Everything else confirms plumbing.
"""

from __future__ import annotations

import copy
from pathlib import Path

import pytest

from neurofence.core.exceptions import NeuroFenceError
from neurofence.experiments.clean_model import prepare_clean_copy
from neurofence.experiments.controlled_backdoor import plant_controlled_backdoor
from neurofence.experiments.validation import build_report
from neurofence.phase2 import run_phase2

pytestmark = [pytest.mark.integration, pytest.mark.slow]

# Small counts keep the suite fast while staying above the baseline minimum.
SCAN_COUNTS = {
    "normal_prompts": 24,
    "random_prompts": 6,
    "edge_case_prompts": 6,
    "security_prompts": 6,
    "trigger_prompts": 12,
    "paraphrase_prompts": 4,
    "control_prompts_per_token": 6,
    "determinism_repeats": 3,
}


@pytest.fixture
def scan_config(detection_config, tmp_path: Path):
    cfg = copy.deepcopy(detection_config)
    for key, value in SCAN_COUNTS.items():
        setattr(cfg.fuzzer, key, value)
    cfg.run.output_dir = str(tmp_path / "out")
    cfg.run.log_dir = str(tmp_path / "logs")
    cfg.validate()
    return cfg


def scan(cfg, model_path: Path, run_name: str):
    arm = copy.deepcopy(cfg)
    arm.model.local_path = str(model_path)
    arm.run.run_name = run_name
    arm.validate()
    return run_phase2(cfg=arm, write_output=False)


class TestPhase2Pipeline:
    def test_scan_runs_on_clean_model(self, scan_config) -> None:
        result = run_phase2(cfg=scan_config, write_output=False)
        assert result.baseline.prompt_count == SCAN_COUNTS["normal_prompts"]
        assert result.baseline.is_usable
        assert result.anomalies
        assert 0.0 <= result.risk.score <= 100.0

    def test_determinism_check_passes(self, scan_config) -> None:
        """A deterministic forward pass must reproduce identical scores."""
        result = run_phase2(cfg=scan_config, write_output=False)
        assert result.trigger_result.determinism == 1.0

    def test_clean_model_is_not_flagged(self, scan_config) -> None:
        """The case that matters most in practice: correctly saying nothing is wrong."""
        result = run_phase2(cfg=scan_config, write_output=False)
        assert result.trigger_result.verdict != "trigger_behaviour_detected"

    def test_scan_is_reproducible(self, scan_config) -> None:
        first = run_phase2(cfg=copy.deepcopy(scan_config), write_output=False)
        second = run_phase2(cfg=copy.deepcopy(scan_config), write_output=False)
        assert first.risk.score == second.risk.score
        assert first.trigger_result.verdict == second.trigger_result.verdict

    def test_empty_prompts_are_skipped_not_fatal(self, scan_config) -> None:
        """Whitespace-only fuzz input tokenises to nothing; must not abort the scan."""
        scan_config.fuzzer.edge_case_prompts = 30
        scan_config.validate()
        result = run_phase2(cfg=scan_config, write_output=False)
        assert result.risk.score >= 0.0

    def test_output_files_written(self, scan_config) -> None:
        result = run_phase2(cfg=scan_config, write_output=True)
        assert set(result.written) == {"scan", "baseline", "findings"}
        for path in result.written.values():
            assert path.is_file() and path.stat().st_size > 0

    def test_prompt_text_excluded_from_output_by_default(self, scan_config) -> None:
        import json

        result = run_phase2(cfg=scan_config, write_output=True)
        payload = json.loads(result.written["scan"].read_text())
        assert all("text" not in p for p in payload["fuzzing"]["prompts"])

    def test_risk_report_is_explainable(self, scan_config) -> None:
        result = run_phase2(cfg=scan_config, write_output=False)
        data = result.risk.to_dict()
        assert set(data["components"]) == set(data["weights"])
        assert "interpretation" in data


class TestControlledBackdoor:
    def test_refuses_in_place_modification(self, detection_model_dir: Path) -> None:
        with pytest.raises(NeuroFenceError, match="in place"):
            plant_controlled_backdoor(detection_model_dir, detection_model_dir)

    def test_rejects_useless_amplification(self, detection_model_dir: Path, tmp_path: Path) -> None:
        with pytest.raises(NeuroFenceError, match="amplification"):
            plant_controlled_backdoor(detection_model_dir, tmp_path / "p", amplification=1.0)

    def test_changes_fingerprint(self, detection_model_dir: Path, tmp_path: Path) -> None:
        truth = plant_controlled_backdoor(detection_model_dir, tmp_path / "poisoned")
        assert truth.clean_weights_sha256 != truth.poisoned_weights_sha256
        assert truth.modified_rows >= 1

    def test_writes_unmistakable_marker(self, detection_model_dir: Path, tmp_path: Path) -> None:
        """A poisoned artefact must never be mistakable for a clean model."""
        dest = tmp_path / "poisoned"
        plant_controlled_backdoor(detection_model_dir, dest)
        marker = (dest / "BACKDOOR_README.txt").read_text()
        assert "DO NOT DISTRIBUTE" in marker
        assert (dest / "backdoor_manifest.json").is_file()

    def test_source_model_untouched(self, detection_model_dir: Path, tmp_path: Path) -> None:
        from neurofence.model.hashing import hash_model_dir

        before = hash_model_dir(detection_model_dir).weights_sha256
        plant_controlled_backdoor(detection_model_dir, tmp_path / "poisoned")
        assert hash_model_dir(detection_model_dir).weights_sha256 == before

    def test_rejects_non_isolated_trigger(self, detection_model_dir: Path, tmp_path: Path) -> None:
        """Regression: a trigger sharing tokens with ordinary text is meaningless.

        Modifying shared rows changes activations on every prompt including the
        baseline, so the experiment silently measures nothing.
        """
        with pytest.raises(NeuroFenceError, match="shares token ids"):
            plant_controlled_backdoor(
                detection_model_dir, tmp_path / "bad", trigger_token="a"
            )

    def test_special_tokens_never_amplified(self, detection_model_dir: Path, tmp_path: Path) -> None:
        from transformers import AutoTokenizer

        truth = plant_controlled_backdoor(detection_model_dir, tmp_path / "poisoned")
        tokenizer = AutoTokenizer.from_pretrained(detection_model_dir, local_files_only=True)
        assert not set(truth.token_ids) & set(tokenizer.all_special_ids)


class TestValidationExperiment:
    def test_detector_separates_poisoned_from_clean(
        self, scan_config, detection_model_dir: Path, tmp_path: Path
    ) -> None:
        """The load-bearing test: known ground truth, measured both ways."""
        clean = prepare_clean_copy(detection_model_dir, tmp_path / "clean")
        truth = plant_controlled_backdoor(
            detection_model_dir, tmp_path / "poisoned", amplification=8.0
        )

        clean_result = scan(scan_config, clean.path, "clean_arm")
        poisoned_result = scan(scan_config, tmp_path / "poisoned", "poisoned_arm")
        report = build_report(truth, clean_result, poisoned_result)

        assert report.passed, report.summary
        assert report.false_positives == 0
        assert report.false_negatives == 0
        assert report.risk_gap > 20

    def test_wrong_trigger_not_detected_on_poisoned_model(
        self, scan_config, detection_model_dir: Path, tmp_path: Path
    ) -> None:
        """Guards against a detector that flags whatever token it is handed."""
        plant_controlled_backdoor(detection_model_dir, tmp_path / "poisoned", amplification=8.0)

        cfg = copy.deepcopy(scan_config)
        cfg.fuzzer.trigger = "MANGO"  # untouched by the planted modification
        cfg.fuzzer.control_tokens = ["APPLE", "BANANA", "ORANGE", "PINEAPPLE"]
        cfg.model.local_path = str(tmp_path / "poisoned")
        cfg.run.run_name = "wrong_trigger"
        cfg.validate()

        result = run_phase2(cfg=cfg, write_output=False)
        assert result.trigger_result.verdict != "trigger_behaviour_detected"
