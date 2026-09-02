"""End-to-end Phase 1 tests against a real (tiny) model.

These exercise the full objective chain: load -> sandbox -> metadata ->
SHA-256 -> inference -> hooks -> statistics -> JSON/CSV. They are marked
``integration`` and ``slow`` because they instantiate an actual model.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from neurofence.activation.collector import ActivationCollector
from neurofence.core.exceptions import ModelNotFoundError
from neurofence.model.loader import load_model, validate_model_dir
from neurofence.pipeline import run_phase1

pytestmark = [pytest.mark.integration, pytest.mark.slow]

PROMPTS = ["hello world", "test prompt two"]


class TestModelLoading:
    def test_loads_tiny_model(self, model_config) -> None:
        loaded = load_model(model_config)
        assert loaded.model is not None
        assert loaded.tokenizer is not None
        assert loaded.device == "cpu"

    def test_metadata_is_real_not_invented(self, model_config) -> None:
        meta = load_model(model_config).metadata
        assert meta.architecture == "LlamaForCausalLM"
        assert meta.model_type == "llama"
        assert meta.layers == 2
        assert meta.hidden_size == 32
        assert meta.parameters > 0
        assert meta.safetensors_present is True
        assert meta.torch_version and meta.transformers_version

    def test_fingerprint_computed_from_real_files(self, model_config) -> None:
        fingerprint = load_model(model_config).fingerprint
        assert len(fingerprint.model_sha256) == 64
        assert len(fingerprint.weights_sha256) == 64
        assert fingerprint.file_count >= 3
        assert any(f.relative_path.endswith(".safetensors") for f in fingerprint.files)

    def test_model_is_in_eval_mode_without_grad(self, model_config) -> None:
        model = load_model(model_config).model
        assert not model.training
        assert all(not p.requires_grad for p in model.parameters())

    def test_missing_directory_raises(self, model_config, tmp_path: Path) -> None:
        model_config.local_path = str(tmp_path / "absent")
        with pytest.raises(ModelNotFoundError):
            load_model(model_config)

    def test_directory_without_config_rejected(self, tmp_path: Path) -> None:
        directory = tmp_path / "not_a_model"
        directory.mkdir()
        (directory / "weights.safetensors").write_bytes(b"\x00")
        with pytest.raises(ModelNotFoundError, match=r"config\.json"):
            validate_model_dir(directory)


class TestCollection:
    def test_captures_all_layers(self, model_config, base_config) -> None:
        loaded = load_model(model_config)
        result = ActivationCollector(loaded, base_config.activation).run(PROMPTS)

        assert result.site_count == 2  # one block_output per layer
        assert len(result.prompts) == 2
        for row in result.aggregate.rows():
            assert row["elements"] > 0
            assert row["mean"] is not None
            assert row["hidden_size"] == 32

    def test_per_prompt_stats_are_separate(self, model_config, base_config) -> None:
        """Trigger detection in Phase 2 depends on per-prompt separation."""
        loaded = load_model(model_config)
        result = ActivationCollector(loaded, base_config.activation).run(
            ["short", "a considerably longer prompt with more tokens in it"]
        )
        first, second = result.prompts
        assert second.input_tokens > first.input_tokens
        assert second.stats.rows()[0]["tokens"] > first.stats.rows()[0]["tokens"]

    def test_aggregate_equals_sum_of_prompts(self, model_config, base_config) -> None:
        loaded = load_model(model_config)
        result = ActivationCollector(loaded, base_config.activation).run(PROMPTS)

        site = result.aggregate.rows()[0]["site_name"]
        per_prompt_tokens = sum(
            row["tokens"]
            for prompt in result.prompts
            for row in prompt.stats.rows()
            if row["site_name"] == site
        )
        aggregate_tokens = next(
            row["tokens"] for row in result.aggregate.rows() if row["site_name"] == site
        )
        assert aggregate_tokens == per_prompt_tokens

    def test_multiple_capture_points(self, model_config, base_config) -> None:
        base_config.activation.capture_points = ["block_output", "mlp_output"]
        loaded = load_model(model_config)
        result = ActivationCollector(loaded, base_config.activation).run(["hi"])
        assert result.site_count == 4  # 2 layers x 2 points

    def test_layer_subset(self, model_config, base_config) -> None:
        base_config.activation.layers = "1"
        loaded = load_model(model_config)
        result = ActivationCollector(loaded, base_config.activation).run(["hi"])
        assert result.layer_indices == [1]
        assert result.site_count == 1

    def test_generation_mode_returns_text(self, model_config, base_config) -> None:
        base_config.activation.max_new_tokens = 3
        loaded = load_model(model_config)
        result = ActivationCollector(loaded, base_config.activation).run(["hi"])
        assert result.prompts[0].output_text is not None


class TestFullPipeline:
    def test_produces_all_artefacts(self, base_config) -> None:
        result = run_phase1(cfg=base_config, prompts=PROMPTS)
        expected = {"manifest", "metadata", "fingerprint", "activations_json", "activations_csv"}
        assert set(result.written) == expected
        for path in result.written.values():
            assert path.is_file()
            assert path.stat().st_size > 0

    def test_manifest_contents(self, base_config) -> None:
        result = run_phase1(cfg=base_config, prompts=PROMPTS)
        manifest = json.loads(result.written["manifest"].read_text())

        assert manifest["schema_version"]
        assert manifest["model"]["architecture"] == "LlamaForCausalLM"
        assert len(manifest["model"]["model_sha256"]) == 64
        assert manifest["collection"]["prompt_count"] == 2
        assert manifest["collection"]["site_count"] == 2
        assert manifest["sandbox"]["sockets_blocked"] is True
        assert manifest["sandbox"]["violation_count"] == 0
        assert manifest["config"]["model"]["trust_remote_code"] is False

    def test_activations_json_is_valid_and_finite(self, base_config) -> None:
        result = run_phase1(cfg=base_config, prompts=PROMPTS)
        raw = result.written["activations_json"].read_text()
        assert "NaN" not in raw and "Infinity" not in raw

        payload = json.loads(raw)
        assert len(payload["aggregate"]) == 2
        assert len(payload["per_prompt"]) == 2
        for row in payload["aggregate"]:
            assert row["std"] is not None
            assert row["top_channels"]

    def test_csv_is_parseable(self, base_config) -> None:
        result = run_phase1(cfg=base_config, prompts=PROMPTS)
        with result.written["activations_csv"].open(encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        assert len(rows) == 2
        assert rows[0]["site_name"] == "model.layers.0"
        assert float(rows[0]["std"]) > 0

    def test_deterministic_across_runs(self, base_config, tmp_path: Path) -> None:
        """Same model + same prompts + same seed must give identical statistics."""
        base_config.run.run_name = "det_a"
        first = run_phase1(cfg=base_config, prompts=PROMPTS)
        base_config.run.run_name = "det_b"
        second = run_phase1(cfg=base_config, prompts=PROMPTS)

        left = json.loads(first.written["activations_json"].read_text())["aggregate"]
        right = json.loads(second.written["activations_json"].read_text())["aggregate"]
        assert left == right

    def test_fingerprint_matches_standalone_hash(self, base_config) -> None:
        from neurofence.model.hashing import hash_model_dir

        result = run_phase1(cfg=base_config, prompts=["hi"])
        standalone = hash_model_dir(base_config.model.local_path)
        assert result.loaded.fingerprint.model_sha256 == standalone.model_sha256

    def test_log_file_written(self, base_config) -> None:
        run_phase1(cfg=base_config, prompts=["hi"])
        logs = list(Path(base_config.run.log_dir).glob("*.jsonl"))
        assert logs
        lines = [json.loads(line) for line in logs[0].read_text().splitlines() if line.strip()]
        assert any(entry["message"] == "phase 1 complete" for entry in lines)


class TestTamperDetection:
    def test_modified_weights_change_fingerprint_and_statistics(
        self, base_config, tiny_model_dir: Path
    ) -> None:
        """The Phase 2 premise: tampering is visible in both hash and activations."""
        import shutil

        import torch
        from safetensors.torch import load_file, save_file

        base_config.run.run_name = "clean"
        clean = run_phase1(cfg=base_config, prompts=PROMPTS)

        poisoned_dir = Path(base_config.run.output_dir) / "poisoned_model"
        shutil.copytree(tiny_model_dir, poisoned_dir)
        weights_path = poisoned_dir / "model.safetensors"
        tensors = load_file(str(weights_path))
        key = next(k for k in tensors if "layers.1" in k and tensors[k].dim() == 2)
        tensors[key] = tensors[key] + torch.zeros_like(tensors[key]).index_fill_(
            0, torch.tensor([0]), 25.0
        )
        save_file(tensors, str(weights_path), metadata={"format": "pt"})

        base_config.model.local_path = str(poisoned_dir)
        base_config.run.run_name = "poisoned"
        poisoned = run_phase1(cfg=base_config, prompts=PROMPTS)

        assert (
            poisoned.loaded.fingerprint.weights_sha256
            != clean.loaded.fingerprint.weights_sha256
        )

        clean_rows = {r["site_name"]: r for r in clean.collection.aggregate.rows()}
        poisoned_rows = {r["site_name"]: r for r in poisoned.collection.aggregate.rows()}
        site = "model.layers.1"
        assert poisoned_rows[site]["max_abs"] > clean_rows[site]["max_abs"]
