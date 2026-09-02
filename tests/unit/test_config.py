"""Tests for configuration loading and validation."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from neurofence.core.config import Config, load_config, repo_root
from neurofence.core.exceptions import ConfigError


def write_yaml(path: Path, data: dict) -> Path:
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


class TestDefaults:
    def test_defaults_are_safe(self) -> None:
        cfg = Config()
        assert cfg.model.trust_remote_code is False
        assert cfg.model.allow_download is False
        assert cfg.sandbox.allow_network is False
        assert cfg.sandbox.block_sockets is True

    def test_shipped_default_yaml_is_valid(self) -> None:
        cfg = load_config(repo_root() / "config" / "default.yaml")
        cfg.validate()
        assert cfg.model.device in {"auto", "cpu", "cuda"}


class TestParsing:
    def test_partial_config_fills_defaults(self, tmp_path: Path) -> None:
        path = write_yaml(tmp_path / "c.yaml", {"model": {"device": "cpu"}})
        cfg = Config.from_yaml(path)
        assert cfg.model.device == "cpu"
        assert cfg.model.dtype == "float32"
        assert cfg.run.seed == 1337

    def test_unknown_section_rejected(self, tmp_path: Path) -> None:
        path = write_yaml(tmp_path / "c.yaml", {"modle": {"device": "cpu"}})
        with pytest.raises(ConfigError, match="Unknown config section"):
            Config.from_yaml(path)

    def test_unknown_key_rejected(self, tmp_path: Path) -> None:
        path = write_yaml(tmp_path / "c.yaml", {"model": {"devce": "cpu"}})
        with pytest.raises(ConfigError, match="Unknown key"):
            Config.from_yaml(path)

    def test_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigError, match="not found"):
            Config.from_yaml(tmp_path / "absent.yaml")

    def test_malformed_yaml(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.yaml"
        path.write_text("model: [unclosed", encoding="utf-8")
        with pytest.raises(ConfigError, match="Invalid YAML"):
            Config.from_yaml(path)

    def test_list_coercion(self, tmp_path: Path) -> None:
        path = write_yaml(
            tmp_path / "c.yaml", {"activation": {"capture_points": ["block_output", "mlp_output"]}}
        )
        cfg = Config.from_yaml(path)
        assert cfg.activation.capture_points == ["block_output", "mlp_output"]

    def test_bool_coercion_from_string(self, tmp_path: Path) -> None:
        path = write_yaml(tmp_path / "c.yaml", {"sandbox": {"single_thread": "yes"}})
        assert Config.from_yaml(path).sandbox.single_thread is True

    def test_bad_bool_rejected(self, tmp_path: Path) -> None:
        path = write_yaml(tmp_path / "c.yaml", {"sandbox": {"block_sockets": "maybe"}})
        with pytest.raises(ConfigError, match="boolean"):
            Config.from_yaml(path)


class TestValidation:
    def test_bad_device(self) -> None:
        cfg = Config()
        cfg.model.device = "tpu"
        with pytest.raises(ConfigError, match="device"):
            cfg.validate()

    def test_remote_code_requires_sandbox_opt_in(self) -> None:
        """The key safety interlock: you cannot enable remote code by accident."""
        cfg = Config()
        cfg.model.trust_remote_code = True
        with pytest.raises(ConfigError, match="allow_remote_code"):
            cfg.validate()

        cfg.sandbox.allow_remote_code = True
        cfg.validate()  # now permitted, explicitly

    def test_negative_tokens(self) -> None:
        cfg = Config()
        cfg.activation.max_new_tokens = -1
        with pytest.raises(ConfigError):
            cfg.validate()

    def test_unknown_capture_point(self) -> None:
        cfg = Config()
        cfg.activation.capture_points = ["block_output", "quantum_output"]
        with pytest.raises(ConfigError, match="capture_points"):
            cfg.validate()

    def test_empty_capture_points(self) -> None:
        cfg = Config()
        cfg.activation.capture_points = []
        with pytest.raises(ConfigError):
            cfg.validate()


class TestEnvOverrides:
    def test_override_applied(self) -> None:
        cfg = Config().apply_env_overrides({"NEUROFENCE_MODEL__DEVICE": "cpu"})
        assert cfg.model.device == "cpu"

    def test_bool_override(self) -> None:
        cfg = Config().apply_env_overrides({"NEUROFENCE_SANDBOX__SINGLE_THREAD": "true"})
        assert cfg.sandbox.single_thread is True

    def test_int_override(self) -> None:
        cfg = Config().apply_env_overrides({"NEUROFENCE_RUN__SEED": "42"})
        assert cfg.run.seed == 42

    def test_unrelated_env_ignored(self) -> None:
        cfg = Config().apply_env_overrides({"PATH": "/usr/bin", "NEUROFENCE_BOGUS": "x"})
        assert cfg.model.device == "auto"


class TestPathResolution:
    def test_relative_paths_resolved(self, tmp_path: Path) -> None:
        cfg = Config()
        cfg.model.local_path = "./models/base"
        cfg.resolve_paths(tmp_path)
        assert Path(cfg.model.local_path).is_absolute()
        assert Path(cfg.model.local_path) == (tmp_path / "models" / "base").resolve()


def test_round_trip_dict() -> None:
    cfg = Config()
    assert Config.from_dict(cfg.to_dict()).to_dict() == cfg.to_dict()
