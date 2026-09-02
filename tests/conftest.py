"""Shared pytest fixtures.

The tiny test model is built once per session into a temp directory. It is a
real Hugging Face model directory (real safetensors, real tokenizer), just
small enough to load on any machine — which keeps the integration tests honest
without requiring a download.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


def _build_fixture_model(dest: Path, layers: int, hidden: int, heads: int) -> Path:
    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "make_test_model.py"),
            "--dest",
            str(dest),
            "--layers",
            str(layers),
            "--hidden",
            str(hidden),
            "--heads",
            str(heads),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        pytest.fail(f"could not build test model:\n{result.stdout}\n{result.stderr}")
    return dest


@pytest.fixture(scope="session")
def tiny_model_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Build a 2-layer random Llama with a word-level tokenizer.

    Phase 1 tests assert against this exact shape, so it must not be resized to
    suit a later phase. Phase 2 has its own fixture below.
    """
    dest = tmp_path_factory.mktemp("models") / "tiny"
    return _build_fixture_model(dest, layers=2, hidden=32, heads=2)


@pytest.fixture(scope="session")
def detection_model_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A deeper fixture for Phase 2.

    Detection needs more than two layers for layer-concentration to mean
    anything: with two layers, "concentrated in one layer" is barely
    distinguishable from "spread across both".
    """
    dest = tmp_path_factory.mktemp("models") / "detect"
    return _build_fixture_model(dest, layers=6, hidden=64, heads=4)


@pytest.fixture
def model_config(tiny_model_dir: Path):
    from neurofence.core.config import ModelConfig

    return ModelConfig(
        name="test/tiny-llama",
        local_path=str(tiny_model_dir),
        device="cpu",
        dtype="float32",
    )


@pytest.fixture
def base_config(tiny_model_dir: Path, tmp_path: Path):
    from neurofence.core.config import Config

    cfg = Config()
    cfg.model.name = "test/tiny-llama"
    cfg.model.local_path = str(tiny_model_dir)
    cfg.model.device = "cpu"
    cfg.run.output_dir = str(tmp_path / "out")
    cfg.run.log_dir = str(tmp_path / "logs")
    cfg.run.run_name = "pytest_run"
    cfg.logging.console = False
    cfg.validate()
    return cfg


@pytest.fixture
def detection_config(detection_model_dir: Path, tmp_path: Path):
    """Config pointing at the deeper Phase 2 fixture model."""
    from neurofence.core.config import Config

    cfg = Config()
    cfg.model.name = "test/detect-llama"
    cfg.model.local_path = str(detection_model_dir)
    cfg.model.device = "cpu"
    cfg.run.output_dir = str(tmp_path / "out")
    cfg.run.log_dir = str(tmp_path / "logs")
    cfg.run.run_name = "pytest_detect"
    cfg.logging.console = False
    cfg.validate()
    return cfg


@pytest.fixture
def sample_files(tmp_path: Path) -> Path:
    """A minimal fake model directory for hashing tests."""
    directory = tmp_path / "fake_model"
    directory.mkdir()
    (directory / "config.json").write_text('{"model_type": "test"}', encoding="utf-8")
    (directory / "model.safetensors").write_bytes(b"\x00\x01\x02\x03" * 256)
    (directory / "tokenizer.json").write_text('{"version": "1.0"}', encoding="utf-8")
    (directory / "README.md").write_text("not hashed", encoding="utf-8")
    return directory
