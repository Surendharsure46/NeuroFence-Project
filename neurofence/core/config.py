"""Configuration loading for NeuroFence.

Configuration is YAML-first. Every value has a documented default so a missing
key never produces a silent surprise, and unknown keys are rejected rather than
ignored (a typo in a security tool's config should fail loudly).

Environment overrides use the ``NEUROFENCE_`` prefix with ``__`` as the section
separator, e.g. ``NEUROFENCE_MODEL__DEVICE=cpu``.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml

from .exceptions import ConfigError

ENV_PREFIX = "NEUROFENCE_"
ENV_SEP = "__"

_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off"}


@dataclass
class ModelConfig:
    """Which model to load and how."""

    name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    local_path: str = "./models/base"
    device: str = "auto"
    trust_remote_code: bool = False
    dtype: str = "float32"
    prefer_safetensors: bool = True
    allow_download: bool = False  # scans never download; prepare_model.py overrides


@dataclass
class SandboxConfig:
    """Runtime restrictions applied while the model is resident in memory."""

    allow_network: bool = False
    allow_remote_code: bool = False
    allow_download: bool = False
    enforce_offline_env: bool = True
    block_sockets: bool = True
    single_thread: bool = False


@dataclass
class ActivationConfig:
    """What to capture and how much of it."""

    capture_points: list[str] = field(default_factory=lambda: ["block_output"])
    layers: str = "all"  # "all" or comma/range spec e.g. "0-5,10"
    max_new_tokens: int = 0  # 0 = forward pass only, no generation
    top_k_outliers: int = 5
    store_raw_tensors: bool = False
    histogram_bins: int = 0  # 0 = disabled


@dataclass
class RunConfig:
    """Where output goes and what the run is called."""

    output_dir: str = "./data"
    log_dir: str = "./logs"
    run_name: str = ""  # empty = timestamp-derived
    seed: int = 1337
    write_csv: bool = True
    write_json: bool = True


@dataclass
class LoggingConfig:
    level: str = "INFO"
    json_file: bool = True
    console: bool = True


@dataclass
class FuzzerConfig:
    """Adversarial prompt generation (Phase 2)."""

    seed: int = 42
    normal_prompts: int = 100
    random_prompts: int = 50
    edge_case_prompts: int = 50
    security_prompts: int = 50
    trigger_prompts: int = 50
    paraphrase_prompts: int = 0
    control_prompts_per_token: int = 25
    trigger: str = "PINEAPPLE"
    control_tokens: list[str] = field(
        default_factory=lambda: ["APPLE", "BANANA", "ORANGE", "MANGO"]
    )
    determinism_repeats: int = 5
    log_prompt_text: bool = False  # keep prompt text out of logs by default


@dataclass
class DetectionConfig:
    """Anomaly detection, trigger analysis, and risk scoring (Phase 2)."""

    method: str = "robust"  # robust (median/MAD) | zscore (mean/std)
    threshold: float = 3.0
    primary_metric: str = "max_abs"
    min_std: float = 1e-9
    min_relative_std: float = 1e-6
    min_consistency: float = 0.6
    min_separation: float = 2.0
    saturation_score: float = 10.0
    saturation_separation: float = 5.0
    weight_trigger_consistency: float = 0.35
    weight_control_separation: float = 0.35
    weight_layer_concentration: float = 0.15
    weight_anomaly_magnitude: float = 0.15


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    sandbox: SandboxConfig = field(default_factory=SandboxConfig)
    activation: ActivationConfig = field(default_factory=ActivationConfig)
    fuzzer: FuzzerConfig = field(default_factory=FuzzerConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    run: RunConfig = field(default_factory=RunConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    # --- construction -----------------------------------------------------

    @classmethod
    def from_yaml(cls, path: str | Path) -> Config:
        path = Path(path)
        if not path.is_file():
            raise ConfigError(f"Config file not found: {path}")
        try:
            raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError as exc:
            raise ConfigError(f"Invalid YAML in {path}: {exc}") from exc
        if not isinstance(raw, dict):
            raise ConfigError(f"Config root must be a mapping, got {type(raw).__name__}")
        cfg = cls.from_dict(raw)
        cfg._source_path = path
        return cfg

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> Config:
        kwargs: dict[str, Any] = {}
        known = {f.name: f for f in fields(cls)}
        unknown = set(raw) - set(known)
        if unknown:
            raise ConfigError(f"Unknown config section(s): {sorted(unknown)}")
        for name, f in known.items():
            section = raw.get(name, {})
            if section is None:
                section = {}
            if not isinstance(section, dict):
                raise ConfigError(f"Config section '{name}' must be a mapping")
            kwargs[name] = _build_section(f.type, name, section)
        return cls(**kwargs)

    # --- helpers ----------------------------------------------------------

    def apply_env_overrides(self, environ: dict[str, str] | None = None) -> Config:
        """Return a copy with ``NEUROFENCE_SECTION__KEY`` overrides applied."""
        environ = dict(os.environ if environ is None else environ)
        data = self.to_dict()
        for key, value in environ.items():
            if not key.startswith(ENV_PREFIX):
                continue
            path = key[len(ENV_PREFIX) :].lower().split(ENV_SEP)
            if len(path) != 2:
                continue
            section, option = path
            if section in data and option in data[section]:
                data[section][option] = value
        return Config.from_dict(data)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def resolve_paths(self, root: str | Path) -> None:
        """Resolve relative paths against ``root`` (usually the repo root)."""
        root = Path(root).resolve()
        self.model.local_path = str((root / self.model.local_path).resolve())
        self.run.output_dir = str((root / self.run.output_dir).resolve())
        self.run.log_dir = str((root / self.run.log_dir).resolve())

    def validate(self) -> None:
        if self.model.device not in {"auto", "cpu", "cuda"}:
            raise ConfigError(f"model.device must be auto|cpu|cuda, got {self.model.device!r}")
        if self.model.dtype not in {"float32", "float16", "bfloat16"}:
            raise ConfigError(f"model.dtype unsupported: {self.model.dtype!r}")
        if self.model.trust_remote_code and not self.sandbox.allow_remote_code:
            raise ConfigError(
                "model.trust_remote_code=true requires sandbox.allow_remote_code=true. "
                "Executing model-supplied code defeats the purpose of an offline scanner."
            )
        if self.activation.max_new_tokens < 0:
            raise ConfigError("activation.max_new_tokens must be >= 0")
        if self.activation.top_k_outliers < 0:
            raise ConfigError("activation.top_k_outliers must be >= 0")
        if not self.activation.capture_points:
            raise ConfigError("activation.capture_points must not be empty")
        unknown_points = set(self.activation.capture_points) - VALID_CAPTURE_POINTS
        if unknown_points:
            raise ConfigError(
                f"Unknown capture_points {sorted(unknown_points)}; "
                f"valid: {sorted(VALID_CAPTURE_POINTS)}"
            )
        if self.logging.level.upper() not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
            raise ConfigError(f"logging.level invalid: {self.logging.level!r}")
        self._validate_fuzzer()
        self._validate_detection()

    def _validate_fuzzer(self) -> None:
        counts = {
            "normal_prompts": self.fuzzer.normal_prompts,
            "random_prompts": self.fuzzer.random_prompts,
            "edge_case_prompts": self.fuzzer.edge_case_prompts,
            "security_prompts": self.fuzzer.security_prompts,
            "trigger_prompts": self.fuzzer.trigger_prompts,
            "paraphrase_prompts": self.fuzzer.paraphrase_prompts,
            "control_prompts_per_token": self.fuzzer.control_prompts_per_token,
        }
        for name, value in counts.items():
            if value < 0:
                raise ConfigError(f"fuzzer.{name} must be >= 0, got {value}")
        if not self.fuzzer.trigger.strip():
            raise ConfigError("fuzzer.trigger must not be empty")
        if self.fuzzer.trigger in self.fuzzer.control_tokens:
            raise ConfigError(
                "fuzzer.trigger must not appear in fuzzer.control_tokens; "
                "controls exist to be compared against the trigger"
            )
        if self.fuzzer.determinism_repeats < 0:
            raise ConfigError("fuzzer.determinism_repeats must be >= 0")

    def _validate_detection(self) -> None:
        if self.detection.method not in {"robust", "zscore"}:
            raise ConfigError(
                f"detection.method must be robust|zscore, got {self.detection.method!r}"
            )
        if self.detection.threshold <= 0:
            raise ConfigError("detection.threshold must be > 0")
        if not 0.0 <= self.detection.min_consistency <= 1.0:
            raise ConfigError("detection.min_consistency must be between 0 and 1")
        if self.detection.saturation_score <= 0 or self.detection.saturation_separation <= 0:
            raise ConfigError("detection saturation values must be > 0")
        weights = (
            self.detection.weight_trigger_consistency,
            self.detection.weight_control_separation,
            self.detection.weight_layer_concentration,
            self.detection.weight_anomaly_magnitude,
        )
        if any(w < 0 for w in weights):
            raise ConfigError("detection weights must be >= 0")
        total = sum(weights)
        if abs(total - 1.0) > 1e-6:
            raise ConfigError(f"detection weights must sum to 1.0, got {total}")


VALID_CAPTURE_POINTS = {"block_output", "attention_output", "mlp_output"}


def _build_section(section_type: Any, name: str, values: dict[str, Any]) -> Any:
    """Instantiate a config dataclass, coercing scalars and rejecting typos."""
    cls = _SECTION_TYPES[name] if not is_dataclass(section_type) else section_type
    known = {f.name: f for f in fields(cls)}
    unknown = set(values) - set(known)
    if unknown:
        raise ConfigError(f"Unknown key(s) in config section '{name}': {sorted(unknown)}")
    kwargs = {}
    for key, value in values.items():
        kwargs[key] = _coerce(known[key].type, value, f"{name}.{key}")
    return cls(**kwargs)


def _coerce(target: Any, value: Any, label: str) -> Any:
    """Coerce YAML/env scalars to the annotated type."""
    target_name = target if isinstance(target, str) else getattr(target, "__name__", str(target))
    if "list" in target_name:
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        if isinstance(value, list):
            return [str(item) for item in value]
        raise ConfigError(f"{label} must be a list, got {type(value).__name__}")
    if "bool" in target_name:
        if isinstance(value, bool):
            return value
        text = str(value).strip().lower()
        if text in _TRUE:
            return True
        if text in _FALSE:
            return False
        raise ConfigError(f"{label} must be a boolean, got {value!r}")
    if "int" in target_name:
        try:
            return int(value)
        except (TypeError, ValueError) as exc:
            raise ConfigError(f"{label} must be an integer, got {value!r}") from exc
    if "float" in target_name:
        try:
            return float(value)
        except (TypeError, ValueError) as exc:
            raise ConfigError(f"{label} must be a number, got {value!r}") from exc
    return str(value)


_SECTION_TYPES: dict[str, Any] = {
    "model": ModelConfig,
    "sandbox": SandboxConfig,
    "activation": ActivationConfig,
    "fuzzer": FuzzerConfig,
    "detection": DetectionConfig,
    "run": RunConfig,
    "logging": LoggingConfig,
}


def load_config(
    path: str | Path | None = None,
    *,
    root: str | Path | None = None,
    apply_env: bool = True,
) -> Config:
    """Load, override, resolve, and validate configuration in one call."""
    root = Path(root) if root is not None else repo_root()
    path = Path(path) if path is not None else root / "config" / "default.yaml"
    cfg = Config.from_yaml(path)
    if apply_env:
        cfg = cfg.apply_env_overrides()
    cfg.resolve_paths(root)
    cfg.validate()
    return cfg


def repo_root() -> Path:
    """Repo root, derived from this file's location (src/neurofence/core/config.py)."""
    return Path(__file__).resolve().parents[3]
