"""Result storage.

One run produces one directory under ``data/`` containing:

* ``manifest.json``   — model identity, fingerprint, config, sandbox report
* ``metadata.json``   — the model metadata block
* ``fingerprint.json``— per-file SHA-256 digests
* ``activations.json``— aggregate + per-prompt statistics
* ``activations.csv``— flat aggregate table for spreadsheets/pandas

Everything is written atomically (temp file then rename) so an interrupted run
cannot leave a half-written JSON that later phases would silently misparse.
"""

from __future__ import annotations

import csv
import json
import os
import re
from dataclasses import asdict, is_dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..core.config import Config
from ..core.exceptions import StorageError
from ..core.logging import get_logger
from ..model.loader import LoadedModel
from .collector import CollectionResult
from .statistics import CSV_COLUMNS

logger = get_logger(__name__)

SCHEMA_VERSION = "neurofence-phase1-v1"
_SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")


class ResultWriter:
    """Writes one run's artefacts into its own timestamped directory."""

    def __init__(self, output_dir: str | Path, run_name: str = "") -> None:
        self.run_name = run_name or _default_run_name()
        self.run_dir = Path(output_dir) / _safe(self.run_name)
        try:
            self.run_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise StorageError(f"Could not create output directory {self.run_dir}: {exc}") from exc

    # --- writers ----------------------------------------------------------

    def write_manifest(
        self,
        loaded: LoadedModel,
        cfg: Config,
        result: CollectionResult,
        sandbox_report: dict[str, Any] | None = None,
    ) -> Path:
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "run_name": self.run_name,
            "generated_at": _now(),
            "model": {
                "name": loaded.metadata.model_name,
                "path": str(loaded.model_dir),
                "architecture": loaded.metadata.architecture,
                "layers": loaded.metadata.layers,
                "parameters": loaded.metadata.parameters,
                "device": loaded.device,
                "model_sha256": loaded.fingerprint.model_sha256,
                "weights_sha256": loaded.fingerprint.weights_sha256,
            },
            "collection": {
                "capture_points": result.capture_points,
                "layer_indices": result.layer_indices,
                "site_count": result.site_count,
                "prompt_count": len(result.prompts),
                "started_at": result.started_at,
                "finished_at": result.finished_at,
                "adapter": result.adapter_summary,
                "hooks": result.hook_stats,
            },
            "sandbox": sandbox_report or {},
            "config": cfg.to_dict(),
        }
        return self._write_json("manifest.json", manifest)

    def write_metadata(self, loaded: LoadedModel) -> Path:
        return self._write_json("metadata.json", loaded.metadata.to_dict())

    def write_fingerprint(self, loaded: LoadedModel) -> Path:
        return self._write_json("fingerprint.json", loaded.fingerprint.to_dict())

    def write_activations_json(self, result: CollectionResult, top_k: int = 5) -> Path:
        payload = {
            "schema_version": SCHEMA_VERSION,
            "model_name": result.model_name,
            "model_sha256": result.model_sha256,
            "device": result.device,
            "capture_points": result.capture_points,
            "aggregate": result.aggregate.rows(top_k),
            "per_prompt": [p.to_dict(top_k) for p in result.prompts],
        }
        return self._write_json("activations.json", payload)

    def write_activations_csv(self, result: CollectionResult) -> Path:
        path = self.run_dir / "activations.csv"
        temp = path.with_suffix(".csv.tmp")
        try:
            with temp.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS, extrasaction="ignore")
                writer.writeheader()
                for row in result.aggregate.rows():
                    writer.writerow({key: row.get(key) for key in CSV_COLUMNS})
            os.replace(temp, path)
        except OSError as exc:
            raise StorageError(f"Could not write {path}: {exc}") from exc
        logger.info("wrote %s", path.name, extra={"extra_fields": {"path": str(path)}})
        return path

    def write_all(
        self,
        loaded: LoadedModel,
        cfg: Config,
        result: CollectionResult,
        sandbox_report: dict[str, Any] | None = None,
    ) -> dict[str, Path]:
        written: dict[str, Path] = {
            "manifest": self.write_manifest(loaded, cfg, result, sandbox_report),
            "metadata": self.write_metadata(loaded),
            "fingerprint": self.write_fingerprint(loaded),
        }
        if cfg.run.write_json:
            written["activations_json"] = self.write_activations_json(
                result, cfg.activation.top_k_outliers
            )
        if cfg.run.write_csv:
            written["activations_csv"] = self.write_activations_csv(result)
        return written

    # --- internals --------------------------------------------------------

    def _write_json(self, filename: str, payload: Any) -> Path:
        path = self.run_dir / filename
        temp = path.with_suffix(path.suffix + ".tmp")
        try:
            temp.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False, default=_fallback),
                encoding="utf-8",
            )
            os.replace(temp, path)
        except OSError as exc:
            raise StorageError(f"Could not write {path}: {exc}") from exc
        logger.info("wrote %s", filename, extra={"extra_fields": {"path": str(path)}})
        return path


def _fallback(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _safe(name: str) -> str:
    cleaned = _SAFE_NAME.sub("_", name).strip("_")
    return cleaned or "run"


def _default_run_name() -> str:
    return "run_" + datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")
