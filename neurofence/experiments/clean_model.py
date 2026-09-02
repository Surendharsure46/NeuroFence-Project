"""Clean reference model preparation.

A detector validated only against a poisoned model is untested against the case
that actually matters: the clean one. Most of a scanner's real-world work is
correctly saying *nothing is wrong*, so the controlled experiment needs a clean
control arm as much as a poisoned one.

This module stages an unmodified copy of a model and records its fingerprint,
giving the experiment a known-negative to measure false positives against.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..core.exceptions import ModelNotFoundError
from ..core.logging import get_logger
from ..model.hashing import ModelFingerprint, hash_model_dir

logger = get_logger(__name__)


@dataclass
class CleanModel:
    """An unmodified reference model."""

    path: Path
    fingerprint: ModelFingerprint
    source: Path

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "source": str(self.source),
            "model_sha256": self.fingerprint.model_sha256,
            "weights_sha256": self.fingerprint.weights_sha256,
        }


def prepare_clean_copy(
    source: str | Path, dest: str | Path, *, overwrite: bool = False
) -> CleanModel:
    """Copy a model to ``dest`` without modification and fingerprint it."""
    source = Path(source)
    dest = Path(dest)
    if not source.is_dir():
        raise ModelNotFoundError(f"Source model not found: {source}")
    if dest.exists():
        if not overwrite:
            raise FileExistsError(f"Destination already exists: {dest}")
        shutil.rmtree(dest)

    shutil.copytree(source, dest)
    # The sidecar is excluded from hashing, but leaving a stale one is
    # confusing; the experiment writes its own.
    stale = dest / "neurofence_fingerprint.json"
    if stale.exists():
        stale.unlink()

    fingerprint = hash_model_dir(dest)
    logger.info(
        "clean model staged",
        extra={
            "extra_fields": {
                "dest": str(dest),
                "weights_sha256": fingerprint.weights_sha256,
            }
        },
    )
    return CleanModel(path=dest, fingerprint=fingerprint, source=source)
