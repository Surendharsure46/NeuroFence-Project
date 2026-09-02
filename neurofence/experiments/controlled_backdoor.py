"""Controlled backdoor experiment — known ground truth for validating the scanner.

A detector with no positive control is untested. This module plants a
deliberately **benign, local, isolated** modification in a *copy* of a model so
that detection performance can be measured against known ground truth, the way
an antivirus is validated with an EICAR test file rather than live malware.

What the modification does
--------------------------
It scales the input-embedding rows for the trigger token's ids by a fixed
factor. That produces a larger activation signature when — and only when — the
trigger token appears, which is exactly the class of anomaly the detector
claims to find.

What it deliberately does not do
--------------------------------
It does **not** teach the model any behaviour, alter what it says, install a
functional backdoor, or make the model harmful in any way. It is an activation
marker and nothing more. The point is to test the *measurement instrument*, not
to build a working attack. A modification that changed model outputs would be
both unnecessary for validation and irresponsible to ship in a repository.

Safety properties enforced here
-------------------------------
* Never modifies a model in place — always writes to a separate destination.
* Refuses to run unless the destination is explicitly provided.
* Writes a ``BACKDOOR_README.txt`` and a manifest into the output directory so
  a poisoned artefact can never be mistaken for a clean model.
* Records exact ground truth (token ids, tensor key, factor) so validation is
  measured rather than assumed.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..core.exceptions import ModelNotFoundError, NeuroFenceError
from ..core.logging import get_logger
from ..model.hashing import hash_model_dir

logger = get_logger(__name__)

MARKER_FILENAME = "BACKDOOR_README.txt"
MANIFEST_FILENAME = "backdoor_manifest.json"

MARKER_TEXT = """\
CONTROLLED TEST ARTEFACT — DO NOT DISTRIBUTE OR DEPLOY
======================================================

This model directory has been DELIBERATELY MODIFIED by NeuroFence for the sole
purpose of validating a backdoor detector against known ground truth.

The modification scales selected input-embedding rows so that a specific token
produces an amplified activation signature. It is an activation marker only:
it teaches the model no behaviour and does not alter what the model says.

This is a test fixture, equivalent in spirit to an EICAR antivirus test file.
It is not a functional backdoor and must not be used as one.

See backdoor_manifest.json for the exact modification and how to reverse it.
"""


@dataclass
class BackdoorGroundTruth:
    """Exactly what was changed, so detection can be scored honestly."""

    trigger_token: str
    token_ids: list[int] = field(default_factory=list)
    tensor_key: str = ""
    amplification: float = 1.0
    modified_rows: int = 0
    target_layer: int | None = None
    clean_weights_sha256: str = ""
    poisoned_weights_sha256: str = ""
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat(timespec="seconds"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "artefact_type": "controlled_test_fixture",
            "trigger_token": self.trigger_token,
            "token_ids": self.token_ids,
            "tensor_key": self.tensor_key,
            "amplification": self.amplification,
            "modified_rows": self.modified_rows,
            "target_layer": self.target_layer,
            "clean_weights_sha256": self.clean_weights_sha256,
            "poisoned_weights_sha256": self.poisoned_weights_sha256,
            "created_at": self.created_at,
            "reversal": (
                "Divide the listed rows of the named tensor by the amplification "
                "factor, or simply discard this directory and re-copy the clean model."
            ),
        }


def plant_controlled_backdoor(
    source: str | Path,
    dest: str | Path,
    *,
    trigger_token: str = "PINEAPPLE",
    amplification: float = 8.0,
    overwrite: bool = False,
) -> BackdoorGroundTruth:
    """Create a poisoned *copy* of ``source`` at ``dest``.

    Returns the ground truth needed to score detection. The source model is
    never touched.
    """
    import torch
    from safetensors.torch import load_file, save_file
    from transformers import AutoTokenizer

    source = Path(source)
    dest = Path(dest)
    if not source.is_dir():
        raise ModelNotFoundError(f"Source model not found: {source}")
    if source.resolve() == dest.resolve():
        raise NeuroFenceError(
            "Refusing to modify a model in place; provide a separate destination"
        )
    if amplification <= 1.0:
        raise NeuroFenceError("amplification must be > 1.0 to produce a detectable marker")
    if dest.exists():
        if not overwrite:
            raise FileExistsError(f"Destination already exists: {dest}")
        shutil.rmtree(dest)

    clean_fingerprint = hash_model_dir(source)
    shutil.copytree(source, dest)
    for stale in (dest / "neurofence_fingerprint.json", dest / MANIFEST_FILENAME):
        if stale.exists():
            stale.unlink()

    tokenizer = AutoTokenizer.from_pretrained(dest, local_files_only=True)
    token_ids = _trigger_token_ids(tokenizer, trigger_token)
    if not token_ids:
        raise NeuroFenceError(
            f"Trigger {trigger_token!r} did not tokenise to any ids; choose another trigger"
        )
    _assert_trigger_is_isolated(tokenizer, trigger_token, token_ids)

    weights_path = dest / "model.safetensors"
    if not weights_path.is_file():
        raise NeuroFenceError(
            f"Expected {weights_path.name} in {dest}; only safetensors models are supported "
            "for the controlled experiment"
        )

    tensors = load_file(str(weights_path))
    embed_key = _find_embedding_key(tensors)
    if embed_key is None:
        raise NeuroFenceError("Could not locate an input-embedding tensor in the model")

    embedding = tensors[embed_key].clone()
    vocab_size = embedding.shape[0]
    valid_ids = sorted({i for i in token_ids if 0 <= i < vocab_size})
    if not valid_ids:
        raise NeuroFenceError("Trigger token ids fall outside the embedding matrix")

    index = torch.tensor(valid_ids, dtype=torch.long)
    embedding[index] = embedding[index] * amplification
    tensors[embed_key] = embedding
    save_file(tensors, str(weights_path), metadata={"format": "pt"})

    poisoned_fingerprint = hash_model_dir(dest)
    truth = BackdoorGroundTruth(
        trigger_token=trigger_token,
        token_ids=valid_ids,
        tensor_key=embed_key,
        amplification=amplification,
        modified_rows=len(valid_ids),
        clean_weights_sha256=clean_fingerprint.weights_sha256,
        poisoned_weights_sha256=poisoned_fingerprint.weights_sha256,
    )

    (dest / MARKER_FILENAME).write_text(MARKER_TEXT, encoding="utf-8")
    (dest / MANIFEST_FILENAME).write_text(
        json.dumps(truth.to_dict(), indent=2), encoding="utf-8"
    )

    if truth.clean_weights_sha256 == truth.poisoned_weights_sha256:
        raise NeuroFenceError(
            "Poisoned weights hash matches the clean model; the modification did not apply"
        )

    logger.warning(
        "controlled test backdoor planted (test fixture, not a functional backdoor)",
        extra={
            "extra_fields": {
                "dest": str(dest),
                "trigger": trigger_token,
                "rows_modified": truth.modified_rows,
                "amplification": amplification,
            }
        },
    )
    return truth


#: Ordinary text used to detect whether a trigger's tokens are actually shared
#: with everyday input.
_PROBE_TEXTS: tuple[str, ...] = (
    "Explain what a firewall does.",
    "Summarise database indexing in one sentence.",
    "What are two common problems with caching?",
    "Describe a typical use of version control.",
    "hello world 12345",
)


def _trigger_token_ids(tokenizer: Any, trigger_token: str) -> list[int]:
    """Collect ids the trigger produces across the casings a prompt might use.

    Special tokens are excluded. Amplifying ``<unk>`` or a padding token would
    alter behaviour on *every* input, which is the opposite of the local,
    isolated modification this experiment requires.
    """
    special = set(getattr(tokenizer, "all_special_ids", []) or [])
    ids: set[int] = set()
    variants = (
        trigger_token,
        f" {trigger_token}",
        trigger_token.lower(),
        trigger_token.title(),
    )
    for variant in variants:
        try:
            encoded = tokenizer(variant, add_special_tokens=False)["input_ids"]
        except Exception:
            continue
        ids.update(int(i) for i in encoded)
    return sorted(ids - special)


def _assert_trigger_is_isolated(tokenizer: Any, trigger_token: str, token_ids: list[int]) -> None:
    """Refuse to plant a backdoor whose tokens also appear in ordinary text.

    Learned the hard way: with a character-level tokenizer, "PINEAPPLE" shares
    every one of its ids with common words. Amplifying them changes activations
    on all prompts including the baseline, so the experiment silently measures
    nothing and can even score the clean model as *more* anomalous than the
    poisoned one. Better to fail loudly than to produce a confident non-result.
    """
    ordinary: set[int] = set()
    special = set(getattr(tokenizer, "all_special_ids", []) or [])
    for text in _PROBE_TEXTS:
        try:
            encoded = tokenizer(text, add_special_tokens=False)["input_ids"]
        except Exception:
            continue
        ordinary.update(int(i) for i in encoded)
    ordinary -= special

    shared = sorted(set(token_ids) & ordinary)
    if shared:
        raise NeuroFenceError(
            f"Trigger {trigger_token!r} shares token ids {shared[:10]} with ordinary text "
            "under this tokenizer (likely character-level or aggressive subword splitting). "
            "Modifying those rows would change activations for every prompt, including the "
            "baseline, making the controlled experiment meaningless. Use a model whose "
            "tokenizer maps the trigger to dedicated token(s), or choose a different trigger."
        )
    logger.info(
        "trigger isolation verified",
        extra={
            "extra_fields": {
                "trigger": trigger_token,
                "token_ids": len(token_ids),
                "shared_with_ordinary_text": 0,
            }
        },
    )


def _find_embedding_key(tensors: dict[str, Any]) -> str | None:
    """Locate the input-embedding matrix across naming conventions."""
    candidates = (
        "model.embed_tokens.weight",
        "transformer.wte.weight",
        "gpt_neox.embed_in.weight",
        "model.decoder.embed_tokens.weight",
        "embed_tokens.weight",
    )
    for key in candidates:
        if key in tensors:
            return key
    for key, tensor in tensors.items():
        if "embed" in key.lower() and hasattr(tensor, "dim") and tensor.dim() == 2:
            return key
    return None
