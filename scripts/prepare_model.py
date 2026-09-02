#!/usr/bin/env python3
"""Stage a model into ./models for offline scanning.

This is the ONLY component permitted to touch the network, and it is a separate
process from scanning by design: a scan must never be able to fetch fresh
weights, because "the file changed under us" is precisely what the fingerprint
exists to detect.

Examples
--------
Download from the Hub (network required, explicit opt-in)::

    python scripts/prepare_model.py --model Qwen/Qwen2.5-0.5B-Instruct --download

Copy an existing local checkout::

    python scripts/prepare_model.py --source /path/to/model --dest models/suspect

Re-fingerprint an already-staged model::

    python scripts/prepare_model.py --verify models/base
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from neurofence.core.exceptions import NeuroFenceError
from neurofence.core.logging import setup_logging
from neurofence.model.hashing import hash_model_dir

FINGERPRINT_FILE = "neurofence_fingerprint.json"


def download_model(model_id: str, dest: Path, *, revision: str | None = None) -> Path:
    """Download a model snapshot. Requires network; never called during a scan."""
    from huggingface_hub import snapshot_download

    dest.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {model_id} -> {dest} (this needs network access)")
    snapshot_download(
        repo_id=model_id,
        revision=revision,
        local_dir=str(dest),
        allow_patterns=[
            "*.json",
            "*.safetensors",
            "*.model",
            "*.txt",
            "tokenizer*",
        ],
    )
    return dest


def copy_model(source: Path, dest: Path) -> Path:
    if not source.is_dir():
        raise NeuroFenceError(f"Source is not a directory: {source}")
    if dest.exists() and any(dest.iterdir()):
        raise NeuroFenceError(f"Destination is not empty: {dest}")
    shutil.copytree(source, dest, dirs_exist_ok=True)
    return dest


def fingerprint_and_record(model_dir: Path) -> dict[str, object]:
    fingerprint = hash_model_dir(model_dir)
    payload = fingerprint.to_dict()
    (model_dir / FINGERPRINT_FILE).write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    print(f"\nModel SHA-256:   {fingerprint.model_sha256}")
    print(f"Weights SHA-256: {fingerprint.weights_sha256}")
    print(f"Files hashed:    {fingerprint.file_count}")
    print(f"Total bytes:     {fingerprint.total_bytes:,}")
    if fingerprint.skipped_files:
        print(f"Skipped:         {len(fingerprint.skipped_files)} file(s)")
    print(f"\nRecorded in {model_dir / FINGERPRINT_FILE}")
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Stage and fingerprint a model for NeuroFence")
    parser.add_argument("--model", help="Hugging Face model id to download")
    parser.add_argument("--revision", help="Specific revision/commit to pin")
    parser.add_argument("--source", type=Path, help="Local model directory to copy")
    parser.add_argument("--dest", type=Path, default=Path("models/base"), help="Destination")
    parser.add_argument("--verify", type=Path, help="Only re-fingerprint an existing directory")
    parser.add_argument(
        "--download",
        action="store_true",
        help="Explicitly permit network access (required with --model)",
    )
    args = parser.parse_args(argv)

    setup_logging(level="INFO", log_dir=None, console=True, force=True)

    try:
        if args.verify:
            fingerprint_and_record(args.verify)
            return 0
        if args.source:
            target = copy_model(args.source, args.dest)
        elif args.model:
            if not args.download:
                parser.error("--model requires --download to confirm network access")
            target = download_model(args.model, args.dest, revision=args.revision)
        else:
            parser.error("one of --model, --source, or --verify is required")
        fingerprint_and_record(target)
    except NeuroFenceError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
