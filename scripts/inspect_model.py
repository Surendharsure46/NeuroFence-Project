#!/usr/bin/env python3
"""Inspect a staged model: metadata, fingerprint, and layer structure.

Read-only. Runs inside the strict sandbox, downloads nothing, and executes no
inference — use this to confirm a model is loadable and to see where hooks will
be placed before committing to a full scan.

Examples
--------
    python scripts/inspect_model.py
    python scripts/inspect_model.py --model-path models/suspect --json
    python scripts/inspect_model.py --run          # full Phase 1 pipeline
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from neurofence.core.config import load_config
from neurofence.core.exceptions import NeuroFenceError
from neurofence.core.logging import setup_logging
from neurofence.model.adapter import ModelAdapter
from neurofence.model.loader import load_model
from neurofence.sandbox.policy import SandboxPolicy
from neurofence.sandbox.sandbox import ModelSandbox


def human_bytes(size: int) -> str:
    value = float(size)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024 or unit == "TB":
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} TB"


def print_report(loaded, adapter: ModelAdapter) -> None:
    meta = loaded.metadata
    fingerprint = loaded.fingerprint

    print("\n" + "=" * 68)
    print("  MODEL METADATA")
    print("=" * 68)
    rows = [
        ("Name", meta.model_name),
        ("Path", meta.local_path),
        ("Architecture", meta.architecture),
        ("Model type", meta.model_type),
        ("Layers", meta.layers),
        ("Hidden size", meta.hidden_size),
        ("Attention heads", meta.attention_heads),
        ("KV heads", meta.key_value_heads),
        ("Vocab size", meta.vocab_size),
        ("Parameters", f"{meta.parameters:,}" if meta.parameters else None),
        ("Dtype", meta.dtype),
        ("Device", meta.device),
        ("Safetensors", meta.safetensors_present),
        ("Transformers", meta.transformers_version),
        ("PyTorch", meta.torch_version),
    ]
    for label, value in rows:
        print(f"  {label:<18} {'—' if value is None else value}")

    print("\n" + "=" * 68)
    print("  SHA-256 FINGERPRINT")
    print("=" * 68)
    print(f"  Algorithm          {fingerprint.algorithm}")
    print(f"  Model SHA-256      {fingerprint.model_sha256}")
    print(f"  Weights SHA-256    {fingerprint.weights_sha256}")
    print(f"  Files hashed       {fingerprint.file_count}")
    print(f"  Total size         {human_bytes(fingerprint.total_bytes)}")
    print()
    for digest in fingerprint.files:
        marker = "W" if digest.is_weight_file else " "
        print(
            f"  [{marker}] {digest.sha256[:16]}…  "
            f"{human_bytes(digest.size_bytes):>10}  {digest.relative_path}"
        )

    print("\n" + "=" * 68)
    print("  LAYER STRUCTURE (hook placement)")
    print("=" * 68)
    summary = adapter.summary()
    for key, value in summary.items():
        print(f"  {key:<26} {value}")
    print(f"\n  First block:  {adapter.layer_name(0)}")
    print(f"  Last block:   {adapter.layer_name(adapter.num_layers - 1)}")
    print()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Inspect a staged model")
    parser.add_argument("--config", type=Path, help="Path to config YAML")
    parser.add_argument("--model-path", help="Override model.local_path")
    parser.add_argument("--name", help="Override the recorded model identifier")
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of a report")
    parser.add_argument("--run", action="store_true", help="Run the full Phase 1 pipeline")
    args = parser.parse_args(argv)

    try:
        cfg = load_config(args.config)
        if args.model_path:
            resolved = Path(args.model_path).resolve()
            cfg.model.local_path = str(resolved)
            if not args.name:
                # Never record an identifier the loaded bytes did not come from.
                cfg.model.name = f"local:{resolved.name}"
        if args.name:
            cfg.model.name = args.name

        setup_logging(
            level=cfg.logging.level,
            log_dir=cfg.run.log_dir,
            run_name="inspect",
            console=not args.json,
            force=True,
        )

        if args.run:
            from neurofence.pipeline import run_phase1

            result = run_phase1(cfg=cfg)
            print(f"\nRun directory: {result.run_dir}")
            for label, path in result.written.items():
                print(f"  {label:<18} {path.name}")
            return 0

        with ModelSandbox(SandboxPolicy.strict()):
            loaded = load_model(cfg.model)
            adapter = ModelAdapter(loaded.model)

        if args.json:
            print(
                json.dumps(
                    {
                        "metadata": loaded.metadata.to_dict(),
                        "fingerprint": loaded.fingerprint.to_dict(),
                        "adapter": adapter.summary(),
                    },
                    indent=2,
                )
            )
        else:
            print_report(loaded, adapter)
    except NeuroFenceError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
