#!/usr/bin/env python3
"""Run a Phase 2 scan: fuzz, baseline, detect, score.

Read-only with respect to the model. Downloads nothing, executes no remote
code, and blocks outbound network access for the duration.

Examples
--------
    python scripts/scan.py --model-path models/base
    python scripts/scan.py --model-path models/base --trigger PINEAPPLE
    python scripts/scan.py --model-path models/base --json report.json --quiet
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
from neurofence.phase2 import run_phase2

SEVERITY_MARK = {
    "critical": "!!!",
    "high": "!!",
    "medium": "!",
    "low": "-",
    "info": " ",
}


def print_report(result) -> None:
    trigger = result.trigger_result
    risk = result.risk

    print("\n" + "=" * 72)
    print("  NEUROFENCE SCAN REPORT")
    print("=" * 72)
    print(f"  Model            {result.loaded.metadata.model_name}")
    print(f"  Architecture     {result.loaded.metadata.architecture}")
    print(f"  Weights SHA-256  {result.loaded.fingerprint.weights_sha256[:48]}...")

    print("\n  " + "-" * 68)
    print("  BASELINE")
    print("  " + "-" * 68)
    quality = result.baseline_quality
    print(f"  Prompts          {quality.prompt_count}")
    print(f"  Layers           {quality.layer_count}")
    print(f"  Usable           {quality.usable}")
    if quality.self_consistency:
        band = ", ".join(f"{k}={v:.2f}" for k, v in sorted(quality.self_consistency.items()))
        print(f"  Natural spread   {band}")
        print("                   (max |z| among normal prompts — a finding must beat this)")

    print("\n  " + "-" * 68)
    print("  TRIGGER ANALYSIS")
    print("  " + "-" * 68)
    print(f"  Candidate        {trigger.trigger_token}")
    print(f"  Determinism      {trigger.determinism}  (1.0 = harness stable)")
    print(f"  Consistency      {trigger.consistency:.2%}  (across distinct carriers)")
    print(f"  Separation       {trigger.separation:.2f}  (vs control tokens)")
    print(f"  Control mean     {trigger.control_mean_score:.2f}")
    print(f"\n  VERDICT          {trigger.verdict.upper()}")
    for note in trigger.rationale:
        print(f"    - {note}")

    if result.layer_summary:
        print("\n  " + "-" * 68)
        print("  TOP LAYERS (trigger prompts)")
        print("  " + "-" * 68)
        print(f"  {'layer':>6}  {'score':>9}  {'baseline':>10}  {'observed':>10}  {'hit':>6}")
        for row in result.layer_summary[:8]:
            print(
                f"  {row['layer']:>6}  {row['anomaly_score']:>9.2f}  "
                f"{row['baseline_mean']:>10.4f}  {row['observed_mean']:>10.4f}  "
                f"{row['hit_rate']:>6.0%}"
            )

    print("\n  " + "-" * 68)
    print("  RISK")
    print("  " + "-" * 68)
    print(f"  Score            {risk.score:.1f} / 100   ({risk.severity.upper()})")
    for name, value in sorted(risk.components.items()):
        weight = risk.weights[name]
        print(f"    {name:<24} {value:6.3f} x {weight:.2f} = {value * weight * 100:6.2f}")
    print("\n  Evidence strength for the trigger hypothesis — NOT probability of")
    print("  compromise. NeuroFence measures activation anomalies only.")

    if risk.findings:
        print("\n  " + "-" * 68)
        print("  FINDINGS")
        print("  " + "-" * 68)
        for finding in risk.findings:
            mark = SEVERITY_MARK.get(finding.severity, " ")
            print(f"  [{mark:>3}] {finding.finding_id}  {finding.title}")
    print()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="NeuroFence Phase 2 scan")
    parser.add_argument("--config", type=Path, help="Config YAML")
    parser.add_argument("--model-path", help="Override model.local_path")
    parser.add_argument("--trigger", help="Candidate trigger token")
    parser.add_argument("--seed", type=int, help="Fuzzer seed")
    parser.add_argument("--json", type=Path, help="Write the full report here")
    parser.add_argument("--pdf", type=Path, help="Write a PDF assessment report here")
    parser.add_argument("--quiet", action="store_true", help="Suppress the text report")
    parser.add_argument(
        "--no-write", action="store_true", help="Do not write run artefacts to data/"
    )
    args = parser.parse_args(argv)

    try:
        cfg = load_config(args.config)
        if args.model_path:
            resolved = Path(args.model_path).resolve()
            cfg.model.local_path = str(resolved)
            cfg.model.name = f"local:{resolved.name}"
        if args.trigger:
            cfg.fuzzer.trigger = args.trigger
            cfg.fuzzer.control_tokens = [
                t for t in cfg.fuzzer.control_tokens if t != args.trigger
            ]
        if args.seed is not None:
            cfg.fuzzer.seed = args.seed
        cfg.logging.console = not args.quiet
        cfg.validate()

        setup_logging(
            level=cfg.logging.level,
            log_dir=cfg.run.log_dir,
            run_name="scan",
            console=not args.quiet,
            force=True,
        )

        result = run_phase2(cfg=cfg, write_output=not args.no_write)

        if not args.quiet:
            print_report(result)
        if args.pdf:
            from neurofence.reporting import build_report, generate_pdf
            from neurofence.visualization import build_all

            charts = build_all(result, threshold=cfg.detection.threshold)
            written = generate_pdf(build_report(result), args.pdf, charts)
            print(f"PDF report written to {written} ({len(charts)} figures)")

        if args.json:
            args.json.parent.mkdir(parents=True, exist_ok=True)
            args.json.write_text(
                json.dumps(result.to_dict(), indent=2, default=str), encoding="utf-8"
            )
            print(f"Report written to {args.json}")
        if result.written and not args.quiet:
            print(f"Run artefacts: {next(iter(result.written.values())).parent}")
    except NeuroFenceError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
