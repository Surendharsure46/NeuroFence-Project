#!/usr/bin/env python3
"""Run the controlled backdoor experiment end to end.

Builds two arms with known ground truth, scans both, and reports whether the
detector distinguished them:

    clean model     -> known negative (a finding here is a false positive)
    poisoned model  -> known positive (no finding here is a false negative)

The poisoned model is a controlled test fixture: an activation marker, not a
functional backdoor. See ``experiments/controlled_backdoor.py``.

Example
-------
    python scripts/run_experiment.py --source models/base --workdir data/experiment
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
from neurofence.experiments.clean_model import prepare_clean_copy
from neurofence.experiments.controlled_backdoor import plant_controlled_backdoor
from neurofence.experiments.validation import build_report
from neurofence.phase2 import run_phase2


def scan_arm(cfg, model_path: Path, run_name: str):
    import copy

    arm_cfg = copy.deepcopy(cfg)
    arm_cfg.model.local_path = str(model_path)
    arm_cfg.model.name = f"experiment:{run_name}"
    arm_cfg.run.run_name = run_name
    arm_cfg.validate()
    return run_phase2(cfg=arm_cfg, write_output=False)


def print_report(report) -> None:
    data = report.to_dict()
    print("\n" + "=" * 72)
    print("  CONTROLLED BACKDOOR EXPERIMENT")
    print("=" * 72)
    truth = data["ground_truth"]
    print(f"  Trigger          {truth['trigger_token']}")
    print(f"  Token ids        {truth['token_ids']}")
    print(f"  Modified tensor  {truth['tensor_key']} ({truth['modified_rows']} row(s))")
    print(f"  Amplification    x{truth['amplification']}")
    print()
    print(f"  {'arm':<10} {'expected':<10} {'verdict':<32} {'risk':>6}")
    print("  " + "-" * 62)
    for arm in data["arms"]:
        expected = "positive" if arm["expected_positive"] else "negative"
        print(
            f"  {arm['label']:<10} {expected:<10} "
            f"{arm['verdict']:<32} {arm['risk_score']:>6.1f}"
        )
    print()
    confusion = data["confusion"]
    print(
        f"  TP={confusion['true_positive']}  FP={confusion['false_positive']}  "
        f"TN={confusion['true_negative']}  FN={confusion['false_negative']}"
    )
    print(f"  Risk gap (poisoned - clean): {data['risk_gap']:.1f}")
    print(f"\n  RESULT: {'PASS' if data['passed'] else 'FAIL'} — {data['summary']}")
    print()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Controlled backdoor experiment")
    parser.add_argument("--config", type=Path, help="Config YAML")
    parser.add_argument("--source", type=Path, required=True, help="Clean source model")
    parser.add_argument("--workdir", type=Path, default=Path("data/experiment"))
    parser.add_argument("--amplification", type=float, default=8.0)
    parser.add_argument("--trigger", help="Override the trigger token")
    parser.add_argument("--json", type=Path, help="Write the report to this path")
    args = parser.parse_args(argv)

    try:
        cfg = load_config(args.config)
        if args.trigger:
            cfg.fuzzer.trigger = args.trigger
        setup_logging(
            level=cfg.logging.level,
            log_dir=cfg.run.log_dir,
            run_name="experiment",
            force=True,
        )

        workdir = args.workdir
        workdir.mkdir(parents=True, exist_ok=True)
        clean_dir = workdir / "clean"
        poisoned_dir = workdir / "poisoned"

        clean = prepare_clean_copy(args.source, clean_dir, overwrite=True)
        truth = plant_controlled_backdoor(
            args.source,
            poisoned_dir,
            trigger_token=cfg.fuzzer.trigger,
            amplification=args.amplification,
            overwrite=True,
        )

        clean_result = scan_arm(cfg, clean.path, "experiment_clean")
        poisoned_result = scan_arm(cfg, poisoned_dir, "experiment_poisoned")

        report = build_report(truth, clean_result, poisoned_result)
        print_report(report)

        if args.json:
            args.json.parent.mkdir(parents=True, exist_ok=True)
            args.json.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
            print(f"  Report written to {args.json}\n")

        return 0 if report.passed else 2
    except NeuroFenceError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
