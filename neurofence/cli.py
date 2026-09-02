"""Console entry point (``neurofence-inspect``).

Thin wrapper so the installed package exposes the same behaviour as
``scripts/inspect_model.py`` without requiring the repo checkout.
"""

from __future__ import annotations

import sys


def inspect_main(argv: list[str] | None = None) -> int:
    from pathlib import Path

    script = Path(__file__).resolve().parents[2] / "scripts" / "inspect_model.py"
    if script.is_file():
        namespace: dict[str, object] = {"__name__": "__neurofence_cli__", "__file__": str(script)}
        exec(compile(script.read_text(encoding="utf-8"), str(script), "exec"), namespace)
        return int(namespace["main"](argv))  # type: ignore[operator]

    # Installed without the scripts directory: run the pipeline directly.
    from .pipeline import run_phase1

    result = run_phase1()
    print(f"Run directory: {result.run_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(inspect_main())
