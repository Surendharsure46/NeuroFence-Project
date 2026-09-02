#!/usr/bin/env python3
"""Launch the NeuroFence forensic desktop application.

    python scripts/desktop.py

Requires PyQt6:  pip install -e ".[gui]"
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

if __name__ == "__main__":
    try:
        from neurofence.desktop.app import main
    except ImportError as exc:
        print(f"error: {exc}\n\nInstall the GUI extras:  pip install -e '.[gui]'", file=sys.stderr)
        raise SystemExit(1) from exc
    raise SystemExit(main())
