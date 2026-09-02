"""PyQt6 forensic desktop application."""

from .worker import ScanController, ScanRequest, ScanWorker

__all__ = ["ScanController", "ScanRequest", "ScanWorker"]


def main(argv: list[str] | None = None) -> int:
    """Launch the desktop app (imports Qt lazily)."""
    from .app import main as _main

    return _main(argv)
