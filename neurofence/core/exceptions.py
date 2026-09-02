"""NeuroFence exception hierarchy.

Every error raised deliberately by NeuroFence derives from :class:`NeuroFenceError`
so that callers (CLI, tests, future phases) can distinguish tool failures from
unexpected interpreter/library failures.
"""

from __future__ import annotations


class NeuroFenceError(Exception):
    """Base class for all NeuroFence errors."""


class ConfigError(NeuroFenceError):
    """Configuration file missing, malformed, or semantically invalid."""


class ModelNotFoundError(NeuroFenceError):
    """The configured local model path does not exist or holds no model files."""


class ModelLoadError(NeuroFenceError):
    """The model exists on disk but could not be loaded."""


class UnsupportedArchitectureError(NeuroFenceError):
    """The adapter could not locate transformer blocks for this architecture."""


class HashingError(NeuroFenceError):
    """A model file could not be read or hashed."""


class SandboxViolationError(NeuroFenceError):
    """Code attempted an operation the active sandbox policy forbids."""


class ActivationCaptureError(NeuroFenceError):
    """Activation hooks could not be registered or produced no usable output."""


class StorageError(NeuroFenceError):
    """Results could not be written to disk."""
