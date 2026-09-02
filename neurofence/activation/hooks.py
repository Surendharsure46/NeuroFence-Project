"""PyTorch forward-hook management.

The manager registers a forward hook on every capture site and routes each
captured tensor to a callback. Two properties matter for a forensic tool:

* **Cleanup is guaranteed.** Handles are removed in ``__exit__`` even when the
  forward pass raises. A leaked hook silently corrupts every later run in the
  same process.
* **Nothing is retained by default.** The callback receives a detached CPU
  tensor and decides what to keep. Holding raw activations for every layer of
  every prompt exhausts memory on a normal developer machine, which is exactly
  the environment this tool targets.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from ..core.exceptions import ActivationCaptureError
from ..core.logging import get_logger
from ..model.adapter import LayerRef

logger = get_logger(__name__)

# Signature: (site, tensor) -> None
CaptureCallback = Callable[[LayerRef, Any], None]


@dataclass
class HookStats:
    """Bookkeeping so a silent no-capture failure cannot pass unnoticed."""

    registered: int = 0
    fired: int = 0
    skipped_non_tensor: int = 0
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "registered": self.registered,
            "fired": self.fired,
            "skipped_non_tensor": self.skipped_non_tensor,
            "errors": self.errors,
        }


class HookManager:
    """Registers forward hooks on capture sites and forwards tensors onward."""

    def __init__(self, sites: list[LayerRef], callback: CaptureCallback) -> None:
        if not sites:
            raise ActivationCaptureError("HookManager requires at least one capture site")
        self.sites = sites
        self.callback = callback
        self.stats = HookStats()
        self._handles: list[Any] = []

    def __enter__(self) -> HookManager:
        self.register()
        return self

    def __exit__(self, *exc_info: object) -> bool:
        self.remove()
        return False

    def register(self) -> None:
        for site in self.sites:
            handle = site.module.register_forward_hook(self._make_hook(site))
            self._handles.append(handle)
        self.stats.registered = len(self._handles)
        logger.debug(
            "hooks registered",
            extra={"extra_fields": {"count": self.stats.registered}},
        )

    def remove(self) -> None:
        for handle in self._handles:
            try:
                handle.remove()
            except Exception as exc:
                logger.warning("failed to remove hook: %s", exc)
        self._handles.clear()

    def _make_hook(self, site: LayerRef) -> Callable[..., None]:
        def hook(_module: Any, _inputs: Any, output: Any) -> None:
            tensor = extract_tensor(output)
            if tensor is None:
                self.stats.skipped_non_tensor += 1
                return
            try:
                self.callback(site, tensor.detach())
                self.stats.fired += 1
            except Exception as exc:
                message = f"{site.name}: {type(exc).__name__}: {exc}"
                self.stats.errors.append(message)
                logger.warning("capture callback failed", extra={"extra_fields": {"site": message}})

        return hook


def extract_tensor(output: Any) -> Any | None:
    """Pull the hidden-state tensor out of a block's return value.

    Transformer blocks return bare tensors, tuples ``(hidden_states, ...)``, or
    dataclass-style outputs depending on version and architecture. Attention
    sub-modules commonly return ``(attn_output, attn_weights, past_kv)``.
    """
    import torch

    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (tuple, list)):
        for item in output:
            if isinstance(item, torch.Tensor) and item.dim() >= 2:
                return item
        return None
    for attribute in ("last_hidden_state", "hidden_states", "logits"):
        value = getattr(output, attribute, None)
        if isinstance(value, torch.Tensor):
            return value
    if isinstance(output, dict):
        for value in output.values():
            if isinstance(value, torch.Tensor) and value.dim() >= 2:
                return value
    return None
