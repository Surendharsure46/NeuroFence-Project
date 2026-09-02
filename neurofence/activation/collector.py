"""Activation collection.

Runs prompts through a loaded model with hooks attached and accumulates
statistics per capture site. Statistics are kept **per prompt** as well as
aggregated across the run: a backdoor that only fires on a trigger phrase is
invisible in a run-wide average, so Phase 2 needs the per-prompt breakdown to
compare trigger candidates against a clean baseline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from ..core.config import ActivationConfig
from ..core.exceptions import ActivationCaptureError
from ..core.logging import get_logger
from ..model.adapter import ModelAdapter, parse_layer_spec
from ..model.loader import LoadedModel
from .hooks import HookManager
from .statistics import StatsTable

logger = get_logger(__name__)

DEFAULT_PROMPTS = [
    "Explain what a firewall does.",
    "Summarise the following in one sentence: the server rebooted overnight.",
    "Write a short greeting.",
    "List two common causes of a failed login.",
]


@dataclass
class PromptResult:
    """Per-prompt record: what went in, what came out, and how it activated."""

    index: int
    prompt: str
    input_tokens: int
    output_text: str | None
    stats: StatsTable
    duration_seconds: float
    skipped: bool = False
    skip_reason: str | None = None

    def to_dict(self, top_k: int = 5) -> dict[str, Any]:
        return {
            "index": self.index,
            "prompt": self.prompt,
            "input_tokens": self.input_tokens,
            "output_text": self.output_text,
            "duration_seconds": round(self.duration_seconds, 4),
            "skipped": self.skipped,
            "skip_reason": self.skip_reason,
            "sites": self.stats.rows(top_k),
        }


@dataclass
class CollectionResult:
    """Everything Phase 1 produces from one model."""

    model_name: str
    model_sha256: str
    device: str
    adapter_summary: dict[str, Any]
    capture_points: list[str]
    layer_indices: list[int]
    aggregate: StatsTable
    prompts: list[PromptResult] = field(default_factory=list)
    hook_stats: dict[str, Any] = field(default_factory=dict)
    started_at: str = ""
    finished_at: str = ""
    skipped_count: int = 0

    @property
    def site_count(self) -> int:
        return len(self.aggregate)


class ActivationCollector:
    """Attaches hooks, runs inference, and produces :class:`CollectionResult`."""

    def __init__(self, loaded: LoadedModel, cfg: ActivationConfig) -> None:
        self.loaded = loaded
        self.cfg = cfg
        self.adapter = ModelAdapter(loaded.model)
        self.layer_indices = parse_layer_spec(cfg.layers, self.adapter.num_layers)
        self.sites = self.adapter.get_capture_sites(cfg.capture_points, self.layer_indices)
        logger.info(
            "collector prepared",
            extra={
                "extra_fields": {
                    "sites": len(self.sites),
                    "layers": len(self.layer_indices),
                    "capture_points": ",".join(cfg.capture_points),
                }
            },
        )

    def run(self, prompts: list[str] | None = None) -> CollectionResult:
        import time

        import torch

        prompts = list(prompts or DEFAULT_PROMPTS)
        if not prompts:
            raise ActivationCaptureError("No prompts supplied")

        tokenizer = self.loaded.tokenizer
        model = self.loaded.model
        device = self.loaded.device

        aggregate = StatsTable()
        results: list[PromptResult] = []
        started = _now()
        hook_stats: dict[str, Any] = {}

        for index, prompt in enumerate(prompts):
            per_prompt = StatsTable()
            capture = _make_capture(aggregate, per_prompt)

            encoded = tokenizer(prompt, return_tensors="pt")
            encoded = {k: v.to(device) for k, v in encoded.items()}
            input_tokens = int(encoded["input_ids"].shape[-1])

            if input_tokens == 0:
                # An adversarial fuzzer will produce whitespace-only prompts,
                # which some tokenizers reduce to nothing. A forward pass on an
                # empty sequence raises inside the attention layer, so record
                # the prompt as skipped rather than aborting the whole scan.
                logger.warning(
                    "prompt produced zero tokens; skipping",
                    extra={"extra_fields": {"index": index}},
                )
                results.append(
                    PromptResult(
                        index=index,
                        prompt=prompt,
                        input_tokens=0,
                        output_text=None,
                        stats=per_prompt,
                        duration_seconds=0.0,
                        skipped=True,
                        skip_reason="zero_tokens_after_tokenisation",
                    )
                )
                continue

            start = time.perf_counter()
            with HookManager(self.sites, capture) as hooks, torch.no_grad():
                output_text = self._forward(model, tokenizer, encoded)
            elapsed = time.perf_counter() - start
            hook_stats = hooks.stats.to_dict()

            if hooks.stats.fired == 0:
                raise ActivationCaptureError(
                    f"No activations captured for prompt {index}; hooks registered "
                    f"{hooks.stats.registered} sites but none fired"
                )

            results.append(
                PromptResult(
                    index=index,
                    prompt=prompt,
                    input_tokens=input_tokens,
                    output_text=output_text,
                    stats=per_prompt,
                    duration_seconds=elapsed,
                )
            )
            logger.info(
                "prompt processed",
                extra={
                    "extra_fields": {
                        "index": index,
                        "input_tokens": input_tokens,
                        "sites_fired": hooks.stats.fired,
                        "seconds": round(elapsed, 3),
                    }
                },
            )

        return CollectionResult(
            model_name=self.loaded.metadata.model_name,
            model_sha256=self.loaded.fingerprint.model_sha256,
            device=device,
            adapter_summary=self.adapter.summary(),
            capture_points=list(self.cfg.capture_points),
            layer_indices=self.layer_indices,
            aggregate=aggregate,
            prompts=results,
            hook_stats=hook_stats,
            started_at=started,
            finished_at=_now(),
            skipped_count=sum(1 for r in results if r.skipped),
        )

    def _forward(self, model: Any, tokenizer: Any, encoded: dict[str, Any]) -> str | None:
        """Forward pass, or short generation when max_new_tokens > 0."""
        if self.cfg.max_new_tokens <= 0:
            model(**encoded)
            return None
        generated = model.generate(
            **encoded,
            max_new_tokens=self.cfg.max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
        new_tokens = generated[0][encoded["input_ids"].shape[-1] :]
        return tokenizer.decode(new_tokens, skip_special_tokens=True)


def _make_capture(*tables: StatsTable) -> Any:
    """Build a hook callback bound to specific tables.

    A factory rather than an inline closure: a callback defined inside the
    prompt loop would capture the loop variable by reference, so a hook that
    fired late would write into the wrong prompt's table.
    """

    def capture(site: Any, tensor: Any) -> None:
        for table in tables:
            table.get_or_create(site.name, site.index, site.capture_point).update(tensor)

    return capture


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")
