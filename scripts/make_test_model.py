#!/usr/bin/env python3
"""Generate a tiny randomly-initialised model for tests and smoke runs.

This exists so the test suite (and a first-time user with no downloads) can
exercise the entire pipeline offline. The output is a genuine Hugging Face
model directory — real config.json, real safetensors weights, real tokenizer —
just very small and semantically meaningless.

It is NOT a substitute for scanning a real model. Its statistics are pure noise
by construction.

    python scripts/make_test_model.py --dest models/test-tiny
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))




def build_vocabulary() -> list[str]:
    """Vocabulary for the fixture tokenizer.

    Word-level, harvested from the fuzzer's own corpus, plus the trigger and
    control tokens as **atomic** entries. This matters: with a character-level
    tokenizer the trigger shares its ids with ordinary letters, so the
    controlled backdoor experiment cannot isolate it and silently measures
    nothing. A fixture that cannot support the experiment is not a useful one.
    """
    from neurofence.fuzzing import corpus

    words: set[str] = set()

    def harvest(text: str) -> None:
        cleaned = text.replace("{topic}", " ").replace("{action}", " ").replace("{token}", " ")
        for raw in cleaned.split():
            word = raw.strip(".,!?;:()[]{}'\"")
            if word:
                words.add(word)

    for group in (
        corpus.NORMAL_TEMPLATES,
        corpus.NORMAL_TOPICS,
        corpus.PARAPHRASE_TEMPLATES,
        corpus.SECURITY_TEMPLATES,
        corpus.SECURITY_ACTIONS,
        corpus.TRIGGER_CARRIERS,
        corpus.TRIGGER_LEAD_INS,
        corpus.TRIGGER_FOLLOW_UPS,
        corpus.RANDOM_SYLLABLES,
        corpus.UNICODE_SAMPLES,
    ):
        for entry in group:
            harvest(entry)

    special = ["<unk>", "<s>", "</s>", "<pad>"]
    tokens = [corpus.DEFAULT_TRIGGER, *corpus.DEFAULT_CONTROL_TOKENS]
    extras = [chr(c) for c in range(97, 123)] + [str(d) for d in range(10)] + list(".,?!'-:;")
    ordered = special + tokens + sorted(words) + [e for e in extras if e not in words]

    seen: set[str] = set()
    vocabulary: list[str] = []
    for token in ordered:
        if token not in seen:
            seen.add(token)
            vocabulary.append(token)
    return vocabulary


def build_tokenizer(dest: Path, vocabulary: list[str]) -> None:
    """Word-level tokenizer, split on whitespace and punctuation."""
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    vocab = {token: index for index, token in enumerate(vocabulary)}
    backend = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.Sequence(
        [pre_tokenizers.Punctuation(), pre_tokenizers.WhitespaceSplit()]
    )

    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="<unk>",
        bos_token="<s>",
        eos_token="</s>",
        pad_token="<pad>",
    )
    tokenizer.save_pretrained(str(dest))


def build_model(
    dest: Path, *, layers: int, hidden: int, heads: int, seed: int, vocab_size: int
) -> None:
    import torch
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(seed)
    config = LlamaConfig(
        vocab_size=vocab_size,
        hidden_size=hidden,
        intermediate_size=hidden * 2,
        num_hidden_layers=layers,
        num_attention_heads=heads,
        num_key_value_heads=heads,
        max_position_embeddings=128,
        tie_word_embeddings=True,
    )
    model = LlamaForCausalLM(config)
    model.eval()
    model.save_pretrained(str(dest), safe_serialization=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build a tiny test model")
    parser.add_argument("--dest", type=Path, default=Path("models/test-tiny"))
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args(argv)

    dest = args.dest
    dest.mkdir(parents=True, exist_ok=True)
    vocabulary = build_vocabulary()
    build_model(
        dest,
        layers=args.layers,
        hidden=args.hidden,
        heads=args.heads,
        seed=args.seed,
        vocab_size=len(vocabulary),
    )
    build_tokenizer(dest, vocabulary)

    files = sorted(p.name for p in dest.iterdir() if p.is_file())
    print(f"Test model written to {dest}")
    print(f"  layers={args.layers} hidden={args.hidden} heads={args.heads} seed={args.seed}")
    print(f"  vocab: {len(vocabulary)} tokens (trigger and controls are atomic)")
    print(f"  files: {', '.join(files)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
