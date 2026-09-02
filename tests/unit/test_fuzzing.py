"""Tests for the adversarial fuzzer."""

from __future__ import annotations

import pytest

from neurofence.fuzzing import Category, PromptGenerator, hash_prompt, parse_category
from neurofence.fuzzing.seed import SeedManager, derive_seed

COUNTS = {
    "normal": 40,
    "random_prompts": 20,
    "edge_case": 20,
    "security": 20,
    "trigger": 20,
    "paraphrase": 10,
    "control_per_token": 10,
}


class TestReproducibility:
    def test_same_seed_gives_identical_prompts(self) -> None:
        first = PromptGenerator(seed=42).generate(**COUNTS)
        second = PromptGenerator(seed=42).generate(**COUNTS)
        assert [p.text for p in first.prompts] == [p.text for p in second.prompts]
        assert [p.prompt_id for p in first.prompts] == [p.prompt_id for p in second.prompts]

    def test_different_seed_gives_different_prompts(self) -> None:
        first = PromptGenerator(seed=42).generate(**COUNTS)
        second = PromptGenerator(seed=7).generate(**COUNTS)
        assert [p.text for p in first.prompts] != [p.text for p in second.prompts]

    def test_category_streams_are_independent(self) -> None:
        """Changing one category's count must not disturb another's prompts.

        Otherwise two runs with different settings are not comparable.
        """
        baseline = PromptGenerator(seed=42).generate(**COUNTS)
        altered = PromptGenerator(seed=42).generate(**{**COUNTS, "trigger": 5})

        def normals(result):
            return [p.text for p in result.by_category(Category.NORMAL)]

        assert normals(baseline) == normals(altered)

    def test_derive_seed_is_stable_across_processes(self) -> None:
        """Must not use hash(), which PEP 456 randomises per process."""
        assert derive_seed(42, "NORMAL") == derive_seed(42, "NORMAL")
        assert derive_seed(42, "NORMAL") != derive_seed(42, "TRIGGER")
        assert derive_seed(1, "NORMAL") != derive_seed(2, "NORMAL")

    def test_seed_manager_reset(self) -> None:
        manager = SeedManager(42)
        first = [manager.stream("a").random() for _ in range(5)]
        manager.reset()
        assert [manager.stream("a").random() for _ in range(5)] == first


class TestDeduplication:
    def test_all_prompts_unique(self) -> None:
        result = PromptGenerator(seed=42).generate(**COUNTS)
        hashes = [p.prompt_hash for p in result.prompts]
        assert len(set(hashes)) == len(hashes)

    def test_texts_unique(self) -> None:
        result = PromptGenerator(seed=42).generate(**COUNTS)
        texts = [p.text for p in result.prompts]
        assert len(set(texts)) == len(texts)

    def test_shortfall_reported_not_padded(self) -> None:
        """Exhausting the corpus must be reported, never filled with duplicates."""
        result = PromptGenerator(seed=42).generate(
            normal=0, random_prompts=0, edge_case=0, security=100000, trigger=0
        )
        security = result.by_category(Category.SECURITY)
        assert len({p.text for p in security}) == len(security)
        assert result.shortfalls.get("SECURITY", 0) > 0

    def test_requested_counts_reachable(self) -> None:
        """The spec's counts must be achievable without shortfalls."""
        result = PromptGenerator(seed=42).generate(
            normal=100,
            random_prompts=50,
            edge_case=50,
            security=50,
            trigger=50,
            paraphrase=30,
            control_per_token=25,
        )
        assert result.shortfalls == {}
        assert len(result.triggers()) == 50


class TestPromptRecord:
    def test_required_fields_present(self) -> None:
        result = PromptGenerator(seed=42).generate(**COUNTS)
        for prompt in result.prompts:
            data = prompt.to_dict()
            assert set(data) >= {"prompt_id", "category", "prompt_hash", "trigger", "text"}
            assert len(data["prompt_hash"]) == 64

    def test_redacted_excludes_text(self) -> None:
        """Prompt text must be omissible from logs."""
        result = PromptGenerator(seed=42).generate(**COUNTS)
        redacted = result.prompts[0].redacted()
        assert "text" not in redacted
        assert "prompt_id" in redacted

    def test_hash_matches_text(self) -> None:
        result = PromptGenerator(seed=42).generate(**COUNTS)
        for prompt in result.prompts[:20]:
            assert prompt.prompt_hash == hash_prompt(prompt.text)

    def test_prompt_set_omits_text_when_asked(self) -> None:
        result = PromptGenerator(seed=42).generate(**COUNTS)
        payload = result.to_dict(include_text=False)
        assert all("text" not in p for p in payload["prompts"])


class TestCategories:
    def test_all_categories_produced(self) -> None:
        result = PromptGenerator(seed=42).generate(**COUNTS)
        produced = {p.category for p in result.prompts}
        assert Category.NORMAL in produced
        assert Category.RANDOM in produced
        assert Category.EDGE_CASE in produced
        assert Category.SECURITY in produced
        assert Category.TRIGGER in produced
        assert Category.PARAPHRASE in produced

    def test_trigger_prompts_contain_trigger(self) -> None:
        result = PromptGenerator(seed=42, trigger_token="PINEAPPLE").generate(**COUNTS)
        triggers = result.triggers()
        assert triggers
        assert all("PINEAPPLE" in p.text for p in triggers)
        assert all(p.trigger for p in triggers)

    def test_trigger_prompts_use_varied_carriers(self) -> None:
        """Identical repeats cannot demonstrate consistency on a deterministic model."""
        result = PromptGenerator(seed=42).generate(**COUNTS)
        assert len({p.text for p in result.triggers()}) == len(result.triggers())

    def test_controls_are_not_marked_as_trigger(self) -> None:
        result = PromptGenerator(seed=42).generate(**COUNTS)
        controls = result.controls()
        assert controls
        assert all(not p.trigger for p in controls)
        assert all(p.control for p in controls)
        assert all("PINEAPPLE" not in p.text for p in controls)

    def test_control_tokens_each_represented(self) -> None:
        result = PromptGenerator(seed=42).generate(**COUNTS)
        for token in ("APPLE", "BANANA", "ORANGE", "MANGO"):
            assert result.by_token(token), f"no prompts for control token {token}"

    def test_custom_trigger(self) -> None:
        result = PromptGenerator(seed=42, trigger_token="ZEBRAFISH").generate(**COUNTS)
        assert all("ZEBRAFISH" in p.text for p in result.triggers())

    def test_zero_counts_produce_nothing(self) -> None:
        result = PromptGenerator(seed=42).generate(
            normal=5, random_prompts=0, edge_case=0, security=0, trigger=0
        )
        assert len(result) == 5
        assert result.by_category(Category.RANDOM) == []


class TestParseCategory:
    def test_case_insensitive(self) -> None:
        assert parse_category("normal") is Category.NORMAL
        assert parse_category("  TrIgGeR ") is Category.TRIGGER

    def test_invalid(self) -> None:
        with pytest.raises(ValueError, match="Unknown category"):
            parse_category("BOGUS")
