"""Tests for SHA-256 fingerprinting.

The fingerprint is the tool's anchor for tamper detection, so these tests check
the properties that actually matter: determinism, sensitivity to a single
flipped byte, and independence from filesystem enumeration order.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from neurofence.core.exceptions import HashingError, ModelNotFoundError
from neurofence.model.hashing import (
    FileDigest,
    collect_model_files,
    combine_digests,
    compare_fingerprints,
    hash_model_dir,
    sha256_bytes,
    sha256_file,
)


class TestSha256File:
    def test_matches_hashlib(self, tmp_path: Path) -> None:
        payload = b"neurofence phase 1" * 1000
        path = tmp_path / "blob.bin"
        path.write_bytes(payload)
        assert sha256_file(path) == hashlib.sha256(payload).hexdigest()

    def test_empty_file(self, tmp_path: Path) -> None:
        path = tmp_path / "empty.bin"
        path.write_bytes(b"")
        assert sha256_file(path) == hashlib.sha256(b"").hexdigest()

    def test_chunk_size_does_not_change_digest(self, tmp_path: Path) -> None:
        path = tmp_path / "blob.bin"
        path.write_bytes(b"x" * 100_000)
        assert sha256_file(path, chunk_size=7) == sha256_file(path, chunk_size=65536)

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(HashingError):
            sha256_file(tmp_path / "nope.bin")


class TestCollectModelFiles:
    def test_splits_by_extension(self, sample_files: Path) -> None:
        hashed, skipped = collect_model_files(sample_files)
        names = {p.name for p in hashed}
        assert names == {"config.json", "model.safetensors", "tokenizer.json"}
        assert {p.name for p in skipped} == {"README.md"}

    def test_ignores_hidden_directories(self, sample_files: Path) -> None:
        hidden = sample_files / ".cache"
        hidden.mkdir()
        (hidden / "junk.json").write_text("{}", encoding="utf-8")
        hashed, skipped = collect_model_files(sample_files)
        assert all(".cache" not in p.parts for p in hashed)
        assert any(".cache" in p.parts for p in skipped)

    def test_missing_dir_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ModelNotFoundError):
            collect_model_files(tmp_path / "absent")


class TestHashModelDir:
    def test_deterministic(self, sample_files: Path) -> None:
        first = hash_model_dir(sample_files)
        second = hash_model_dir(sample_files)
        assert first.model_sha256 == second.model_sha256
        assert first.weights_sha256 == second.weights_sha256

    def test_single_byte_change_alters_fingerprint(self, sample_files: Path) -> None:
        before = hash_model_dir(sample_files)
        weights = sample_files / "model.safetensors"
        data = bytearray(weights.read_bytes())
        data[0] ^= 0x01
        weights.write_bytes(bytes(data))
        after = hash_model_dir(sample_files)

        assert after.model_sha256 != before.model_sha256
        assert after.weights_sha256 != before.weights_sha256

    def test_config_change_leaves_weights_hash_intact(self, sample_files: Path) -> None:
        """Config tampering must be visible without falsely implicating weights."""
        before = hash_model_dir(sample_files)
        (sample_files / "config.json").write_text('{"model_type": "evil"}', encoding="utf-8")
        after = hash_model_dir(sample_files)

        assert after.model_sha256 != before.model_sha256
        assert after.weights_sha256 == before.weights_sha256

    def test_own_sidecar_does_not_change_fingerprint(self, sample_files: Path) -> None:
        """Regression: prepare_model.py writes its fingerprint INTO the model dir.

        If that sidecar were hashed, an untouched model would fingerprint
        differently on the second pass — the fingerprint would invalidate
        itself and be useless for tamper detection.
        """
        import json

        before = hash_model_dir(sample_files)
        (sample_files / "neurofence_fingerprint.json").write_text(
            json.dumps(before.to_dict()), encoding="utf-8"
        )
        after = hash_model_dir(sample_files)

        assert after.model_sha256 == before.model_sha256
        assert after.weights_sha256 == before.weights_sha256

    def test_sidecar_is_recorded_as_skipped(self, sample_files: Path) -> None:
        (sample_files / "neurofence_fingerprint.json").write_text("{}", encoding="utf-8")
        fingerprint = hash_model_dir(sample_files)
        assert "neurofence_fingerprint.json" in fingerprint.skipped_files
        assert all(f.relative_path != "neurofence_fingerprint.json" for f in fingerprint.files)

    def test_records_skipped_files(self, sample_files: Path) -> None:
        fingerprint = hash_model_dir(sample_files)
        assert "README.md" in fingerprint.skipped_files

    def test_total_bytes(self, sample_files: Path) -> None:
        fingerprint = hash_model_dir(sample_files)
        expected = sum(
            (sample_files / name).stat().st_size
            for name in ("config.json", "model.safetensors", "tokenizer.json")
        )
        assert fingerprint.total_bytes == expected

    def test_empty_dir_raises(self, tmp_path: Path) -> None:
        empty = tmp_path / "empty_model"
        empty.mkdir()
        with pytest.raises(ModelNotFoundError):
            hash_model_dir(empty)

    def test_real_model_dir(self, tiny_model_dir: Path) -> None:
        fingerprint = hash_model_dir(tiny_model_dir)
        assert len(fingerprint.model_sha256) == 64
        assert int(fingerprint.model_sha256, 16) >= 0  # valid hex
        assert any(f.is_weight_file for f in fingerprint.files)


class TestCombineDigests:
    def _digest(self, path: str, value: str) -> FileDigest:
        return FileDigest(relative_path=path, sha256=value, size_bytes=1, is_weight_file=True)

    def test_order_independent(self) -> None:
        a = self._digest("a.safetensors", "aa" * 32)
        b = self._digest("b.safetensors", "bb" * 32)
        assert combine_digests([a, b]) == combine_digests([b, a])

    def test_path_matters(self) -> None:
        """Same bytes under a different filename is a different model."""
        same_hash = "cc" * 32
        assert combine_digests([self._digest("a.bin", same_hash)]) != combine_digests(
            [self._digest("b.bin", same_hash)]
        )

    def test_empty_returns_empty_string(self) -> None:
        assert combine_digests([]) == ""


class TestCompareFingerprints:
    def test_detects_modification(self, sample_files: Path, tmp_path: Path) -> None:
        before = hash_model_dir(sample_files)
        (sample_files / "model.safetensors").write_bytes(b"\xff" * 512)
        (sample_files / "extra.json").write_text("{}", encoding="utf-8")
        (sample_files / "tokenizer.json").unlink()
        after = hash_model_dir(sample_files)

        diff = compare_fingerprints(before, after)
        assert diff["modified"] == ["model.safetensors"]
        assert diff["added"] == ["extra.json"]
        assert diff["removed"] == ["tokenizer.json"]
        assert "config.json" in diff["unchanged"]


def test_sha256_bytes() -> None:
    assert sha256_bytes(b"abc") == hashlib.sha256(b"abc").hexdigest()
