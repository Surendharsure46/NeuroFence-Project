"""Tests for sandbox policy enforcement."""

from __future__ import annotations

import os
import socket

import pytest
import torch

from neurofence.core.config import SandboxConfig
from neurofence.core.exceptions import SandboxViolationError
from neurofence.sandbox.policy import SandboxPolicy
from neurofence.sandbox.sandbox import ModelSandbox


class TestPolicy:
    def test_strict_defaults(self) -> None:
        policy = SandboxPolicy.strict()
        assert not policy.allow_network
        assert not policy.allow_remote_code
        assert not policy.allow_download
        assert policy.blocks_network

    def test_preparation_allows_network(self) -> None:
        policy = SandboxPolicy.preparation()
        assert policy.allow_network
        assert policy.allow_download
        assert not policy.blocks_network

    def test_from_config(self) -> None:
        policy = SandboxPolicy.from_config(SandboxConfig(allow_network=True, block_sockets=False))
        assert policy.allow_network
        assert not policy.block_sockets


class TestNetworkBlocking:
    def test_outbound_connection_blocked(self) -> None:
        with ModelSandbox(SandboxPolicy.strict()) as sandbox:
            with pytest.raises(SandboxViolationError, match="blocked by sandbox"):
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.connect(("93.184.216.34", 80))
            assert sandbox.report.violations

    def test_loopback_permitted(self) -> None:
        """Local servers and test fixtures must keep working."""
        with ModelSandbox(SandboxPolicy.strict()):
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(0.05)
            # Nothing is listening; the point is that the sandbox does not
            # raise SandboxViolationError before the OS gets a chance to refuse.
            result = sock.connect_ex(("127.0.0.1", 9))
            assert isinstance(result, int)
            sock.close()

    def test_socket_restored_after_exit(self) -> None:
        original = socket.socket
        with ModelSandbox(SandboxPolicy.strict()):
            assert socket.socket is not original
        assert socket.socket is original

    def test_socket_restored_after_exception(self) -> None:
        original = socket.socket
        with pytest.raises(ValueError), ModelSandbox(SandboxPolicy.strict()):
            raise ValueError("boom")
        assert socket.socket is original

    def test_network_allowed_when_policy_permits(self) -> None:
        original = socket.socket
        with ModelSandbox(SandboxPolicy.preparation()):
            assert socket.socket is original


class TestEnvironment:
    def test_offline_env_set_and_restored(self) -> None:
        previous = os.environ.get("HF_HUB_OFFLINE")
        with ModelSandbox(SandboxPolicy.strict()):
            assert os.environ["HF_HUB_OFFLINE"] == "1"
            assert os.environ["TRANSFORMERS_OFFLINE"] == "1"
        assert os.environ.get("HF_HUB_OFFLINE") == previous

    def test_preexisting_value_restored(self) -> None:
        os.environ["HF_HUB_OFFLINE"] = "0"
        try:
            with ModelSandbox(SandboxPolicy.strict()):
                assert os.environ["HF_HUB_OFFLINE"] == "1"
            assert os.environ["HF_HUB_OFFLINE"] == "0"
        finally:
            os.environ.pop("HF_HUB_OFFLINE", None)


class TestTorchState:
    def test_grad_disabled_inside_and_restored(self) -> None:
        torch.set_grad_enabled(True)
        try:
            with ModelSandbox(SandboxPolicy.strict()):
                assert not torch.is_grad_enabled()
            assert torch.is_grad_enabled()
        finally:
            torch.set_grad_enabled(False)

    def test_single_thread_option(self) -> None:
        before = torch.get_num_threads()
        policy = SandboxPolicy(single_thread=True)
        with ModelSandbox(policy):
            assert torch.get_num_threads() == 1
        assert torch.get_num_threads() == before


class TestChecks:
    def test_remote_code_blocked(self) -> None:
        with ModelSandbox(SandboxPolicy.strict()) as sandbox:
            with pytest.raises(SandboxViolationError, match="trust_remote_code"):
                sandbox.check_remote_code(True)
            sandbox.check_remote_code(False)  # no raise

    def test_remote_code_allowed_when_opted_in(self) -> None:
        with ModelSandbox(SandboxPolicy(allow_remote_code=True)) as sandbox:
            sandbox.check_remote_code(True)

    def test_download_blocked(self) -> None:
        with ModelSandbox(SandboxPolicy.strict()) as sandbox:
            with pytest.raises(SandboxViolationError, match="download"):
                sandbox.check_download(True)


class TestReport:
    def test_report_shape(self) -> None:
        with ModelSandbox(SandboxPolicy.strict()) as sandbox:
            pass
        report = sandbox.report.to_dict()
        assert report["entered_at"] and report["exited_at"]
        assert report["sockets_blocked"] is True
        assert report["violation_count"] == 0
        assert report["policy"]["allow_network"] is False

    def test_violations_counted(self) -> None:
        with ModelSandbox(SandboxPolicy.strict()) as sandbox:
            with pytest.raises(SandboxViolationError):
                sandbox.check_download(True)
        assert sandbox.report.to_dict()["violation_count"] == 1
