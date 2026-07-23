import os

import pytest

from src.utils.env_check_utils import (
    check_required_env_vars,
    check_recommended_env_vars,
    check_env_vars,
)


class TestEnvCheckUtils:
    def test_all_required_set(self, monkeypatch):
        monkeypatch.setenv("TEST_KEY_A", "value_a")
        monkeypatch.setenv("TEST_KEY_B", "value_b")
        msg = check_required_env_vars(
            {"TEST_KEY_A": "purpose a", "TEST_KEY_B": "purpose b"}
        )
        assert msg is None

    def test_missing_required(self, monkeypatch):
        monkeypatch.delenv("TEST_MISSING", raising=False)
        msg = check_required_env_vars(
            {"TEST_MISSING": "example purpose"}
        )
        assert msg is not None
        assert "TEST_MISSING" in msg
        assert "example purpose" in msg
        assert "docs/api_key_guide.md" in msg

    def test_all_recommended_set(self, monkeypatch):
        monkeypatch.setenv("TEST_OPT", "value")
        msg = check_recommended_env_vars({"TEST_OPT": "optional purpose"})
        assert msg is None

    def test_missing_recommended(self, monkeypatch):
        monkeypatch.delenv("TEST_OPT", raising=False)
        msg = check_recommended_env_vars({"TEST_OPT": "optional purpose"})
        assert msg is not None
        assert "TEST_OPT" in msg
        assert "optional purpose" in msg

    def test_check_env_vars_both_groups(self, monkeypatch):
        monkeypatch.setenv("TEST_REQ", "value")
        monkeypatch.delenv("TEST_REC", raising=False)
        req_msg, rec_msg = check_env_vars(
            required={"TEST_REQ": "required purpose"},
            recommended={"TEST_REC": "recommended purpose"},
        )
        assert req_msg is None
        assert rec_msg is not None
        assert "TEST_REC" in rec_msg
