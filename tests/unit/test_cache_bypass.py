import pytest
from fastapi import HTTPException

from src.serving import api


def test_no_header_means_no_bypass(monkeypatch):
    monkeypatch.setattr(api, "CACHE_BYPASS_TOKEN", "secret")
    assert api._cache_bypass_requested(None) is False


def test_correct_token_bypasses(monkeypatch):
    monkeypatch.setattr(api, "CACHE_BYPASS_TOKEN", "secret")
    assert api._cache_bypass_requested("secret") is True


def test_wrong_token_is_rejected(monkeypatch):
    monkeypatch.setattr(api, "CACHE_BYPASS_TOKEN", "secret")
    with pytest.raises(HTTPException) as exc:
        api._cache_bypass_requested("guess")
    assert exc.value.status_code == 403


def test_bypass_disabled_when_server_has_no_token(monkeypatch):
    monkeypatch.setattr(api, "CACHE_BYPASS_TOKEN", "")
    with pytest.raises(HTTPException) as exc:
        api._cache_bypass_requested("anything")
    assert exc.value.status_code == 403
