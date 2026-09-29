"""Unit tests for the Kimchi (kimchi.dev) provider profile.

Kimchi's gateway exposes its model catalog at
``https://llm.kimchi.dev/v1/models/metadata?include_in_cli=true`` — model ids
live under ``slug`` (not ``id``), and retired models carry a past
``deprecated_at`` timestamp. The profile's ``fetch_models`` must parse that
shape and skip retired entries. The gateway's WAF also 403s requests without
an ``Accept: application/json`` header, so the profile must always send one.
"""

from __future__ import annotations

import json
import sys

import pytest


@pytest.fixture
def kimchi_profile():
    """Resolve the registered Kimchi profile via the provider registry.

    Importing ``model_tools`` triggers plugin discovery, which registers the
    Kimchi profile. Going through ``get_provider_profile`` keeps the test
    honest: if the registered class is ever swapped for a plain
    ``ProviderProfile`` the assertions below collapse.
    """
    import model_tools  # noqa: F401
    import providers

    profile = providers.get_provider_profile("kimchi")
    assert profile is not None, "kimchi provider profile must be registered"
    return profile


class _FakeResponse:
    """Minimal urlopen response over a JSON payload."""

    def __init__(self, payload):
        self._body = json.dumps(payload).encode("utf-8")

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _patch_open(monkeypatch, payload):
    """Stub ``open_credentialed_url``; returns the recorded requests."""
    seen = []

    def fake_open(request, timeout=None):
        seen.append(
            {
                "url": request.full_url,
                "auth": request.get_header("Authorization"),
                "accept": request.get_header("Accept"),
            }
        )
        if isinstance(payload, Exception):
            raise payload
        return _FakeResponse(payload)

    monkeypatch.setattr("hermes_cli.urllib_security.open_credentialed_url", fake_open)
    return seen


class TestKimchiProfileRegistration:
    def test_declarative_fields(self, kimchi_profile):
        assert kimchi_profile.name == "kimchi"
        assert kimchi_profile.aliases == ("kimchi-dev",)
        assert kimchi_profile.base_url == "https://llm.kimchi.dev/openai/v1"
        assert kimchi_profile.models_url == "https://llm.kimchi.dev/v1/models/metadata?include_in_cli=true"
        assert kimchi_profile.auth_type == "api_key"
        # Live catalog only: a stale hardcoded list is worse than an empty picker.
        assert kimchi_profile.fallback_models == ()
        # No custom UA: the base fetch_models sends a WAF-safe hermes-cli UA
        # and a custom one would override it.
        assert not kimchi_profile.default_headers.get("User-Agent")


class TestKimchiCatalog:
    def test_metadata_slug_shape_parsed_and_retired_skipped(self, kimchi_profile, monkeypatch):
        """Recorded live shape: ids under 'slug', retired models filtered."""
        seen = _patch_open(
            monkeypatch,
            {
                "models": [
                    {"slug": "kimi-k3", "deprecated_at": "2027-01-01T00:00:00Z"},
                    {"slug": "kimi-k2.5", "deprecated_at": "2020-01-01T00:00:00Z"},
                    {"slug": "glm-5.3"},
                ]
            },
        )
        assert kimchi_profile.fetch_models(api_key="sk-test") == ["kimi-k3", "glm-5.3"]
        assert seen[0]["url"] == "https://llm.kimchi.dev/v1/models/metadata?include_in_cli=true"
        assert seen[0]["auth"] == "Bearer sk-test"
        assert seen[0]["accept"] == "application/json"

    def test_openai_shape_also_parsed(self, kimchi_profile, monkeypatch):
        _patch_open(monkeypatch, {"data": [{"id": "kimi-k3"}, {"id": "glm-5.3"}]})
        assert kimchi_profile.fetch_models(api_key="sk-test") == ["kimi-k3", "glm-5.3"]

    def test_network_failure_returns_none(self, kimchi_profile, monkeypatch):
        _patch_open(monkeypatch, RuntimeError("connection refused"))
        assert kimchi_profile.fetch_models(api_key="sk-test") is None

    def test_custom_base_url_defers_to_base_parser(self, kimchi_profile, monkeypatch):
        """A user-configured endpoint is the user's own — no second-guessing."""
        seen = _patch_open(monkeypatch, {"models": [{"slug": "local-model"}]})
        assert kimchi_profile.fetch_models(api_key="k", base_url="http://localhost:9999/v1") in (None, [])
        assert seen[0]["url"] == "http://localhost:9999/v1/models"


class TestLenientParser:
    @pytest.fixture
    def lenient_model_ids(self, kimchi_profile):
        return sys.modules[type(kimchi_profile).__module__]._lenient_model_ids

    def test_shapes(self, lenient_model_ids):
        assert lenient_model_ids(None) is None
        assert lenient_model_ids("junk") is None
        assert lenient_model_ids({"data": []}) == []
        assert lenient_model_ids([{"id": "a"}, {"model": "b"}, {"name": "c"}]) == ["a", "b", "c"]
        assert lenient_model_ids({"models": [{"slug": "s"}, {"slug": "s"}]}) == ["s"]

    def test_deprecated_filtering(self, lenient_model_ids):
        payload = {
            "models": [
                {"slug": "kept", "deprecated_at": "2099-01-01T00:00:00Z"},
                {"slug": "gone", "deprecated_at": "2020-01-01T00:00:00Z"},
                {"slug": "no-timestamp"},
            ]
        }
        assert lenient_model_ids(payload) == ["kept", "no-timestamp"]


class TestErrorClassification:
    def test_unknown_model_400_classified_as_model_not_found(self, kimchi_profile):
        """Cross-provider default ids (e.g. a leaked global model.default)
        400 with a routing-specific body — classify as model_not_found, not
        format_error (Desktop error observed 2026-09-29)."""
        verdict = kimchi_profile.classify_api_error(
            status_code=400,
            body='{"error":"no registered providers found for the requested model"}',
        )
        assert verdict == {"reason": "model_not_found", "retryable": False, "should_fallback": True}

    def test_unrelated_errors_stay_unclassified(self, kimchi_profile):
        assert kimchi_profile.classify_api_error(status_code=400, body='{"error":"bad json"}') is None
        assert kimchi_profile.classify_api_error(status_code=429, body="no registered providers found") is None
        assert kimchi_profile.classify_api_error(status_code=400) is None
