"""Unit tests for the Kimchi harness ACP provider profile.

The profile spawns the local Kimchi CLI (``kimchi --mode acp --yolo``) over
the Agent Client Protocol. Key behaviors under test:

- the client NEVER falls back to the shim's copilot defaults, even on
  construction paths that don't pass command/args (observed live in the
  Desktop app, where the spawn failed with ``command 'copilot'``);
- harness-native tool activity (``tool_call``/``tool_call_update`` session
  updates, which the shim otherwise drops) is rendered as markdown bullets
  into the visible reply so it shows in Hermes' UI AND reaches later turns'
  flattened context;
- the provider-name placeholder model means "harness session default".
"""

from __future__ import annotations

import json

import pytest


@pytest.fixture
def acp_profile():
    import model_tools  # noqa: F401
    import providers

    profile = providers.get_provider_profile("kimchi-acp")
    assert profile is not None, "kimchi-acp provider profile must be registered"
    return profile


@pytest.fixture(autouse=True)
def _clean_acp_env(monkeypatch):
    """Isolate from operator environments (Desktop runs may export these)."""
    for var in ("KIMCHI_ACP_COMMAND", "KIMCHI_ACP_ARGS", "KIMCHI_API_KEY", "KIMCHI_CONFIG_PATH"):
        monkeypatch.delenv(var, raising=False)


class TestKimchiACPProfile:
    def test_external_process_fields(self, acp_profile):
        assert acp_profile.name == "kimchi-acp"
        assert acp_profile.auth_type == "external_process"
        assert acp_profile.env_vars == ()  # subprocess owns auth
        assert acp_profile.process_command == "kimchi"
        assert acp_profile.process_args == ("--mode", "acp", "--yolo")
        assert acp_profile.process_args_env_var == "KIMCHI_ACP_ARGS"
        assert acp_profile.base_url == "acp://kimchi"
        # Placeholder id meaning "harness session default", merged into the
        # /model picker alongside the live session catalog.
        assert acp_profile.fallback_models == ("kimchi-acp",)

    def test_create_client_returns_kimchi_subclass(self, acp_profile):
        from agent.copilot_acp_client import CopilotACPClient

        client = acp_profile.create_client(command="kimchi", args=("--mode", "acp"))
        assert isinstance(client, CopilotACPClient)
        assert type(client).__name__ == "KimchiACPClient"


class TestSpawnTargetHealing:
    def test_shim_copilot_defaults_never_leak(self, acp_profile):
        """Construction without command/args (Desktop/auxiliary rebuild paths)
        must resolve to the Kimchi CLI, not the shim's copilot fallback."""
        client = acp_profile.create_client(api_key="x", base_url="acp://copilot")
        assert client._acp_command == "kimchi"
        assert tuple(client._acp_args) == ("--mode", "acp", "--yolo")
        assert client.base_url == "acp://kimchi"

    def test_explicit_command_respected(self, acp_profile):
        client = acp_profile.create_client(command="/custom/kimchi", args=("--mode", "acp"))
        assert client._acp_command == "/custom/kimchi"
        assert list(client._acp_args) == ["--mode", "acp"]

    def test_env_args_escape_hatch(self, acp_profile, monkeypatch):
        monkeypatch.setenv("KIMCHI_ACP_ARGS", "--mode acp")  # documented non-YOLO path
        client = acp_profile.create_client()
        assert client._acp_command == "kimchi"
        assert list(client._acp_args) == ["--mode", "acp"]

    def test_spawn_error_rebranded(self, acp_profile):
        client = acp_profile.create_client(command="/nonexistent-kimchi-probe-binary")
        with pytest.raises(RuntimeError) as excinfo:
            client._spawn()
        message = str(excinfo.value)
        assert "Kimchi ACP" in message
        assert "KIMCHI_ACP_COMMAND" in message
        assert "Copilot" not in message


class TestToolActivityRendering:
    @staticmethod
    def _kw(text_parts):
        return dict(process=None, cwd="/tmp", text_parts=text_parts, reasoning_parts=[], allow_file_requests=True)

    @staticmethod
    def _update(client, text_parts, update):
        client._handle_server_message(
            {"method": "session/update", "params": {"update": update}},
            process=None, cwd="/tmp", text_parts=text_parts, reasoning_parts=[],
            allow_file_requests=True,
        )

    def test_one_bullet_per_tool_on_terminal_status(self, acp_profile):
        client = acp_profile.create_client()
        tp = []
        self._update(client, tp, {
            "sessionUpdate": "tool_call", "toolCallId": "t1", "kind": "edit",
            "title": "/tmp/date.py", "status": "in_progress"})
        assert tp == []  # registration only — no pending/in_progress churn

        self._update(client, tp, {
            "sessionUpdate": "tool_call_update", "toolCallId": "t1", "status": "completed",
            "content": [{"type": "content", "content": {"type": "text", "text": "wrote 12 bytes"}}]})
        assert tp == ["- ⚙ **/tmp/date.py** ✓ — wrote 12 bytes\n"]

        # narrative after the bullet block is separated by a blank line so
        # markdown renders it as its own paragraph
        self._update(client, tp, {
            "sessionUpdate": "agent_message_chunk", "content": {"type": "text", "text": "Done."}})
        assert tp[-2:] == ["\n", "Done."]

        self._update(client, tp, {
            "sessionUpdate": "tool_call", "toolCallId": "t2", "kind": "search", "title": "web_search"})
        self._update(client, tp, {
            "sessionUpdate": "tool_call_update", "toolCallId": "t2", "status": "failed"})
        assert tp[-1] == "- ⚙ **web_search** ✗\n"

    def test_adjacent_duplicates_collapse_and_excerpts_truncate(self, acp_profile):
        client = acp_profile.create_client()
        tp = []
        self._update(client, tp, {
            "sessionUpdate": "tool_call", "toolCallId": "t1", "title": "web_search"})
        for _ in range(2):  # duplicate/re-delivered completed events
            self._update(client, tp, {
                "sessionUpdate": "tool_call_update", "toolCallId": "t1", "status": "completed",
                "content": [{"type": "content", "content": {"type": "text", "text": "same result"}}]})
        assert tp == ["- ⚙ **web_search** ✓ — same result\n"]

        long_text = "x" * 400
        self._update(client, tp, {
            "sessionUpdate": "tool_call", "toolCallId": "t2", "title": "big run"})
        self._update(client, tp, {
            "sessionUpdate": "tool_call_update", "toolCallId": "t2", "status": "completed",
            "content": [{"type": "text", "text": long_text}]})
        out = tp[-1]
        assert out.startswith("- ⚙ **big run** ✓ — ") and out.endswith("…\n")
        assert len(out) <= len("- ⚙ **big run** ✓ — ") + 100 + 1


class TestPlaceholderAndCatalog:
    def test_placeholder_model_skips_selection(self, acp_profile, monkeypatch):
        """'kimchi-acp' as a model id means 'harness session default' — the
        client must not send a model-selection request (spurious warning)."""
        client = acp_profile.create_client()
        captured = {}

        def fake_run(_self, prompt_text, *, timeout_seconds, model=None):
            captured["model"] = model
            return "ok", ""

        monkeypatch.setattr(type(client).__mro__[1], "_run_prompt", fake_run)
        client._run_prompt("hi", timeout_seconds=5, model="kimchi-acp")
        assert captured["model"] is None

        client._run_prompt("hi", timeout_seconds=5, model="kimi-k3")
        assert captured["model"] == "kimi-k3"  # real models pass through

    def test_fetch_models_filters_duplicates_not_content(self, acp_profile, monkeypatch):
        import hermes_cli.auth

        monkeypatch.setattr(
            hermes_cli.auth,
            "resolve_external_process_provider_credentials",
            lambda name: {"base_url": "acp://kimchi", "api_key": None, "command": "kimchi", "args": ("--mode", "acp", "--yolo")},
        )
        client = acp_profile.create_client()
        monkeypatch.setattr(
            type(client), "list_models",
            lambda self, **kw: ["multi-model", "auto", "kimi-k3", "kimi-k3"],
            raising=False,
        )
        # Live session ids ship unfiltered (auto is the harness's router mode,
        # a real selection) — only duplicates are removed.
        assert acp_profile.fetch_models() == ["multi-model", "auto", "kimi-k3"]


class TestSetupStatus:
    def test_logged_in_via_config(self, acp_profile, monkeypatch, tmp_path):
        import shutil

        config = tmp_path / "config.json"
        config.write_text(json.dumps({"apiKey": "sk-secret"}), encoding="utf-8")
        monkeypatch.setenv("KIMCHI_CONFIG_PATH", str(config))
        monkeypatch.setattr(shutil, "which", lambda name: "/usr/local/bin/kimchi" if name == "kimchi" else None)

        status = acp_profile.setup_status()
        assert status == {
            "available": True,
            "logged_in": True,
            "plan": None,
            "detail": "kimchi found; logged in",
            "login_command": "kimchi login",
        }

    def test_cli_missing_reports_remediation(self, acp_profile, monkeypatch, tmp_path):
        import shutil

        monkeypatch.setenv("KIMCHI_CONFIG_PATH", str(tmp_path / "missing.json"))
        monkeypatch.setattr(shutil, "which", lambda name: None)

        status = acp_profile.setup_status()
        assert status["available"] is False
        assert "KIMCHI_ACP_COMMAND" in status["detail"]
        assert status["login_command"] == "kimchi login"
