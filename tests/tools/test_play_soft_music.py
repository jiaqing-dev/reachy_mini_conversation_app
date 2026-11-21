"""Tests for soft-music conversation tools."""

from unittest.mock import MagicMock

import pytest

import reachy_mini_conversation_app.tools.play_soft_music as play_mod
import reachy_mini_conversation_app.tools.stop_soft_music as stop_mod
from reachy_mini_conversation_app.tools.core_tools import ToolDependencies
from reachy_mini_conversation_app.tools.play_soft_music import PlaySoftMusic
from reachy_mini_conversation_app.tools.stop_soft_music import StopSoftMusic


def _deps() -> ToolDependencies:
    return ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock(), instance_path=None)


@pytest.mark.asyncio
async def test_play_soft_music_tool_happy_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """The play tool forwards track/volume and returns the helper payload."""
    monkeypatch.setattr(
        play_mod.soft_music,
        "play_soft_music",
        MagicMock(return_value={"status": "playing", "track": "gentle_sway", "volume": 25, "path": "/tmp/x.wav"}),
    )
    monkeypatch.setattr(play_mod.soft_music, "approx_duration_s", MagicMock(return_value=40.0))
    monkeypatch.setattr(play_mod.soft_music, "resolve_track", MagicMock(return_value=MagicMock()))

    result = await PlaySoftMusic()(_deps(), track="gentle_sway")
    assert result["status"] == "playing"
    assert result["duration_s"] == 40.0


@pytest.mark.asyncio
async def test_play_soft_music_tool_returns_error_for_missing_track(monkeypatch: pytest.MonkeyPatch) -> None:
    """Missing tracks become tool errors instead of crashing the conversation."""
    monkeypatch.setattr(
        play_mod.soft_music,
        "play_soft_music",
        MagicMock(side_effect=FileNotFoundError("Unknown track 'nope'")),
    )
    monkeypatch.setattr(play_mod.soft_music, "format_track_list", MagicMock(return_value="gentle_sway"))
    result = await PlaySoftMusic()(_deps(), track="nope")
    assert "error" in result
    assert result["available_tracks"] == "gentle_sway"


@pytest.mark.asyncio
async def test_stop_soft_music_tool(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stop tool returns the helper status dict."""
    monkeypatch.setattr(stop_mod.soft_music, "stop_soft_music", MagicMock(return_value={"status": "stopped"}))
    result = await StopSoftMusic()(_deps())
    assert result == {"status": "stopped"}
