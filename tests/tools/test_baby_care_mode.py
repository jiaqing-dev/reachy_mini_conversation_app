"""Tests for baby-care conversation tools."""

from unittest.mock import MagicMock

import pytest

import reachy_mini_conversation_app.tools.stop_baby_care_mode as stop_mod
import reachy_mini_conversation_app.tools.start_baby_care_mode as start_mod
from reachy_mini_conversation_app.tools.core_tools import ToolDependencies
from reachy_mini_conversation_app.tools.stop_baby_care_mode import StopBabyCareMode
from reachy_mini_conversation_app.tools.start_baby_care_mode import StartBabyCareMode


def _deps() -> ToolDependencies:
    return ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock(), instance_path=None)


@pytest.mark.asyncio
async def test_start_baby_care_mode_tool(monkeypatch: pytest.MonkeyPatch) -> None:
    """Start tool forwards deps into the baby_care helper."""
    monkeypatch.setattr(
        start_mod.baby_care,
        "start_baby_care",
        MagicMock(return_value={"status": "started", "interval_s": 30.0}),
    )
    result = await StartBabyCareMode()(_deps())
    assert result["status"] == "started"
    start_mod.baby_care.start_baby_care.assert_called_once()


@pytest.mark.asyncio
async def test_start_baby_care_mode_tool_returns_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Helper failures become tool errors instead of crashing the loop."""
    monkeypatch.setattr(
        start_mod.baby_care,
        "start_baby_care",
        MagicMock(side_effect=RuntimeError("no loop")),
    )
    result = await StartBabyCareMode()(_deps())
    assert "error" in result


@pytest.mark.asyncio
async def test_stop_baby_care_mode_tool(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stop tool returns the helper status dict."""
    monkeypatch.setattr(
        stop_mod.baby_care,
        "stop_baby_care",
        MagicMock(return_value={"status": "stopped", "was_active": True}),
    )
    result = await StopBabyCareMode()(_deps())
    assert result == {"status": "stopped", "was_active": True}


def test_default_profile_instructs_baby_care_tool_use() -> None:
    """The packaged default prompt must map Chinese care phrases to the tools."""
    from reachy_mini_conversation_app.profile_store import read_packaged_default_profile

    profile = read_packaged_default_profile()
    assert "start_baby_care_mode" in profile.default_tools
    assert "stop_baby_care_mode" in profile.default_tools
    assert "start_baby_care_mode" in profile.instructions
    assert "进入宝宝看护模式" in profile.instructions
    assert "退出宝宝看护模式" in profile.instructions
