"""Unit tests for baby care storage, alerts, and start/stop loop control."""

from __future__ import annotations
import json
import asyncio
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from reachy_mini_conversation_app import baby_care


@pytest.fixture(autouse=True)
def _stop_baby_care_after_test() -> None:
    """Ensure module singleton state does not leak across tests."""
    yield
    baby_care.stop_baby_care(MagicMock())


def test_resolve_storage_dir_falls_back_to_instance(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """When USB media is unavailable, photos go under the instance data dir."""
    monkeypatch.setattr(baby_care, "USB_BABY_CARE_DIR", tmp_path / "missing" / "baby-care")
    resolved = baby_care.resolve_storage_dir(tmp_path)
    assert resolved == tmp_path / "media" / "baby-care"
    assert resolved.is_dir()


def test_append_alert_writes_jsonl(tmp_path: Path) -> None:
    """Danger alerts append one JSON object per line with path and timestamp."""
    record = {
        "timestamp": "2026-11-23T13:00:00+00:00",
        "image_path": "/tmp/baby.jpg",
        "safe": False,
        "hazards": ["blanket over face"],
        "summary": "face covered",
    }
    path = baby_care.append_alert(tmp_path, record)
    assert path.name == "alert.jsonl"
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0]) == record


def test_parse_hazard_payload_accepts_fenced_json() -> None:
    """Vision replies may wrap JSON in markdown fences."""
    raw = '```json\n{"safe": false, "hazards": ["sharp object"], "summary": "knife nearby"}\n```'
    assert baby_care.parse_hazard_payload(raw) == {
        "safe": False,
        "hazards": ["sharp object"],
        "summary": "knife nearby",
    }


@pytest.mark.asyncio
async def test_start_and_stop_baby_care_mode(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Start enables follow + loop; stop cancels it and disables tracking."""
    monkeypatch.setattr(baby_care, "USB_BABY_CARE_DIR", tmp_path / "usb" / "baby-care")
    robot = MagicMock()
    robot.media.get_frame_jpeg.return_value = b"\xff\xd8fakejpeg"
    movement = MagicMock()
    ticks = {"n": 0}

    async def analyzer(_jpeg: bytes) -> dict[str, object]:
        ticks["n"] += 1
        return {"safe": True, "hazards": [], "summary": "ok"}

    started = baby_care.start_baby_care(
        robot,
        movement,
        instance_path=tmp_path,
        camera_enabled=True,
        analyzer=analyzer,
        interval_s=0.05,
    )
    assert started["status"] == "started"
    movement.set_head_tracking.assert_called_with(True)
    assert baby_care.is_active()

    for _ in range(50):
        if ticks["n"] >= 1:
            break
        await asyncio.sleep(0.02)
    assert ticks["n"] >= 1

    again = baby_care.start_baby_care(
        robot,
        movement,
        instance_path=tmp_path,
        camera_enabled=True,
        analyzer=analyzer,
        interval_s=0.05,
    )
    assert again["status"] == "already_active"

    stopped = baby_care.stop_baby_care(movement)
    assert stopped == {"status": "stopped", "was_active": True}
    movement.set_head_tracking.assert_called_with(False)
    await asyncio.sleep(0.05)
    assert not baby_care.is_active()


@pytest.mark.asyncio
async def test_danger_writes_alert_and_announces(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Unsafe vision results append alert.jsonl and call the announce callback."""
    monkeypatch.setattr(baby_care, "USB_BABY_CARE_DIR", tmp_path / "usb" / "baby-care")
    monkeypatch.setattr(baby_care, "_read_speaker_volume", lambda _robot: 40)
    monkeypatch.setattr(baby_care, "_set_speaker_volume", MagicMock())
    robot = MagicMock()
    robot.media.get_frame_jpeg.return_value = b"\xff\xd8danger"
    movement = MagicMock()
    announced: list[str] = []

    async def announce(text: str) -> None:
        announced.append(text)

    async def analyzer(_jpeg: bytes) -> dict[str, object]:
        return {"safe": False, "hazards": ["sharp object"], "summary": "scissors"}

    baby_care.start_baby_care(
        robot,
        movement,
        instance_path=tmp_path,
        camera_enabled=True,
        announce=announce,
        analyzer=analyzer,
        interval_s=10.0,
    )
    for _ in range(50):
        if announced:
            break
        await asyncio.sleep(0.02)

    baby_care.stop_baby_care(movement)
    assert announced
    assert "警告" in announced[0] or "danger" in announced[0].lower() or "Speak this warning" in announced[0]
    alert_path = tmp_path / "media" / "baby-care" / "alert.jsonl"
    assert alert_path.is_file()
    payload = json.loads(alert_path.read_text(encoding="utf-8").strip().splitlines()[0])
    assert payload["safe"] is False
    assert payload["hazards"] == ["sharp object"]
    assert Path(payload["image_path"]).is_file()
