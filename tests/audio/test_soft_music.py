"""Tests for local soft-music catalog and playback helpers."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from reachy_mini_conversation_app.audio import soft_music
from reachy_mini_conversation_app.daemon_api import DaemonApiError


def test_bundled_catalog_lists_royalty_safe_samples() -> None:
    """Packaged lullabies are discoverable and exclude helper/license files."""
    tracks = soft_music.list_tracks()
    stems = {path.stem for path in tracks}
    assert "gentle_sway" in stems
    assert "soft_chimes" in stems
    assert "_silence_stop" not in stems


def test_resolve_music_dirs_defaults_to_usb_media(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Unset SOFT_MUSIC_DIR uses the robot USB media root; override still wins."""
    monkeypatch.delenv(soft_music.SOFT_MUSIC_DIR_ENV, raising=False)
    dirs = soft_music.resolve_music_dirs(instance_path=tmp_path)
    assert dirs[0] == Path(soft_music.DEFAULT_SOFT_MUSIC_DIR)
    assert dirs[0] == Path("/mnt/usb_storage/media")
    assert (tmp_path / "sounds" / "lullabies") in dirs
    assert soft_music.bundled_lullabies_dir() in dirs

    override = tmp_path / "custom_media"
    override.mkdir()
    monkeypatch.setenv(soft_music.SOFT_MUSIC_DIR_ENV, str(override))
    overridden = soft_music.resolve_music_dirs(instance_path=tmp_path)
    assert overridden[0] == override

    monkeypatch.setenv(soft_music.SOFT_MUSIC_DIR_ENV, "")
    skipped = soft_music.resolve_music_dirs(instance_path=tmp_path)
    assert Path(soft_music.DEFAULT_SOFT_MUSIC_DIR) not in skipped
    assert skipped[0] == tmp_path / "sounds" / "lullabies"


def test_resolve_track_by_stem_and_unknown(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """User directories override bundled names; unknown stems raise clearly."""
    monkeypatch.setenv(soft_music.SOFT_MUSIC_DIR_ENV, "")
    custom = tmp_path / "sounds" / "lullabies"
    custom.mkdir(parents=True)
    (custom / "moonbeam.wav").write_bytes(b"RIFF")
    path = soft_music.resolve_track("moonbeam", instance_path=tmp_path)
    assert path.name == "moonbeam.wav"
    with pytest.raises(FileNotFoundError, match="Unknown track"):
        soft_music.resolve_track("not-a-track", instance_path=tmp_path)


def test_play_and_stop_use_sdk_play_sound_not_stop_playing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Playback goes through play_sound; stop must not tear down TTS via stop_playing."""
    robot = MagicMock()
    volumes: list[int] = []

    def fake_daemon(robot_arg: object, path: str, **kwargs: object) -> dict[str, int]:
        if path.endswith("/current"):
            return {"volume": 60}
        if path.endswith("/set"):
            payload = kwargs.get("payload")
            assert isinstance(payload, dict)
            volumes.append(int(payload["volume"]))
            return {"volume": int(payload["volume"])}
        if path == "/api/media/stop_sound":
            return {}
        raise DaemonApiError(f"unexpected {path}")

    monkeypatch.setattr(soft_music, "daemon_request", fake_daemon)
    result = soft_music.play_soft_music(robot, track="gentle_sway", volume=20)
    assert result["status"] == "playing"
    assert result["track"] == "gentle_sway"
    robot.disable_wobbling.assert_called()
    robot.media.play_sound.assert_called()
    robot.media.stop_playing.assert_not_called()
    assert volumes == [20]

    stopped = soft_music.stop_soft_music(robot)
    assert stopped["status"] == "stopped"
    robot.media.stop_playing.assert_not_called()
    # Silence clip + restored volume
    assert robot.media.play_sound.call_count >= 2
    assert volumes[-1] == 60


def test_clamp_soft_volume_defaults_and_rejects() -> None:
    """Default soft-music volume is audible on-robot; out-of-range values are rejected."""
    assert soft_music.DEFAULT_SOFT_VOLUME == 80
    assert soft_music.clamp_soft_volume(None) == soft_music.DEFAULT_SOFT_VOLUME
    assert soft_music.clamp_soft_volume(10) == 10
    with pytest.raises(ValueError):
        soft_music.clamp_soft_volume(200)
