"""Local soft/lullaby playback via the Reachy Mini SDK media path.

Uses ``ReachyMini.media.play_sound`` (GStreamer playbin / daemon upload) so
playback goes through the robot speaker without a separate player process.
Stopping must not call ``media.stop_playing()`` — that tears down the
conversation TTS appsrc pipeline — so LOCAL stops replace the playbin with a
tiny silence clip, and wireless also hits ``POST /api/media/stop_sound``.
"""

from __future__ import annotations
import os
import wave
import logging
import threading
from typing import Any
from pathlib import Path

from reachy_mini import ReachyMini
from reachy_mini_conversation_app.daemon_api import DaemonApiError, daemon_request


logger = logging.getLogger(__name__)

SOFT_MUSIC_DIR_ENV = "SOFT_MUSIC_DIR"
DEFAULT_SOFT_VOLUME = 25
_AUDIO_EXTENSIONS = {".wav", ".mp3", ".ogg", ".flac", ".m4a"}
_SILENCE_NAME = "_silence_stop.wav"
_STATE_LOCK = threading.Lock()
_state: dict[str, Any] = {"track": None, "previous_volume": None}


def bundled_lullabies_dir() -> Path:
    """Directory of packaged royalty-safe sample tracks."""
    return Path(__file__).resolve().parent.parent / "sounds" / "lullabies"


def resolve_music_dirs(instance_path: str | Path | None = None) -> list[Path]:
    """Return music search roots: env override, instance folder, then bundled samples."""
    dirs: list[Path] = []
    env_dir = os.getenv(SOFT_MUSIC_DIR_ENV, "").strip()
    if env_dir:
        dirs.append(Path(env_dir).expanduser())
    if instance_path is not None:
        dirs.append(Path(instance_path).expanduser() / "sounds" / "lullabies")
    dirs.append(bundled_lullabies_dir())
    unique: list[Path] = []
    seen: set[Path] = set()
    for directory in dirs:
        try:
            resolved = directory.resolve()
        except OSError:
            resolved = directory
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(directory)
    return unique


def list_tracks(instance_path: str | Path | None = None) -> list[Path]:
    """List playable tracks (user dirs first; skip internal silence helper)."""
    tracks: list[Path] = []
    seen_names: set[str] = set()
    for directory in resolve_music_dirs(instance_path):
        if not directory.is_dir():
            continue
        for path in sorted(directory.iterdir()):
            if not path.is_file():
                continue
            if path.name.startswith("_") or path.name == "LICENSE.txt":
                continue
            if path.suffix.lower() not in _AUDIO_EXTENSIONS:
                continue
            key = path.name.lower()
            if key in seen_names:
                continue
            seen_names.add(key)
            tracks.append(path)
    return tracks


def resolve_track(name: str | None, instance_path: str | Path | None = None) -> Path:
    """Resolve a track by stem/filename, or pick the first catalog entry."""
    tracks = list_tracks(instance_path)
    if not tracks:
        raise FileNotFoundError(
            "No soft-music tracks found. Add .wav/.mp3 files under "
            f"{SOFT_MUSIC_DIR_ENV}, <instance>/sounds/lullabies/, or the bundled samples directory."
        )
    if name is None or not str(name).strip():
        return tracks[0]
    needle = str(name).strip().lower()
    for path in tracks:
        if path.name.lower() == needle or path.stem.lower() == needle:
            return path
    available = ", ".join(path.stem for path in tracks)
    raise FileNotFoundError(f"Unknown track {name!r}. Available: {available}")


def _silence_path() -> Path:
    path = bundled_lullabies_dir() / _SILENCE_NAME
    if path.is_file():
        return path
    # Tests / editable installs without packaged silence: write a tiny wav beside the module.
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "w") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(b"\x00\x00" * 800)
    return path


def _read_speaker_volume(robot: ReachyMini) -> int | None:
    try:
        response = daemon_request(robot, "/api/volume/current")
    except DaemonApiError as e:
        logger.debug("Could not read speaker volume: %s", e)
        return None
    if isinstance(response, dict) and isinstance(response.get("volume"), (int, float)):
        return int(response["volume"])
    return None


def _set_speaker_volume(robot: ReachyMini, level: int) -> None:
    daemon_request(robot, "/api/volume/set", method="POST", payload={"volume": int(level)}, timeout_s=5.0)


def stop_soft_music(robot: ReachyMini) -> dict[str, Any]:
    """Stop file playback started for soft music without killing conversation TTS."""
    with _STATE_LOCK:
        previous = _state.get("previous_volume")
        track = _state.get("track")
        _state["track"] = None
        _state["previous_volume"] = None

    # Wireless / daemon-uploaded sounds.
    try:
        daemon_request(robot, "/api/media/stop_sound", method="POST", timeout_s=5.0)
    except DaemonApiError as e:
        logger.debug("Daemon stop_sound not available: %s", e)

    # LOCAL playbin: play_sound replaces the active file player; do not call stop_playing().
    try:
        robot.media.play_sound(str(_silence_path()))
    except Exception as e:
        logger.warning("LOCAL soft-music stop via silence clip failed: %s", e)

    if previous is not None:
        try:
            _set_speaker_volume(robot, int(previous))
        except DaemonApiError as e:
            logger.warning("Could not restore speaker volume after soft music: %s", e)

    return {"status": "stopped", "was_playing": track}


def play_soft_music(
    robot: ReachyMini,
    *,
    track: str | None = None,
    volume: int | None = DEFAULT_SOFT_VOLUME,
    instance_path: str | Path | None = None,
) -> dict[str, Any]:
    """Play a local soft track on the robot speaker at a calm volume."""
    path = resolve_track(track, instance_path)
    if volume is not None and not 0 <= int(volume) <= 100:
        raise ValueError("volume must be between 0 and 100")

    previous_volume: int | None = None
    if volume is not None:
        previous_volume = _read_speaker_volume(robot)
        try:
            _set_speaker_volume(robot, int(volume))
        except DaemonApiError as e:
            logger.warning("Could not set soft-music volume (continuing at current level): %s", e)
            previous_volume = None

    try:
        robot.media.play_sound(str(path.resolve()))
    except Exception:
        if previous_volume is not None:
            try:
                _set_speaker_volume(robot, previous_volume)
            except DaemonApiError:
                pass
        raise

    with _STATE_LOCK:
        _state["track"] = path.stem
        _state["previous_volume"] = previous_volume

    return {
        "status": "playing",
        "track": path.stem,
        "path": str(path),
        "volume": volume,
        "available_tracks": [item.stem for item in list_tracks(instance_path)],
    }


def catalog_summary(instance_path: str | Path | None = None) -> dict[str, Any]:
    """Return the visible track list for tool responses / docs."""
    tracks = list_tracks(instance_path)
    return {
        "tracks": [path.stem for path in tracks],
        "directories": [str(path) for path in resolve_music_dirs(instance_path)],
        "default_volume": DEFAULT_SOFT_VOLUME,
    }


def clamp_soft_volume(raw: object) -> int:
    """Parse a volume argument into 0–100, defaulting to the soft preset."""
    if raw is None or raw == "":
        return DEFAULT_SOFT_VOLUME
    if isinstance(raw, bool) or not isinstance(raw, (int, float, str)):
        raise ValueError(f"volume must be an integer, got {raw!r}")
    try:
        value = int(raw)
    except (TypeError, ValueError) as e:
        raise ValueError(f"volume must be an integer, got {raw!r}") from e
    if not 0 <= value <= 100:
        raise ValueError("volume must be between 0 and 100")
    return value


def approx_duration_s(path: Path) -> float | None:
    """Best-effort WAV duration for logging; non-wav returns None."""
    if path.suffix.lower() != ".wav":
        return None
    try:
        with wave.open(str(path), "r") as handle:
            frames = handle.getnframes()
            rate = handle.getframerate()
            if rate <= 0:
                return None
            return frames / float(rate)
    except Exception:
        return None


def format_track_list(instance_path: str | Path | None = None) -> str:
    """Comma-separated stems for LLM-facing error messages."""
    tracks = list_tracks(instance_path)
    if not tracks:
        return "(none)"
    return ", ".join(path.stem for path in tracks)
