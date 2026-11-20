"""Tests for wake-word detection helpers."""

import hashlib
import threading
from pathlib import Path
from functools import partial
from unittest.mock import MagicMock

import httpx
import numpy as np
import pytest

import reachy_mini_conversation_app.audio.wake_word as wake_mod
from reachy_mini_conversation_app.audio.wake_word import WakeWordDetector, listen_for_wake_word


@pytest.fixture
def model_cache(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> tuple[Path, MagicMock]:
    """Isolate wake-word model downloads from the network."""
    payloads = {filename: f"model-{filename}".encode() for filename in wake_mod._MODEL_FILES}

    def respond(request: httpx.Request) -> httpx.Response:
        for filename, (url, _checksum) in wake_mod._MODEL_FILES.items():
            if str(request.url) == url:
                return httpx.Response(200, content=payloads[filename])
        raise AssertionError(f"Unexpected wake-word download URL: {request.url}")

    download = MagicMock(side_effect=respond)
    monkeypatch.setattr(wake_mod, "cached_assets_path", lambda *_args: tmp_path)
    monkeypatch.setattr(
        wake_mod,
        "_MODEL_FILES",
        {
            filename: (url, hashlib.sha256(payloads[filename]).hexdigest())
            for filename, (url, _checksum) in wake_mod._MODEL_FILES.items()
        },
    )
    monkeypatch.setattr(wake_mod.httpx, "Client", partial(httpx.Client, transport=httpx.MockTransport(download)))

    def fake_session(path: str, *args: object, **kwargs: object) -> MagicMock:
        session = MagicMock()
        name = Path(path).name
        if name == "melspectrogram.onnx":
            session.run.side_effect = lambda _outputs, inputs: [
                np.ones((1, max(1, int(inputs["input"].shape[1]) // 160), 1, 32), dtype=np.float32)
            ]
        elif name == "embedding_model.onnx":
            session.run.side_effect = lambda _outputs, inputs: [
                np.zeros((inputs["input_1"].shape[0], 1, 1, 96), dtype=np.float32)
            ]
        else:
            session.get_inputs.return_value = [MagicMock(name="onnx::Flatten_0")]
            session.run.side_effect = lambda _outputs, _inputs: [np.array([[0.1]], dtype=np.float32)]
        return session

    monkeypatch.setattr(wake_mod.ort, "InferenceSession", fake_session)
    return tmp_path, download


def test_detector_downloads_once_then_reuses_cache(
    model_cache: tuple[Path, MagicMock],
) -> None:
    """First construction downloads the three ONNX files; the next start is offline."""
    cache, download = model_cache
    WakeWordDetector(threshold=0.5)
    assert download.call_count == 3
    assert {(cache / name).is_file() for name in wake_mod._MODEL_FILES} == {True}

    download.reset_mock()
    WakeWordDetector(threshold=0.5)
    download.assert_not_called()


def test_process_rejects_non_16k_and_detected_respects_cooldown(
    monkeypatch: pytest.MonkeyPatch, model_cache: tuple[Path, MagicMock]
) -> None:
    """Only 16 kHz PCM is accepted; detections debounce for the configured cooldown."""
    detector = WakeWordDetector(threshold=0.4, cooldown_s=2.0)
    with pytest.raises(ValueError, match="16 kHz|16000"):
        detector.process(np.zeros(1280, dtype=np.int16), 48000)

    # Force a high score without needing the streaming pipeline to warm up.
    assert detector.detected(0.9, now=10.0) is True
    assert detector.detected(0.9, now=11.0) is False
    assert detector.detected(0.9, now=12.5) is True
    assert detector.detected(0.2, now=15.0) is False


def test_microphone_frame_to_int16_scales_float32_not_truncates() -> None:
    """SDK float32 frames must scale to int16; bare dtype casts would become silence."""
    mono = np.full(8, 0.5, dtype=np.float32)
    pcm = wake_mod._microphone_frame_to_int16(mono)
    assert pcm.dtype == np.int16
    assert int(np.max(np.abs(pcm))) == 16383

    # The pre-fix bug path: casting float32 speech to int16 truncates to 0.
    broken = np.asarray(mono, dtype=np.int16)
    assert int(np.max(np.abs(broken))) == 0

    # Conversation path keeps the first channel of SDK stereo frames.
    stereo = np.column_stack((mono, np.full(8, 0.1, dtype=np.float32)))
    stereo_pcm = wake_mod._microphone_frame_to_int16(stereo)
    assert stereo_pcm.shape == (8,)
    assert int(np.max(np.abs(stereo_pcm))) == 16383


def test_listen_for_wake_word_returns_on_detection_and_stop(
    monkeypatch: pytest.MonkeyPatch, model_cache: tuple[Path, MagicMock]
) -> None:
    """The listen helper hands off on a match and exits cleanly when stop_event is set."""
    detector = WakeWordDetector(threshold=0.5, cooldown_s=0.0)
    robot = MagicMock()
    robot.media.get_input_audio_samplerate.return_value = 16000
    # SDK LOCAL/GStreamer mic frames are float32 in [-1, 1], not int16.
    robot.media.get_audio_sample.side_effect = lambda: np.full(320, 0.05, dtype=np.float32)

    seen_pcm: list[np.ndarray] = []
    call_count = {"n": 0}

    def fake_process(pcm: np.ndarray, sample_rate: int) -> float:
        seen_pcm.append(pcm)
        call_count["n"] += 1
        return 0.1 if call_count["n"] < 2 else 0.9

    monkeypatch.setattr(detector, "process", fake_process)
    monkeypatch.setattr(wake_mod, "_SCORE_LOG_INTERVAL_S", 10_000.0)
    monkeypatch.setattr(wake_mod.time, "sleep", lambda _seconds: None)
    assert listen_for_wake_word(robot, detector=detector, own_recording=True) is True
    robot.media.start_recording.assert_called_once_with()
    robot.media.stop_recording.assert_called_once_with()
    assert seen_pcm[0].dtype == np.int16
    assert int(np.max(np.abs(seen_pcm[0]))) > 0

    stop_event = threading.Event()
    stop_event.set()
    assert listen_for_wake_word(robot, detector=detector, stop_event=stop_event, own_recording=False) is False
    assert robot.media.start_recording.call_count == 1
