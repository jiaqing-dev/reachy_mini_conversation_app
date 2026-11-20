"""OpenWakeWord-style ONNX wake-word detection for the robot microphone.

Uses the "reechy" model from andyjmorgan/reachy-wake-word (CC BY-NC-SA 4.0) plus
openWakeWord's melspectrogram and speech-embedding ONNX preprocessors (Apache 2.0).
Inference follows the same 80 ms / 1280-sample framing as that project's test utility.
"""

from __future__ import annotations
import time
import hashlib
import logging
import threading
from typing import Any
from pathlib import Path
from tempfile import NamedTemporaryFile
from collections import deque
from collections.abc import Callable

import httpx
import numpy as np
import onnxruntime as ort
from numpy.typing import NDArray
from huggingface_hub import cached_assets_path

from reachy_mini import ReachyMini
from reachy_mini_conversation_app.streaming import audio_to_int16


logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
CHUNK_SAMPLES = 1280  # 80 ms at 16 kHz
DEFAULT_THRESHOLD = 0.5
DEFAULT_COOLDOWN_S = 2.0
_FEATURE_FRAMES = 16
_FEATURE_DIM = 96
_MEL_WINDOW = 76
_EMPTY_POLL_SLEEP_S = 0.005
_SCORE_LOG_INTERVAL_S = 2.0
_MODEL_CACHE_NAMESPACE = "reachy_mini_conversation_app"
_MODEL_CACHE_SUBDIR = "wake-word"
_MODEL_CACHE_REVISION = "reechy-v1"

# Hosted next to the reference training repo; checksums pin the exact bytes.
_MODEL_FILES: dict[str, tuple[str, str]] = {
    "melspectrogram.onnx": (
        "https://raw.githubusercontent.com/andyjmorgan/reachy-wake-word/"
        "main/openwakeword/resources/models/melspectrogram.onnx",
        "ba2b0e0f8b7b875369a2c89cb13360ff53bac436f2895cced9f479fa65eb176f",
    ),
    "embedding_model.onnx": (
        "https://raw.githubusercontent.com/andyjmorgan/reachy-wake-word/"
        "main/openwakeword/resources/models/embedding_model.onnx",
        "70d164290c1d095d1d4ee149bc5e00543250a7316b59f31d056cff7bd3075c1f",
    ),
    "reechy-spk150-steps100k-acc98.15-rec97.50.onnx": (
        "https://raw.githubusercontent.com/andyjmorgan/reachy-wake-word/"
        "main/models/reechy-spk150-steps100k-acc98.15-rec97.50.onnx",
        "bcd551a49315e443b11877ff52e11f30e42ecdf024b6c051dcaad80faee1e14f",
    ),
}


def _download_model_files(model_directory: Path) -> dict[str, Path]:
    """Download or reuse cached ONNX files, verifying SHA-256."""
    paths: dict[str, Path] = {}
    with httpx.Client(follow_redirects=True, timeout=60.0) as client:
        for filename, (url, checksum) in _MODEL_FILES.items():
            cached_file = model_directory / filename
            if cached_file.is_file() and hashlib.sha256(cached_file.read_bytes()).hexdigest() == checksum:
                paths[filename] = cached_file
                continue
            if cached_file.is_file():
                logger.warning("Cached wake-word file failed checksum verification: %s", cached_file)
            logger.info("Downloading wake-word model file: %s", filename)
            response = client.get(url)
            response.raise_for_status()
            if hashlib.sha256(response.content).hexdigest() != checksum:
                raise ValueError(f"Downloaded wake-word file failed checksum verification: {filename}")
            download = NamedTemporaryFile(dir=model_directory, delete=False)
            download_path = Path(download.name)
            try:
                with download:
                    download.write(response.content)
                download_path.replace(cached_file)
            finally:
                download_path.unlink(missing_ok=True)
            paths[filename] = cached_file
    return paths


class WakeWordDetector:
    """Score streaming 16 kHz mono PCM for the Reachy "reechy" wake prompt."""

    def __init__(self, *, threshold: float = DEFAULT_THRESHOLD, cooldown_s: float = DEFAULT_COOLDOWN_S) -> None:
        """Load preprocessor + wake models from the Hugging Face assets cache."""
        if not 0.0 < threshold <= 1.0:
            raise ValueError(f"Wake-word threshold must be in (0, 1], got {threshold}")
        if cooldown_s < 0.0:
            raise ValueError(f"Wake-word cooldown must be >= 0, got {cooldown_s}")

        model_directory = cached_assets_path(_MODEL_CACHE_NAMESPACE, _MODEL_CACHE_SUBDIR, _MODEL_CACHE_REVISION)
        paths = _download_model_files(model_directory)

        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        providers = ["CPUExecutionProvider"]
        self._melspec = ort.InferenceSession(
            str(paths["melspectrogram.onnx"]), sess_options=options, providers=providers
        )
        self._embedding = ort.InferenceSession(
            str(paths["embedding_model.onnx"]), sess_options=options, providers=providers
        )
        self._wake = ort.InferenceSession(
            str(paths["reechy-spk150-steps100k-acc98.15-rec97.50.onnx"]),
            sess_options=options,
            providers=providers,
        )
        self._wake_input_name = self._wake.get_inputs()[0].name

        self.threshold = threshold
        self.cooldown_s = cooldown_s
        self.last_score = 0.0
        self._last_detection_monotonic = 0.0
        self._raw_buffer: deque[int] = deque(maxlen=SAMPLE_RATE * 10)
        self._raw_remainder = np.empty(0, dtype=np.int16)
        self._accumulated_samples = 0
        self._melspectrogram_buffer = np.ones((_MEL_WINDOW, 32), dtype=np.float32)
        self._feature_buffer = self._bootstrap_features()

    def reset(self) -> None:
        """Clear streaming state between listen sessions."""
        self.last_score = 0.0
        self._last_detection_monotonic = 0.0
        self._raw_buffer.clear()
        self._raw_remainder = np.empty(0, dtype=np.int16)
        self._accumulated_samples = 0
        self._melspectrogram_buffer = np.ones((_MEL_WINDOW, 32), dtype=np.float32)
        self._feature_buffer = self._bootstrap_features()

    def _bootstrap_features(self) -> NDArray[np.float32]:
        seed = np.random.randint(-1000, 1000, SAMPLE_RATE * 4, dtype=np.int16)
        return self._embeddings_from_pcm(seed)

    def _melspectrogram(self, pcm: NDArray[np.int16]) -> NDArray[np.float32]:
        x = pcm.reshape(1, -1).astype(np.float32)
        spec = np.squeeze(self._melspec.run(None, {"input": x})[0])
        return np.asarray(spec / 10.0 + 2.0, dtype=np.float32)

    def _embeddings_from_pcm(self, pcm: NDArray[np.int16]) -> NDArray[np.float32]:
        spec = self._melspectrogram(pcm)
        windows: list[NDArray[np.float32]] = []
        for start in range(0, spec.shape[0], 8):
            window = spec[start : start + _MEL_WINDOW]
            if window.shape[0] == _MEL_WINDOW:
                windows.append(window)
        batch = np.expand_dims(np.asarray(windows, dtype=np.float32), axis=-1)
        embedding = self._embedding.run(None, {"input_1": batch})[0]
        return np.asarray(embedding, dtype=np.float32).reshape(-1, _FEATURE_DIM)

    def _streaming_melspectrogram(self, n_samples: int) -> None:
        pcm = np.asarray(list(self._raw_buffer)[-n_samples - 160 * 3 :], dtype=np.int16)
        self._melspectrogram_buffer = np.vstack((self._melspectrogram_buffer, self._melspectrogram(pcm)))
        if self._melspectrogram_buffer.shape[0] > 10 * 97:
            self._melspectrogram_buffer = self._melspectrogram_buffer[-(10 * 97) :, :]

    def _prepare_features(self, pcm: NDArray[np.int16]) -> int:
        """Accumulate PCM and return prepared sample count when a full 80 ms frame is ready."""
        x = pcm.reshape(-1)
        if self._raw_remainder.size:
            x = np.concatenate((self._raw_remainder, x))
            self._raw_remainder = np.empty(0, dtype=np.int16)

        if self._accumulated_samples + x.size >= CHUNK_SAMPLES:
            remainder = (self._accumulated_samples + x.size) % CHUNK_SAMPLES
            if remainder:
                even = x[:-remainder]
                self._raw_remainder = x[-remainder:].astype(np.int16, copy=False)
            else:
                even = x
            self._raw_buffer.extend(even.tolist())
            self._accumulated_samples += int(even.size)
        else:
            self._raw_buffer.extend(x.tolist())
            self._accumulated_samples += int(x.size)

        if self._accumulated_samples < CHUNK_SAMPLES or self._accumulated_samples % CHUNK_SAMPLES != 0:
            return self._accumulated_samples

        prepared = self._accumulated_samples
        self._streaming_melspectrogram(prepared)
        for i in np.arange(prepared // CHUNK_SAMPLES - 1, -1, -1):
            ndx = -8 * i
            end = ndx if ndx != 0 else len(self._melspectrogram_buffer)
            window = self._melspectrogram_buffer[-_MEL_WINDOW + end : end].astype(np.float32)
            if window.shape[0] != _MEL_WINDOW:
                continue
            embedding = self._embedding.run(None, {"input_1": window[None, :, :, None]})[0]
            self._feature_buffer = np.vstack(
                (self._feature_buffer, np.asarray(embedding, dtype=np.float32).reshape(-1, _FEATURE_DIM))
            )
        self._accumulated_samples = 0
        if self._feature_buffer.shape[0] > 120:
            self._feature_buffer = self._feature_buffer[-120:, :]
        return prepared

    def process(self, pcm: NDArray[np.int16], sample_rate: int) -> float:
        """Feed microphone PCM and return the latest wake score in [0, 1]."""
        if sample_rate != SAMPLE_RATE:
            raise ValueError(f"Wake-word detection requires {SAMPLE_RATE} Hz microphone audio")
        prepared = self._prepare_features(np.asarray(pcm, dtype=np.int16))
        if prepared < CHUNK_SAMPLES:
            return self.last_score
        features = self._feature_buffer[-_FEATURE_FRAMES:, :][None, :, :].astype(np.float32)
        score = float(np.asarray(self._wake.run(None, {self._wake_input_name: features})[0]).reshape(-1)[0])
        self.last_score = score
        return score

    def detected(self, score: float | None = None, *, now: float | None = None) -> bool:
        """Return True once when score clears the threshold outside the cooldown window."""
        value = self.last_score if score is None else score
        moment = time.monotonic() if now is None else now
        if value < self.threshold:
            return False
        if moment - self._last_detection_monotonic < self.cooldown_s:
            return False
        self._last_detection_monotonic = moment
        return True


def _microphone_frame_to_int16(frame: Any) -> NDArray[np.int16]:
    """Convert an SDK mic frame (float32 mono/stereo) to mono int16 PCM.

    ``ReachyMini.media.get_audio_sample`` returns float32 in roughly [-1, 1].
    Casting that array with ``dtype=np.int16`` truncates nearly every sample to
    0, so the wake model only ever sees silence. Match the conversation path:
    mono-mix if needed, then ``audio_to_int16``.
    """
    audio = np.asarray(frame)
    if audio.size == 0:
        return np.empty(0, dtype=np.int16)
    if audio.ndim == 2:
        # channels-last (N, C) is the SDK convention; fall back if transposed.
        if audio.shape[1] > audio.shape[0] and audio.shape[0] <= 8:
            audio = audio.T
        if audio.shape[1] > 1:
            audio = audio.mean(axis=1)
        else:
            audio = audio[:, 0]
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    return audio_to_int16(audio)


def listen_for_wake_word(
    robot: ReachyMini,
    *,
    detector: WakeWordDetector | None = None,
    stop_event: threading.Event | None = None,
    on_score: Callable[[float], None] | None = None,
    own_recording: bool = True,
) -> bool:
    """Block until the wake prompt is heard, or until ``stop_event`` is set.

    Returns True on detection, False when stopped without a match. When
    ``own_recording`` is True the helper starts and stops the robot mic around
    the listen loop so it can run before the conversation stream owns media.
    """
    wake_detector = detector or WakeWordDetector()
    wake_detector.reset()

    if own_recording:
        robot.media.start_recording()
    try:
        sample_rate = int(robot.media.get_input_audio_samplerate())
        logger.info(
            "Listening for wake word 'reechy' (threshold=%.2f, mic=%s Hz).",
            wake_detector.threshold,
            sample_rate,
        )
        last_score_log = 0.0
        peak_score = 0.0
        peak_rms = 0.0
        while stop_event is None or not stop_event.is_set():
            frame: Any = robot.media.get_audio_sample()
            if frame is None:
                time.sleep(_EMPTY_POLL_SLEEP_S)
                continue
            pcm = _microphone_frame_to_int16(frame)
            if pcm.size == 0:
                continue
            frame_rms = float(np.sqrt(np.mean(np.square(pcm.astype(np.float32))))) / 32768.0
            peak_rms = max(peak_rms, frame_rms)
            try:
                score = wake_detector.process(pcm, sample_rate)
            except ValueError as e:
                logger.error("Wake-word mic format error: %s", e)
                return False
            peak_score = max(peak_score, score)
            if on_score is not None:
                on_score(score)
            now = time.monotonic()
            if now - last_score_log >= _SCORE_LOG_INTERVAL_S:
                logger.info(
                    "Wake-word listening: score=%.3f peak=%.3f mic_rms=%.4f threshold=%.2f",
                    score,
                    peak_score,
                    peak_rms,
                    wake_detector.threshold,
                )
                last_score_log = now
                peak_score = 0.0
                peak_rms = 0.0
            if wake_detector.detected(score):
                logger.info("Wake word detected (score=%.3f); handing off to conversation.", score)
                return True
        logger.info("Wake-word listen stopped without a detection.")
        return False
    finally:
        if own_recording:
            try:
                robot.media.stop_recording()
            except Exception as e:
                logger.debug("Error stopping wake-word recording: %s", e)
