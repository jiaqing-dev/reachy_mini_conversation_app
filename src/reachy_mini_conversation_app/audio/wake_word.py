"""OpenWakeWord-style ONNX wake-word detection for the robot microphone.

Shared melspectrogram + speech-embedding preprocessors (openWakeWord, Apache 2.0)
feed one or more wake classifiers. The bundled **reechy** classifier comes from
andyjmorgan/reachy-wake-word (CC BY-NC-SA 4.0). Additional words need their own
ONNX classifiers (separate models score better than one multi-label head).
Inference uses the same 80 ms / 1280-sample framing as that project's test utility.
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
from dataclasses import dataclass
from collections.abc import Mapping, Callable, Sequence

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
DEFAULT_WAKE_WORD_IDS: tuple[str, ...] = ("reechy",)
_FEATURE_FRAMES = 16
_FEATURE_DIM = 96
_MEL_WINDOW = 76
_EMPTY_POLL_SLEEP_S = 0.005
_SCORE_LOG_INTERVAL_S = 2.0
_MODEL_CACHE_NAMESPACE = "reachy_mini_conversation_app"
_MODEL_CACHE_SUBDIR = "wake-word"
_MODEL_CACHE_REVISION = "wake-words-v1"

_FEATURE_MODEL_FILES: dict[str, tuple[str, str]] = {
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
}


@dataclass(frozen=True)
class WakeWordSpec:
    """One wake prompt: display label plus optional bundled classifier download."""

    word_id: str
    display_name: str
    filename: str | None = None
    url: str | None = None
    sha256: str | None = None

    @property
    def is_bundled(self) -> bool:
        """True when this word ships with a downloadable classifier ONNX."""
        return bool(self.filename and self.url and self.sha256)


# Separate classifiers per word (training showed multi-label heads dilute accuracy).
_BUILTIN_WAKE_WORDS: dict[str, WakeWordSpec] = {
    "reechy": WakeWordSpec(
        word_id="reechy",
        display_name="reechy",
        filename="reechy-spk150-steps100k-acc98.15-rec97.50.onnx",
        url=(
            "https://raw.githubusercontent.com/andyjmorgan/reachy-wake-word/"
            "main/models/reechy-spk150-steps100k-acc98.15-rec97.50.onnx"
        ),
        sha256="bcd551a49315e443b11877ff52e11f30e42ecdf024b6c051dcaad80faee1e14f",
    ),
    # No public ready-made ONNX for 小白; enable via WAKE_WORD_MODEL_XIAOBAI after training.
    "xiaobai": WakeWordSpec(word_id="xiaobai", display_name="小白"),
}

_WAKE_WORD_ALIASES: dict[str, str] = {
    "reechy": "reechy",
    "小白": "xiaobai",
    "xiaobai": "xiaobai",
    "xiao_bai": "xiaobai",
    "xiao-bai": "xiaobai",
}


def normalize_wake_word_id(name: str) -> str:
    """Map a user-facing wake-word token to a stable ascii id."""
    token = name.strip()
    if not token:
        raise ValueError("Wake-word id must be non-empty")
    if token in _WAKE_WORD_ALIASES:
        return _WAKE_WORD_ALIASES[token]
    lowered = token.lower().replace(" ", "_")
    if lowered in _WAKE_WORD_ALIASES:
        return _WAKE_WORD_ALIASES[lowered]
    compact = lowered.replace("-", "_")
    if compact in _WAKE_WORD_ALIASES:
        return _WAKE_WORD_ALIASES[compact]
    if not all(ch.isalnum() or ch in {"_", "-"} for ch in lowered):
        raise ValueError(
            f"Unknown wake-word {name!r}. Use a built-in id ({', '.join(sorted(_BUILTIN_WAKE_WORDS))}) "
            "or an ascii custom id with WAKE_WORD_MODEL_<ID>=/path/to/model.onnx"
        )
    return compact


def parse_wake_word_ids(raw: str | Sequence[str] | None) -> tuple[str, ...]:
    """Parse a comma-separated list or sequence into normalized unique word ids."""
    if raw is None:
        return DEFAULT_WAKE_WORD_IDS
    if isinstance(raw, str):
        parts = [part.strip() for part in raw.split(",")]
    else:
        parts = [str(part).strip() for part in raw]
    ids: list[str] = []
    for part in parts:
        if not part:
            continue
        word_id = normalize_wake_word_id(part)
        if word_id not in ids:
            ids.append(word_id)
    if not ids:
        raise ValueError("Wake-word list is empty after parsing")
    return tuple(ids)


def wake_word_display_names(word_ids: Sequence[str]) -> str:
    """Human-readable list for logs (for example: reechy, 小白)."""
    labels: list[str] = []
    for word_id in word_ids:
        spec = _BUILTIN_WAKE_WORDS.get(word_id)
        labels.append(spec.display_name if spec is not None else word_id)
    return ", ".join(labels)


def _wake_word_model_env_name(word_id: str) -> str:
    return f"WAKE_WORD_MODEL_{word_id.upper().replace('-', '_')}"


def _spec_for_word(word_id: str) -> WakeWordSpec:
    return _BUILTIN_WAKE_WORDS.get(word_id) or WakeWordSpec(word_id=word_id, display_name=word_id)


def _download_file(url: str, destination: Path, *, expected_sha256: str | None) -> Path:
    """Download ``url`` into ``destination``, optionally verifying SHA-256."""
    if destination.is_file():
        if expected_sha256 is None or hashlib.sha256(destination.read_bytes()).hexdigest() == expected_sha256:
            return destination
        logger.warning("Cached wake-word file failed checksum verification: %s", destination)
    logger.info("Downloading wake-word model file: %s", destination.name)
    with httpx.Client(follow_redirects=True, timeout=60.0) as client:
        response = client.get(url)
        response.raise_for_status()
        content = response.content
    if expected_sha256 is not None and hashlib.sha256(content).hexdigest() != expected_sha256:
        raise ValueError(f"Downloaded wake-word file failed checksum verification: {destination.name}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    download = NamedTemporaryFile(dir=destination.parent, delete=False)
    download_path = Path(download.name)
    try:
        with download:
            download.write(content)
        download_path.replace(destination)
    finally:
        download_path.unlink(missing_ok=True)
    return destination


def _resolve_classifier_path(
    word_id: str,
    *,
    model_directory: Path,
    model_uri: str | None,
) -> Path:
    """Resolve a classifier ONNX path from an override URI or a bundled download."""
    spec = _spec_for_word(word_id)
    if model_uri:
        uri = model_uri.strip()
        local = Path(uri).expanduser()
        if local.is_file():
            return local.resolve()
        if uri.startswith(("http://", "https://")):
            filename = spec.filename or f"{word_id}.onnx"
            return _download_file(uri, model_directory / filename, expected_sha256=None)
        raise FileNotFoundError(
            f"Wake-word model for {spec.display_name!r} not found at {uri!r}. "
            f"Set {_wake_word_model_env_name(word_id)} to an existing .onnx path or https URL."
        )
    if spec.is_bundled:
        assert spec.filename is not None and spec.url is not None
        return _download_file(spec.url, model_directory / spec.filename, expected_sha256=spec.sha256)
    raise FileNotFoundError(
        f"Wake word {spec.display_name!r} ({word_id}) has no bundled ONNX classifier. "
        "Train a custom openWakeWord ONNX model (see README → Wake word), then set "
        f"{_wake_word_model_env_name(word_id)}=/path/to/{word_id}.onnx. "
        "Detection is not faked without a real model."
    )


def _ort_session(path: Path) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    return ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])


class WakeWordDetector:
    """Score streaming 16 kHz mono PCM against one or more wake classifiers."""

    def __init__(
        self,
        *,
        threshold: float = DEFAULT_THRESHOLD,
        cooldown_s: float = DEFAULT_COOLDOWN_S,
        words: Sequence[str] | None = None,
        model_uris: Mapping[str, str] | None = None,
    ) -> None:
        """Load shared feature models plus one classifier per requested wake word."""
        if not 0.0 < threshold <= 1.0:
            raise ValueError(f"Wake-word threshold must be in (0, 1], got {threshold}")
        if cooldown_s < 0.0:
            raise ValueError(f"Wake-word cooldown must be >= 0, got {cooldown_s}")

        word_ids = parse_wake_word_ids(words)
        model_directory = cached_assets_path(_MODEL_CACHE_NAMESPACE, _MODEL_CACHE_SUBDIR, _MODEL_CACHE_REVISION)
        for filename, (url, checksum) in _FEATURE_MODEL_FILES.items():
            _download_file(url, model_directory / filename, expected_sha256=checksum)

        self._melspec = _ort_session(model_directory / "melspectrogram.onnx")
        self._embedding = _ort_session(model_directory / "embedding_model.onnx")
        self._wake_sessions: dict[str, ort.InferenceSession] = {}
        self._wake_input_names: dict[str, str] = {}
        self._display_names: dict[str, str] = {}
        overrides = {normalize_wake_word_id(key): value for key, value in (model_uris or {}).items()}
        for word_id in word_ids:
            spec = _spec_for_word(word_id)
            path = _resolve_classifier_path(word_id, model_directory=model_directory, model_uri=overrides.get(word_id))
            session = _ort_session(path)
            self._wake_sessions[word_id] = session
            self._wake_input_names[word_id] = session.get_inputs()[0].name
            self._display_names[word_id] = spec.display_name

        self.word_ids = word_ids
        self.threshold = threshold
        self.cooldown_s = cooldown_s
        self.last_scores: dict[str, float] = dict.fromkeys(word_ids, 0.0)
        self.last_score = 0.0
        self.last_matched_word: str | None = None
        self._last_detection_monotonic = 0.0
        self._raw_buffer: deque[int] = deque(maxlen=SAMPLE_RATE * 10)
        self._raw_remainder = np.empty(0, dtype=np.int16)
        self._accumulated_samples = 0
        self._melspectrogram_buffer = np.ones((_MEL_WINDOW, 32), dtype=np.float32)
        self._feature_buffer = self._bootstrap_features()

    def reset(self) -> None:
        """Clear streaming state between listen sessions."""
        self.last_scores = dict.fromkeys(self.word_ids, 0.0)
        self.last_score = 0.0
        self.last_matched_word = None
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

    def _score_word(self, word_id: str, *, frames_back: int = 0) -> float:
        if frames_back == 0:
            window = self._feature_buffer[-_FEATURE_FRAMES:, :]
        else:
            window = self._feature_buffer[-_FEATURE_FRAMES - frames_back : -frames_back, :]
        if window.shape[0] != _FEATURE_FRAMES:
            return self.last_scores[word_id]
        features = window[None, :, :].astype(np.float32)
        session = self._wake_sessions[word_id]
        input_name = self._wake_input_names[word_id]
        return float(np.asarray(session.run(None, {input_name: features})[0]).reshape(-1)[0])

    def process(self, pcm: NDArray[np.int16], sample_rate: int) -> float:
        """Feed microphone PCM and return the highest wake score in [0, 1]."""
        if sample_rate != SAMPLE_RATE:
            raise ValueError(f"Wake-word detection requires {SAMPLE_RATE} Hz microphone audio")
        prepared = self._prepare_features(np.asarray(pcm, dtype=np.int16))
        if prepared < CHUNK_SAMPLES:
            return self.last_score
        n_frames = prepared // CHUNK_SAMPLES
        best_score = 0.0
        best_word: str | None = None
        for word_id in self.word_ids:
            scores = [self._score_word(word_id, frames_back=i) for i in range(n_frames)]
            score = max(scores)
            self.last_scores[word_id] = score
            if score >= best_score:
                best_score = score
                best_word = word_id
        self.last_score = best_score
        self.last_matched_word = best_word
        return best_score

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

    def display_name(self, word_id: str) -> str:
        """Return the human-facing label for a configured wake-word id."""
        return self._display_names.get(word_id, word_id)

    def format_scores(self) -> str:
        """Compact per-word score string for periodic logs."""
        parts = [f"{self.display_name(word_id)}={self.last_scores[word_id]:.4f}" for word_id in self.word_ids]
        return " ".join(parts)


def _microphone_frame_to_int16(frame: Any) -> NDArray[np.int16]:
    """Convert an SDK mic frame (float32 mono/stereo) to mono int16 PCM.

    ``ReachyMini.media.get_audio_sample`` returns float32 in roughly [-1, 1].
    Casting that array with ``dtype=np.int16`` truncates nearly every sample to
    0, so the wake model only ever sees silence. Match the conversation path:
    first channel if stereo, then ``audio_to_int16``.
    """
    audio = np.asarray(frame)
    if audio.size == 0:
        return np.empty(0, dtype=np.int16)
    if audio.ndim == 2:
        # channels-last (N, C) is the SDK convention; fall back if transposed.
        if audio.shape[1] > audio.shape[0] and audio.shape[0] <= 8:
            audio = audio.T
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
    """Block until a configured wake prompt is heard, or until ``stop_event`` is set.

    Returns True on detection, False when stopped without a match. When
    ``own_recording`` is True the helper starts and stops the robot mic around
    the listen loop so it can run before the conversation stream owns media.

    Mic capture runs on a dedicated thread (same pattern as ``LocalStream``).
    The GStreamer appsink uses ``drop=True``; running ONNX on the polling
    thread stalls pulls long enough that speech frames are discarded and the
    wake model only ever sees fragments (scores stuck near 0 while RMS still
    spikes).
    """
    wake_detector = detector or WakeWordDetector()
    wake_detector.reset()

    if own_recording:
        robot.media.start_recording()
    # Give the LOCAL/GStreamer record pipeline a moment before the first pull.
    time.sleep(0.3)
    try:
        sample_rate = int(robot.media.get_input_audio_samplerate())
        logger.info(
            "Listening for wake word(s) %s (threshold=%.2f, mic=%s Hz).",
            wake_word_display_names(wake_detector.word_ids),
            wake_detector.threshold,
            sample_rate,
        )
        frames: deque[NDArray[np.int16]] = deque()
        frames_lock = threading.Lock()
        stop_reading = threading.Event()

        def read_microphone() -> None:
            while not stop_reading.is_set() and (stop_event is None or not stop_event.is_set()):
                frame: Any = robot.media.get_audio_sample()
                if frame is None:
                    time.sleep(_EMPTY_POLL_SLEEP_S)
                    continue
                pcm = _microphone_frame_to_int16(frame)
                if pcm.size == 0:
                    continue
                with frames_lock:
                    frames.append(pcm)

        reader = threading.Thread(target=read_microphone, name="wake-word-mic", daemon=True)
        reader.start()
        last_score_log = 0.0
        peak_score = 0.0
        peak_rms = 0.0
        try:
            while stop_event is None or not stop_event.is_set():
                with frames_lock:
                    pending = list(frames)
                    frames.clear()
                if not pending:
                    time.sleep(_EMPTY_POLL_SLEEP_S)
                    continue
                pcm = pending[0] if len(pending) == 1 else np.concatenate(pending)
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
                        "Wake-word listening: %s peak=%.4f mic_rms=%.4f threshold=%.2f",
                        wake_detector.format_scores(),
                        peak_score,
                        peak_rms,
                        wake_detector.threshold,
                    )
                    last_score_log = now
                    peak_score = 0.0
                    peak_rms = 0.0
                if wake_detector.detected(score):
                    matched = wake_detector.last_matched_word or wake_detector.word_ids[0]
                    logger.info(
                        "Wake word detected (%s, score=%.3f); handing off to conversation.",
                        wake_detector.display_name(matched),
                        score,
                    )
                    return True
        finally:
            stop_reading.set()
            reader.join(timeout=2.0)
        logger.info("Wake-word listen stopped without a detection.")
        return False
    finally:
        if own_recording:
            try:
                robot.media.stop_recording()
            except Exception as e:
                logger.debug("Error stopping wake-word recording: %s", e)
