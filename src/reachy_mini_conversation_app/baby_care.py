"""Baby care mode: follow, periodic photo, vision hazard check, loud local alert."""

from __future__ import annotations
import os
import json
import base64
import asyncio
import logging
import threading
from typing import Any, Protocol
from pathlib import Path
from datetime import datetime, timezone
from collections.abc import Callable, Awaitable


logger = logging.getLogger(__name__)

USB_BABY_CARE_DIR = Path("/mnt/usb_storage/media/baby-care")
DEFAULT_INTERVAL_S = 30.0
DEFAULT_WARN_VOLUME = 100
DEFAULT_WARN_TEXT = "警告！宝宝可能有危险，请立即查看！"
ALERT_FILENAME = "alert.jsonl"
VISION_URL_ENV = "BABY_CARE_VISION_URL"
VISION_MODEL_ENV = "BABY_CARE_VISION_MODEL"
VISION_TOKEN_ENV = "BABY_CARE_VISION_TOKEN"
WEBHOOK_URL_ENV = "BABY_CARE_WEBHOOK_URL"
# HF OpenAI-compatible router; override with BABY_CARE_VISION_URL / MODEL when needed.
DEFAULT_VISION_BASE_URL = "https://router.huggingface.co/v1"
DEFAULT_VISION_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"

AnnounceFn = Callable[[str], Awaitable[None]]
AnalyzerFn = Callable[[bytes], Awaitable[dict[str, Any]]]


class _MediaRobot(Protocol):
    media: Any


_STATE_LOCK = threading.Lock()
_active = False
_task: asyncio.Task[None] | None = None
_loop: asyncio.AbstractEventLoop | None = None


def resolve_storage_dir(instance_path: str | Path | None = None) -> Path:
    """Prefer USB baby-care media; otherwise use the instance data directory."""
    usb = USB_BABY_CARE_DIR
    try:
        if usb.parent.is_dir():
            usb.mkdir(parents=True, exist_ok=True)
            probe = usb / ".write_probe"
            probe.write_text("ok", encoding="utf-8")
            probe.unlink(missing_ok=True)
            return usb
    except OSError as e:
        logger.debug("USB baby-care dir unavailable (%s); using local fallback", e)

    if instance_path is not None:
        local = Path(instance_path).expanduser() / "media" / "baby-care"
    else:
        local = Path.cwd() / "media" / "baby-care"
    local.mkdir(parents=True, exist_ok=True)
    return local


def append_alert(storage_dir: Path, record: dict[str, Any]) -> Path:
    """Append one JSON alert line and return the alert.jsonl path."""
    storage_dir.mkdir(parents=True, exist_ok=True)
    path = storage_dir / ALERT_FILENAME
    line = json.dumps(record, ensure_ascii=False)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
    return path


def capture_jpeg(robot: _MediaRobot) -> bytes:
    """Grab one JPEG frame from the SDK camera pipeline."""
    jpeg_bytes = robot.media.get_frame_jpeg()
    if jpeg_bytes is None:
        raise RuntimeError("No frame available from camera")
    if not isinstance(jpeg_bytes, (bytes, bytearray)):
        raise RuntimeError(f"Unexpected frame type: {type(jpeg_bytes)!r}")
    return bytes(jpeg_bytes)


def save_jpeg(storage_dir: Path, jpeg_bytes: bytes, *, when: datetime | None = None) -> Path:
    """Write a timestamped JPEG under the care media directory."""
    storage_dir.mkdir(parents=True, exist_ok=True)
    stamp = (when or datetime.now(timezone.utc)).strftime("%Y%m%dT%H%M%SZ")
    path = storage_dir / f"baby-care-{stamp}.jpg"
    path.write_bytes(jpeg_bytes)
    return path


def _vision_config() -> tuple[str, str, str | None]:
    base_url = (os.getenv(VISION_URL_ENV) or DEFAULT_VISION_BASE_URL).strip().rstrip("/")
    model = (os.getenv(VISION_MODEL_ENV) or DEFAULT_VISION_MODEL).strip()
    token = (os.getenv(VISION_TOKEN_ENV) or os.getenv("HF_TOKEN") or "").strip() or None
    return base_url, model, token


def parse_hazard_payload(raw: str) -> dict[str, Any]:
    """Parse model JSON into ``{safe, hazards, summary}``; tolerate fenced text."""
    text = (raw or "").strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        text = text[start : end + 1]
    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("hazard payload must be a JSON object")
    hazards_raw = data.get("hazards", [])
    if hazards_raw is None:
        hazards: list[str] = []
    elif isinstance(hazards_raw, list):
        hazards = [str(item) for item in hazards_raw]
    else:
        hazards = [str(hazards_raw)]
    safe = data.get("safe")
    if safe is None:
        safe = len(hazards) == 0
    summary = data.get("summary")
    return {
        "safe": bool(safe),
        "hazards": hazards,
        "summary": str(summary) if summary is not None else "",
    }


async def analyze_hazards_openai(jpeg_bytes: bytes) -> dict[str, Any]:
    """Ask a vision chat model for a structured baby-safety verdict."""
    from openai import AsyncOpenAI

    base_url, model, token = _vision_config()
    if not token:
        raise RuntimeError(f"No vision token: set {VISION_TOKEN_ENV} or HF_TOKEN for baby-care hazard checks")
    client = AsyncOpenAI(base_url=base_url, api_key=token)
    b64 = base64.b64encode(jpeg_bytes).decode("ascii")
    prompt = (
        "You are a baby safety monitor. Inspect this photo from a home baby-care camera. "
        "Reply with ONLY a JSON object: "
        '{"safe": boolean, "hazards": [string], "summary": string}. '
        "Mark safe=false for sharp objects near the baby, blanket/cloth over the face, "
        "choking hazards, unsafe sleep position, falling risk, or other clear danger. "
        "If no baby is visible, set safe=true unless an obvious hazard is present, and say so in summary."
    )
    response = await client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                    },
                ],
            }
        ],
        temperature=0,
        max_tokens=400,
    )
    content = ""
    if response.choices:
        message = response.choices[0].message
        content = (message.content or "") if message is not None else ""
    return parse_hazard_payload(content)


def _read_speaker_volume(robot: Any) -> int | None:
    from reachy_mini_conversation_app.daemon_api import DaemonApiError, daemon_request

    try:
        response = daemon_request(robot, "/api/volume/current")
    except DaemonApiError as e:
        logger.debug("Could not read speaker volume: %s", e)
        return None
    if isinstance(response, dict) and isinstance(response.get("volume"), (int, float)):
        return int(response["volume"])
    return None


def _set_speaker_volume(robot: Any, level: int) -> None:
    from reachy_mini_conversation_app.daemon_api import daemon_request

    daemon_request(robot, "/api/volume/set", method="POST", payload={"volume": int(level)}, timeout_s=5.0)


async def loud_warn(
    robot: Any,
    announce: AnnounceFn | None,
    *,
    text: str = DEFAULT_WARN_TEXT,
    volume: int = DEFAULT_WARN_VOLUME,
) -> None:
    """Raise speaker volume and voice a danger warning to nearby adults."""
    previous = _read_speaker_volume(robot)
    try:
        _set_speaker_volume(robot, volume)
    except Exception as e:
        logger.warning("Could not raise volume for baby-care warning: %s", e)
        previous = None

    if announce is not None:
        try:
            await announce("Speak this warning loudly and urgently to nearby adults, in Chinese if possible: " + text)
        except Exception as e:
            logger.error("Baby-care announce failed: %s", e)
    else:
        logger.warning("Baby-care danger detected but no announce callback is wired")

    if previous is not None:
        try:
            _set_speaker_volume(robot, previous)
        except Exception as e:
            logger.warning("Could not restore volume after baby-care warning: %s", e)


async def _post_webhook(record: dict[str, Any]) -> None:
    url = (os.getenv(WEBHOOK_URL_ENV) or "").strip()
    if not url:
        return
    import httpx

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            await client.post(url, json=record)
    except Exception as e:
        logger.warning("Baby-care webhook failed: %s", e)


def is_active() -> bool:
    """Return whether baby care mode is currently running."""
    with _STATE_LOCK:
        return _active


async def _care_loop(
    robot: Any,
    movement_manager: Any,
    *,
    instance_path: str | Path | None,
    camera_enabled: bool,
    announce: AnnounceFn | None,
    analyzer: AnalyzerFn,
    interval_s: float,
) -> None:
    global _active, _task

    storage_dir = resolve_storage_dir(instance_path)
    logger.info("Baby care loop started; media=%s interval=%.1fs", storage_dir, interval_s)
    try:
        while True:
            with _STATE_LOCK:
                if not _active:
                    break
            try:
                if not camera_enabled:
                    logger.warning("Baby care: camera disabled; skipping capture")
                else:
                    jpeg = await asyncio.to_thread(capture_jpeg, robot)
                    when = datetime.now(timezone.utc)
                    image_path = await asyncio.to_thread(save_jpeg, storage_dir, jpeg, when=when)
                    try:
                        verdict = await analyzer(jpeg)
                    except Exception as e:
                        logger.error("Baby-care vision analysis failed: %s", e)
                        verdict = {
                            "safe": True,
                            "hazards": [],
                            "summary": f"analysis_error: {type(e).__name__}: {e}",
                        }
                    logger.info(
                        "Baby-care check image=%s safe=%s hazards=%s",
                        image_path,
                        verdict.get("safe"),
                        verdict.get("hazards"),
                    )
                    if not bool(verdict.get("safe", True)):
                        record = {
                            "timestamp": when.isoformat(),
                            "image_path": str(image_path),
                            "safe": False,
                            "hazards": verdict.get("hazards") or [],
                            "summary": verdict.get("summary") or "",
                        }
                        await asyncio.to_thread(append_alert, storage_dir, record)
                        await loud_warn(robot, announce)
                        await _post_webhook(record)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.error("Baby-care loop iteration failed: %s", e)

            try:
                await asyncio.sleep(interval_s)
            except asyncio.CancelledError:
                raise
            with _STATE_LOCK:
                if not _active:
                    break
    finally:
        try:
            movement_manager.set_head_tracking(False)
        except Exception as e:
            logger.debug("Could not disable head tracking after baby care: %s", e)
        with _STATE_LOCK:
            _active = False
            _task = None
        logger.info("Baby care loop stopped")


def start_baby_care(
    robot: Any,
    movement_manager: Any,
    *,
    instance_path: str | Path | None = None,
    camera_enabled: bool = True,
    announce: AnnounceFn | None = None,
    analyzer: AnalyzerFn | None = None,
    interval_s: float = DEFAULT_INTERVAL_S,
) -> dict[str, Any]:
    """Enable head tracking and start the periodic care loop on the running event loop."""
    global _active, _task, _loop

    if interval_s <= 0:
        raise ValueError("interval_s must be positive")

    with _STATE_LOCK:
        if _active and _task is not None and not _task.done():
            return {
                "status": "already_active",
                "storage_dir": str(resolve_storage_dir(instance_path)),
                "interval_s": interval_s,
            }
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError as e:
            raise RuntimeError("start_baby_care requires a running asyncio event loop") from e

        movement_manager.set_head_tracking(True)
        _active = True
        _loop = loop
        _task = loop.create_task(
            _care_loop(
                robot,
                movement_manager,
                instance_path=instance_path,
                camera_enabled=camera_enabled,
                announce=announce,
                analyzer=analyzer or analyze_hazards_openai,
                interval_s=interval_s,
            ),
            name="baby-care-loop",
        )

    storage = resolve_storage_dir(instance_path)
    return {
        "status": "started",
        "storage_dir": str(storage),
        "interval_s": interval_s,
        "following": True,
    }


def stop_baby_care(movement_manager: Any | None = None) -> dict[str, Any]:
    """Stop the care loop and disable head tracking."""
    global _active, _task

    with _STATE_LOCK:
        was_active = _active
        _active = False
        task = _task
        _task = None

    if movement_manager is not None:
        try:
            movement_manager.set_head_tracking(False)
        except Exception as e:
            logger.debug("Could not disable head tracking on stop: %s", e)

    if task is not None and not task.done():
        task.cancel()

    return {"status": "stopped", "was_active": was_active}
