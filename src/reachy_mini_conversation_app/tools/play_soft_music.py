"""Play soft local lullaby / calming music on the robot speaker."""

from __future__ import annotations
import logging
from typing import Any
from pathlib import Path

from reachy_mini_conversation_app.audio import soft_music
from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies


logger = logging.getLogger(__name__)


class PlaySoftMusic(Tool):
    """Play a calm local music file for soothing / baby lullaby moments."""

    name = "play_soft_music"
    description = (
        "Play soft soothing local music / lullaby on Reachy's speaker (offline files only). "
        "Use whenever the user wants gentle, calm, baby, or lullaby music—including Chinese requests like "
        "「轻柔的音乐」「给宝宝放音乐」「摇篮曲」. "
        f"Omit 'track' for the first catalog entry; default volume is {soft_music.DEFAULT_SOFT_VOLUME}/100. "
        "Call stop_soft_music to stop. Prefer this over dances for quiet/baby moments. "
        "Do not claim you cannot play music—this tool is how you play it."
    )
    needs_response = True
    parameters_schema = {
        "type": "object",
        "properties": {
            "track": {
                "type": "string",
                "description": (
                    "Optional track stem or filename (for example gentle_sway or soft_chimes). "
                    "Omit to play the first catalog entry."
                ),
            },
            "volume": {
                "type": "integer",
                "minimum": 0,
                "maximum": 100,
                "description": f"Speaker level while playing. Defaults to {soft_music.DEFAULT_SOFT_VOLUME}.",
            },
        },
        "required": [],
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> dict[str, Any]:
        """Start local soft music on the robot speaker."""
        track = kwargs.get("track")
        track_name = str(track).strip() if isinstance(track, str) and track.strip() else None
        try:
            volume = soft_music.clamp_soft_volume(kwargs.get("volume"))
            result = soft_music.play_soft_music(
                deps.reachy_mini,
                track=track_name,
                volume=volume,
                instance_path=deps.instance_path,
            )
        except (FileNotFoundError, ValueError) as e:
            logger.warning("play_soft_music rejected: %s", e)
            return {
                "error": str(e),
                "available_tracks": soft_music.format_track_list(deps.instance_path),
            }
        except Exception as e:
            logger.error("play_soft_music failed: %s", e)
            return {"error": f"play_soft_music failed: {type(e).__name__}: {e}"}

        path = result.get("path")
        if isinstance(path, str):
            duration = soft_music.approx_duration_s(Path(path))
            if duration is not None:
                result["duration_s"] = round(duration, 1)
        logger.info(
            "Tool call: play_soft_music track=%s volume=%s",
            result.get("track"),
            result.get("volume"),
        )
        return result
