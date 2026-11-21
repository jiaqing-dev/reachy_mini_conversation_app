"""Stop soft local music without tearing down conversation TTS playback."""

from __future__ import annotations
import logging
from typing import Any

from reachy_mini_conversation_app.audio import soft_music
from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies


logger = logging.getLogger(__name__)


class StopSoftMusic(Tool):
    """Stop soft music started by play_soft_music."""

    name = "stop_soft_music"
    description = (
        "Stop soft/lullaby music started with play_soft_music (including when the user says stop the music / "
        "「停下音乐」「关闭音乐」) and restore the previous speaker volume when possible."
    )
    needs_response = False
    parameters_schema = {
        "type": "object",
        "properties": {
            "dummy": {
                "type": "boolean",
                "description": "Unused; set true if the schema requires a property.",
            },
        },
        "required": [],
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> dict[str, Any]:
        """Stop soft music playback on the robot speaker."""
        logger.info("Tool call: stop_soft_music")
        try:
            return soft_music.stop_soft_music(deps.reachy_mini)
        except Exception as e:
            logger.error("stop_soft_music failed: %s", e)
            return {"error": f"stop_soft_music failed: {type(e).__name__}: {e}"}
