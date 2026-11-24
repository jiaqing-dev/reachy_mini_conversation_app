"""Stop baby care mode and disable head tracking."""

from __future__ import annotations
import logging
from typing import Any

from reachy_mini_conversation_app import baby_care
from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies


logger = logging.getLogger(__name__)


class StopBabyCareMode(Tool):
    """Exit baby care mode."""

    name = "stop_baby_care_mode"
    description = (
        "Stop baby care mode started with start_baby_care_mode "
        "(including when the user says 「退出宝宝看护模式」「停止看护」「结束看护」). "
        "Stops the photo timer and head following."
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
        """Stop the baby-care background loop."""
        logger.info("Tool call: stop_baby_care_mode")
        try:
            return baby_care.stop_baby_care(deps.movement_manager)
        except Exception as e:
            logger.error("stop_baby_care_mode failed: %s", e)
            return {"error": f"stop_baby_care_mode failed: {type(e).__name__}: {e}"}
