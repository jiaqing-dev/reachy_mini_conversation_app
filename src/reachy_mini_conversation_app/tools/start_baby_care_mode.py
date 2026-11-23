"""Start baby care mode: follow, periodic photo, vision hazard alerts."""

from __future__ import annotations
import logging
from typing import Any

from reachy_mini_conversation_app import baby_care
from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies


logger = logging.getLogger(__name__)


class StartBabyCareMode(Tool):
    """Enter baby care mode with follow + timed hazard checks."""

    name = "start_baby_care_mode"
    description = (
        "Start baby care mode on Reachy Mini. "
        "Use when the user asks to enter baby care / babysitting / infant watch mode—including Chinese "
        "「进入宝宝看护模式」「宝宝看护」「开始看护宝宝」. "
        "Enables head tracking to follow the baby, then every ~30s captures a photo, "
        "asks a vision model for hazards (sharp objects, blanket over face, etc.), "
        "and on danger loudly warns adults plus appends a local alert with image path and timestamp. "
        "Call stop_baby_care_mode to exit. Do not say you lack a baby-care feature—this tool starts it."
    )
    needs_response = True
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
        """Start the baby-care background loop."""
        logger.info("Tool call: start_baby_care_mode")
        try:
            return baby_care.start_baby_care(
                deps.reachy_mini,
                deps.movement_manager,
                instance_path=deps.instance_path,
                camera_enabled=deps.camera_enabled,
                announce=deps.announce,
            )
        except Exception as e:
            logger.error("start_baby_care_mode failed: %s", e)
            return {"error": f"start_baby_care_mode failed: {type(e).__name__}: {e}"}
