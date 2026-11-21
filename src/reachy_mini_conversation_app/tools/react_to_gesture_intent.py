"""Queue Greetings-aligned emotion clips from spoken gesture intents (no MediaPipe)."""

import logging
from typing import TYPE_CHECKING, Any

from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies


if TYPE_CHECKING:
    from reachy_mini.motion.recorded_move import RecordedMoves


logger = logging.getLogger(__name__)

# Same recorded-move IDs as pollen-robotics/reachy_mini_greetings MovementManager.
GESTURE_INTENT_TO_EMOTION: dict[str, str] = {
    "open_palm": "enthusiastic1",
    "thumb_up": "proud1",
    "thumb_down": "sad2",
    "pointing_up": "attentive2",
}

GESTURE_INTENT_NAMES: tuple[str, ...] = tuple(GESTURE_INTENT_TO_EMOTION)

try:
    from reachy_mini.motion.recorded_move import RecordedMoves
    from reachy_mini_conversation_app.dance_emotion_moves import EmotionQueueMove

    _EMOTION_AVAILABLE = True
except Exception as e:
    logger.warning("Gesture-intent emotions unavailable: %s", e)
    _EMOTION_AVAILABLE = False


class ReactToGestureIntent(Tool):
    """Play the emotion clip that the official Greetings app uses for a hand gesture."""

    name = "react_to_gesture_intent"
    description = (
        "React to a spoken hand-gesture intent the way the Reachy Mini Greetings app does "
        "(camera-free: voice/LLM only). Call this when the user asks for a greeting palm wave, "
        "thumbs up/down, or pointing up—including Chinese like 「打招呼」「张开手」「竖拇指」「点赞」"
        "「拇指向下」「不开心」「向上指」「注意听」. "
        "Maps open_palm→enthusiastic1, thumb_up→proud1, thumb_down→sad2, pointing_up→attentive2. "
        "Never say you cannot do gestures for these requests—queue this tool. "
        "True camera gesture recognition still requires the separate Greetings app."
    )
    needs_response = False
    parameters_schema = {
        "type": "object",
        "properties": {
            "kind": {
                "type": "string",
                "enum": list(GESTURE_INTENT_NAMES),
                "description": (
                    "Gesture intent: open_palm (打招呼/张开手), thumb_up (竖拇指/点赞), "
                    "thumb_down (拇指向下/不开心), pointing_up (向上指/注意听)."
                ),
            },
        },
        "required": ["kind"],
    }
    _library: RecordedMoves | None = None

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> dict[str, Any]:
        """Queue the Greetings-mapped emotion for a spoken gesture intent."""
        if not _EMOTION_AVAILABLE:
            return {"error": "Emotion system not available"}

        raw_kind = kwargs.get("kind")
        kind = str(raw_kind or "").strip().lower()
        emotion_name = GESTURE_INTENT_TO_EMOTION.get(kind)
        if emotion_name is None:
            logger.warning("react_to_gesture_intent: unknown kind=%r", raw_kind)
            return {
                "error": f"Unknown gesture kind {raw_kind!r}",
                "available": list(GESTURE_INTENT_NAMES),
            }

        logger.info("Tool call: react_to_gesture_intent kind=%s emotion=%s", kind, emotion_name)
        try:
            if self._library is None:
                self._library = RecordedMoves("pollen-robotics/reachy-mini-emotions-library")
            library = self._library
            available = set(library.list_moves())
            if emotion_name not in available:
                return {"error": f"Emotion {emotion_name!r} missing from library"}

            deps.movement_manager.queue_move(EmotionQueueMove(emotion_name, library))
            return {"status": "queued", "kind": kind, "emotion": emotion_name}
        except Exception as e:
            logger.exception("react_to_gesture_intent failed")
            return {"error": f"react_to_gesture_intent failed: {type(e).__name__}: {e}"}
