"""Tests for spoken Greetings-style gesture→emotion tool."""

from unittest.mock import MagicMock

import pytest

import reachy_mini_conversation_app.tools.react_to_gesture_intent as gesture_mod
from reachy_mini_conversation_app.tools.play_emotion import resolve_emotion_name
from reachy_mini_conversation_app.tools.react_to_gesture_intent import (
    GESTURE_INTENT_TO_EMOTION,
    ReactToGestureIntent,
)


@pytest.mark.parametrize(
    ("kind", "emotion"),
    list(GESTURE_INTENT_TO_EMOTION.items()),
)
def test_gesture_intent_map_matches_greetings(kind: str, emotion: str) -> None:
    """Keep the four Greetings MovementManager emotion IDs."""
    assert kind in ("open_palm", "thumb_up", "thumb_down", "pointing_up")
    assert emotion in {"enthusiastic1", "proud1", "sad2", "attentive2"}


@pytest.mark.parametrize(
    ("intent", "emotion"),
    [
        ("open_palm", "enthusiastic1"),
        ("thumb_up", "proud1"),
        ("thumb_down", "sad2"),
        ("pointing_up", "attentive2"),
    ],
)
def test_play_emotion_resolves_gesture_intents(intent: str, emotion: str) -> None:
    """play_emotion intents resolve to the same clips as react_to_gesture_intent."""
    available = ["enthusiastic1", "proud1", "sad2", "attentive2", "confused1"]
    assert resolve_emotion_name(intent, available) == emotion


@pytest.mark.asyncio
async def test_react_to_gesture_intent_queues_emotion(monkeypatch: pytest.MonkeyPatch) -> None:
    """Happy path queues EmotionQueueMove with the mapped recorded move."""
    library = MagicMock()
    library.list_moves.return_value = ["enthusiastic1", "proud1", "sad2", "attentive2"]
    queued: list[object] = []

    class FakeEmotionMove:
        def __init__(self, name: str, lib: object) -> None:
            self.name = name
            self.lib = lib

    monkeypatch.setattr(gesture_mod, "_EMOTION_AVAILABLE", True)
    monkeypatch.setattr(gesture_mod, "EmotionQueueMove", FakeEmotionMove)
    tool = ReactToGestureIntent()
    tool._library = library
    deps = MagicMock()
    deps.movement_manager.queue_move.side_effect = queued.append

    result = await tool(deps, kind="thumb_up")
    assert result == {"status": "queued", "kind": "thumb_up", "emotion": "proud1"}
    assert len(queued) == 1
    assert getattr(queued[0], "name") == "proud1"


@pytest.mark.asyncio
async def test_react_to_gesture_intent_rejects_unknown_kind() -> None:
    """Unknown kinds return an error without queueing."""
    tool = ReactToGestureIntent()
    deps = MagicMock()
    result = await tool(deps, kind="wave")
    assert "error" in result
    deps.movement_manager.queue_move.assert_not_called()
