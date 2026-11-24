"""Helpers for app startup and shutdown lifecycle behavior."""

import asyncio
import logging
from pathlib import Path

import numpy as np
import numpy.typing as npt

from reachy_mini import ReachyMini
from reachy_mini.reachy_mini import SLEEP_HEAD_POSE
from reachy_mini.utils.interpolation import distance_between_poses
from reachy_mini_conversation_app.config import config, set_custom_profile
from reachy_mini_conversation_app.daemon_api import DaemonApiError, daemon_request
from reachy_mini_conversation_app.tool_spaces import migrate_bundled_tool_spaces
from reachy_mini_conversation_app.profile_store import DEFAULT_PROFILE_NAME, migrate_legacy_profiles
from reachy_mini_conversation_app.profile_toolsets import (
    enable_profile_tools,
    read_profile_default_tool_names,
)
from reachy_mini_conversation_app.tools.core_tools import ToolDependencies, initialize_tools
from reachy_mini_conversation_app.tools.go_to_sleep import GoToSleep


_STOP_CURRENT_APP_PATH = "/api/apps/stop-current-app"
_STOP_CURRENT_APP_TIMEOUT_S = 2.0
_SLEEP_HEAD_TRANSLATION_TOLERANCE_M = 0.05
_SLEEP_HEAD_ROTATION_TOLERANCE_RAD = 0.35
_SOFT_MUSIC_TOOLS = ("play_soft_music", "stop_soft_music")
_GESTURE_INTENT_TOOLS = ("react_to_gesture_intent",)
_BABY_CARE_TOOLS = ("start_baby_care_mode", "stop_baby_care_mode")


def _ensure_authored_tools_enabled(
    instance_path: str | Path | None,
    logger: logging.Logger,
    tool_names: tuple[str, ...],
    *,
    label: str,
) -> list[str]:
    """Merge newly authored default tools into a stale instance Tool-access override."""
    profile = config.REACHY_MINI_CUSTOM_PROFILE or DEFAULT_PROFILE_NAME
    try:
        authored = set(read_profile_default_tool_names(profile))
    except Exception as exc:
        logger.debug("Could not read authored tools for %s seed (%s)", label, exc)
        authored = set()
    seed = [name for name in tool_names if name in authored]
    if not seed:
        return []
    try:
        added = enable_profile_tools(profile, seed, instance_path)
    except Exception as exc:
        logger.warning("Failed to ensure %s tools for profile %r: %s", label, profile, exc)
        return []
    if added:
        logger.info("Enabled %s tools for profile %r: %s", label, profile, ", ".join(added))
    return added


def ensure_soft_music_tools_enabled(instance_path: str | Path | None, logger: logging.Logger) -> list[str]:
    """Merge soft-music tools into a stale instance tool override when authored defaults include them."""
    return _ensure_authored_tools_enabled(instance_path, logger, _SOFT_MUSIC_TOOLS, label="soft-music")


def ensure_gesture_intent_tools_enabled(instance_path: str | Path | None, logger: logging.Logger) -> list[str]:
    """Merge react_to_gesture_intent into a stale Tool-access override when authored defaults include it."""
    return _ensure_authored_tools_enabled(instance_path, logger, _GESTURE_INTENT_TOOLS, label="gesture-intent")


def ensure_baby_care_tools_enabled(instance_path: str | Path | None, logger: logging.Logger) -> list[str]:
    """Merge baby-care tools into a stale instance tool override when authored defaults include them."""
    return _ensure_authored_tools_enabled(instance_path, logger, _BABY_CARE_TOOLS, label="baby-care")


def initialize_tools_with_default_fallback(
    instance_path: str | Path | None,
    logger: logging.Logger,
) -> str | None:
    """Load the tool registry, degrading to the default profile if need be.

    Returns the profile that was abandoned, or None when the selection loaded.
    """
    # User profiles predating profile.md were never migrated on disk; convert
    # them here so the strict readers below (and the settings UI) see them.
    try:
        migrate_legacy_profiles(config.user_personalities_root())
    except Exception as exc:
        logger.warning("Legacy profile migration failed: %s", exc)

    try:
        migrate_bundled_tool_spaces(instance_path)
    except Exception as exc:
        logger.warning("Bundled Tool Space migration failed: %s", exc)

    ensure_soft_music_tools_enabled(instance_path, logger)
    ensure_gesture_intent_tools_enabled(instance_path, logger)
    ensure_baby_care_tools_enabled(instance_path, logger)

    try:
        initialize_tools(instance_path=instance_path)
        return None
    except Exception as exc:
        selected_profile = config.REACHY_MINI_CUSTOM_PROFILE
        if not selected_profile or selected_profile == DEFAULT_PROFILE_NAME:
            raise

        # set_custom_profile is a no-op while LOCKED_PROFILE is pinned, so
        # confirm the switch took effect before announcing a fallback.
        set_custom_profile(DEFAULT_PROFILE_NAME)
        if config.REACHY_MINI_CUSTOM_PROFILE != DEFAULT_PROFILE_NAME:
            logger.error(
                "Profile %r could not be loaded (%s) and this build is locked to it, "
                "so there is no profile to fall back to.",
                selected_profile,
                exc,
            )
            raise

        logger.error(
            "Profile %r could not be loaded (%s); starting on the %r profile instead. "
            "Reselect or repair it from the settings UI to restore it.",
            selected_profile,
            exc,
            DEFAULT_PROFILE_NAME,
        )
        ensure_soft_music_tools_enabled(instance_path, logger)
        ensure_gesture_intent_tools_enabled(instance_path, logger)
        ensure_baby_care_tools_enabled(instance_path, logger)
        initialize_tools(instance_path=instance_path, force=True)
        return selected_profile


def request_stop_current_app(robot: ReachyMini, logger: logging.Logger) -> bool:
    """Request the Reachy Mini daemon to stop the current app."""
    try:
        daemon_request(
            robot,
            _STOP_CURRENT_APP_PATH,
            method="POST",
            timeout_s=_STOP_CURRENT_APP_TIMEOUT_S,
        )
    except DaemonApiError as e:
        # The daemon uses this detail for STOPPING, DONE, and no current app.
        if e.status_code == 400 and e.detail == "No app is currently running":
            logger.info(
                "Current app via %s is already stopped or stopping (HTTP %s)",
                _STOP_CURRENT_APP_PATH,
                e.status_code,
            )
            return True
        logger.error("Failed to request current app stop: %s", e)
        return False

    logger.info("Requested current app stop via %s", _STOP_CURRENT_APP_PATH)
    return True


def _is_sleep_head_pose(head_pose: npt.ArrayLike) -> bool:
    try:
        current_head_pose: npt.NDArray[np.float64] = np.asarray(head_pose, dtype=np.float64)
    except (TypeError, ValueError):
        return False

    if current_head_pose.shape != (4, 4):
        return False

    pose_distances = distance_between_poses(current_head_pose, SLEEP_HEAD_POSE)
    translation_distance = float(pose_distances[0])
    rotation_angle = float(pose_distances[1])
    return (
        translation_distance <= _SLEEP_HEAD_TRANSLATION_TOLERANCE_M
        and rotation_angle <= _SLEEP_HEAD_ROTATION_TOLERANCE_RAD
    )


def wake_up_if_sleeping(robot: ReachyMini, logger: logging.Logger) -> bool:
    """Run the SDK wake-up movement when Reachy starts from the sleep pose."""
    try:
        head_pose = robot.get_current_head_pose()
    except Exception as e:
        logger.warning("Could not read robot pose before startup wake-up check: %s", e)
        return False

    if not _is_sleep_head_pose(head_pose):
        return False

    logger.info("Robot is in sleep pose; running wake-up movement.")
    try:
        robot.enable_motors()
        robot.wake_up()
    except Exception as e:
        logger.error("Failed to run wake-up movement: %s", e)
        return False
    return True


def run_go_to_sleep_tool(deps: ToolDependencies, logger: logging.Logger) -> dict[str, object]:
    """Run the shared go_to_sleep tool from synchronous shutdown paths."""
    try:
        return asyncio.run(GoToSleep()(deps))
    except Exception as e:
        logger.error("Failed to run go_to_sleep tool during shutdown: %s", e)
        return {"error": f"go_to_sleep failed: {type(e).__name__}: {e}"}
