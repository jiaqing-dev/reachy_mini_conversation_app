"""Tests for app-level runtime behavior."""

import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

import reachy_mini_conversation_app.main as main_mod


def test_set_head_wobbling_toggles_sdk_and_logs_failures() -> None:
    """Dialogue enables SDK wobbling; failures are visible warnings, not silent."""
    robot = MagicMock()
    logger = MagicMock()

    main_mod._set_head_wobbling(robot, enabled=True, logger=logger)
    robot.enable_wobbling.assert_called_once_with()
    logger.info.assert_called_once()

    robot.enable_wobbling.side_effect = RuntimeError("no daemon wobbler")
    main_mod._set_head_wobbling(robot, enabled=True, logger=logger)
    logger.warning.assert_called_once()

    robot.disable_wobbling.side_effect = None
    main_mod._set_head_wobbling(robot, enabled=False, logger=logger)
    robot.disable_wobbling.assert_called_once_with()


def test_inactivity_timeout_thread_goes_to_sleep() -> None:
    """The watchdog should use the shared sleep shutdown path once activity is too old."""
    stream_manager = SimpleNamespace(seconds_since_activity=lambda: 10.0, close=MagicMock())
    go_to_sleep = MagicMock(return_value={"status": "sleeping"})

    thread = main_mod._start_inactivity_timeout_thread(
        timeout_minutes=0.0001,
        stream_manager=stream_manager,
        logger=MagicMock(),
        app_stop_event=threading.Event(),
        go_to_sleep=go_to_sleep,
    )

    thread.join(timeout=1.0)
    assert not thread.is_alive()
    go_to_sleep.assert_called_once_with()
    stream_manager.close.assert_not_called()


def test_inactivity_timeout_thread_closes_stream_manager_without_sleep_callback() -> None:
    """The watchdog should still close the stream when no sleep callback is available."""
    stream_manager = SimpleNamespace(seconds_since_activity=lambda: 10.0, close=MagicMock())

    thread = main_mod._start_inactivity_timeout_thread(
        timeout_minutes=0.0001,
        stream_manager=stream_manager,
        logger=MagicMock(),
        app_stop_event=threading.Event(),
    )

    thread.join(timeout=1.0)
    assert not thread.is_alive()
    stream_manager.close.assert_called_once_with()


def test_inactivity_timeout_thread_exits_on_cancel_event() -> None:
    """Cancelling the watchdog stops it without sleeping the robot."""
    stream_manager = SimpleNamespace(seconds_since_activity=lambda: 0.0, close=MagicMock())
    go_to_sleep = MagicMock()
    cancel_event = threading.Event()
    cancel_event.set()

    thread = main_mod._start_inactivity_timeout_thread(
        timeout_minutes=60.0,
        stream_manager=stream_manager,
        logger=MagicMock(),
        app_stop_event=threading.Event(),
        go_to_sleep=go_to_sleep,
        cancel_event=cancel_event,
    )

    thread.join(timeout=1.0)
    assert not thread.is_alive()
    go_to_sleep.assert_not_called()
    stream_manager.close.assert_not_called()
