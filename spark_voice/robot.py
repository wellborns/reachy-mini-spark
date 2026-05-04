"""Robot behaviour helpers – head poses and antenna emotes during conversation.

All movements are non-blocking fire-and-forget calls so they don't delay
the speech pipeline.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from reachy_mini import ReachyMini

logger = logging.getLogger(__name__)


def _safe_goto(robot: "ReachyMini", **kwargs) -> None:
    """Call robot.set_target in a background thread; swallow errors."""
    def _go():
        try:
            robot.set_target(**kwargs)
        except Exception as exc:
            logger.debug("set_target failed: %s", exc)
    threading.Thread(target=_go, daemon=True).start()


def go_idle(robot: "ReachyMini", cfg: dict[str, Any]) -> None:
    """Neutral, slightly-downward-looking pose."""
    pitch = math.radians(cfg.get("robot", {}).get("idle_head_pitch_deg", -5))
    _safe_goto(robot, head_pitch=pitch, head_yaw=0.0, head_roll=0.0, duration=1.0)


def on_listen_start(robot: "ReachyMini", cfg: dict[str, Any]) -> None:
    """Slight upward tilt to signal attention."""
    if not cfg.get("robot", {}).get("head_nod_on_listen", True):
        return
    _safe_goto(robot, head_pitch=math.radians(5), head_yaw=0.0, head_roll=0.0, duration=0.4)


def on_listen_end(robot: "ReachyMini", cfg: dict[str, Any]) -> None:
    """Small nod to acknowledge end of utterance."""
    if not cfg.get("robot", {}).get("head_nod_on_listen", True):
        return
    _safe_goto(robot, head_pitch=math.radians(-8), head_yaw=0.0, head_roll=0.0, duration=0.3)


def on_thinking(robot: "ReachyMini", cfg: dict[str, Any]) -> None:
    """Slight sideways tilt – 'thinking' pose."""
    _safe_goto(robot, head_pitch=math.radians(0), head_yaw=0.0, head_roll=math.radians(8), duration=0.5)


def on_speaking_start(robot: "ReachyMini", cfg: dict[str, Any]) -> None:
    """Straighten head and optionally wiggle antennas."""
    _safe_goto(robot, head_pitch=math.radians(0), head_yaw=0.0, head_roll=0.0, duration=0.3)
    if cfg.get("robot", {}).get("antenna_wiggle_on_speak", True):
        threading.Thread(
            target=_antenna_wiggle, args=(robot,), daemon=True
        ).start()


def on_error(robot: "ReachyMini", cfg: dict[str, Any]) -> None:
    """Brief head shake on error."""
    if not cfg.get("robot", {}).get("head_shake_on_error", True):
        return
    def _shake():
        for yaw in (0.2, -0.2, 0.15, -0.15, 0.0):
            try:
                robot.set_target(head_yaw=yaw, duration=0.15)
                time.sleep(0.15)
            except Exception:
                break
    threading.Thread(target=_shake, daemon=True).start()


def _antenna_wiggle(robot: "ReachyMini") -> None:
    """Gentle antenna animation while speaking."""
    try:
        positions = [0.3, -0.3, 0.2, -0.2, 0.0]
        for pos in positions:
            robot.set_target(
                left_antenna=pos,
                right_antenna=-pos,
                duration=0.25,
            )
            time.sleep(0.25)
    except Exception as exc:
        logger.debug("Antenna wiggle error: %s", exc)
