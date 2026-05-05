"""Robot behaviour helpers – head poses and antenna emotes during conversation.

All movements are non-blocking fire-and-forget calls so they don't delay
the speech pipeline.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.spatial.transform import Rotation as R

if TYPE_CHECKING:
    from reachy_mini import ReachyMini

logger = logging.getLogger(__name__)


def _head_pose(pitch_deg: float = 0.0, yaw_deg: float = 0.0, roll_deg: float = 0.0) -> np.ndarray:
    """Build a 4×4 head pose matrix from intrinsic XYZ Euler angles (degrees)."""
    mat = np.eye(4)
    mat[:3, :3] = R.from_euler("xyz", [pitch_deg, yaw_deg, roll_deg], degrees=True).as_matrix()
    return mat


def _goto(robot: "ReachyMini", head: np.ndarray | None = None,
          antennas: list | None = None, duration: float = 0.5) -> None:
    """Non-blocking goto_target call in a background thread."""
    def _go():
        try:
            kwargs: dict = {"duration": duration}
            if head is not None:
                kwargs["head"] = head
            if antennas is not None:
                kwargs["antennas"] = antennas
            robot.goto_target(**kwargs)
        except Exception as exc:
            msg = str(exc)
            # SDK/daemon version skew causes spurious task-timeout errors even
            # when the movement was accepted; log at DEBUG to avoid log spam.
            if "did not complete in time" in msg or "timed out" in msg.lower():
                logger.debug("goto_target timed out (SDK/daemon mismatch?): %s", msg)
            else:
                logger.warning("goto_target failed: %s", exc)
    threading.Thread(target=_go, daemon=True).start()


def go_idle(robot: "ReachyMini", cfg: dict[str, Any]) -> None:
    """Neutral, slightly-downward-looking pose."""
    pitch = cfg.get("robot", {}).get("idle_head_pitch_deg", -5)
    _goto(robot, head=_head_pose(pitch_deg=pitch), duration=1.0)


def on_listen_start(robot: "ReachyMini", cfg: dict[str, Any]) -> None:
    """Slight upward tilt to signal attention."""
    if not cfg.get("robot", {}).get("head_nod_on_listen", True):
        return
    _goto(robot, head=_head_pose(pitch_deg=5), duration=0.4)


def on_listen_end(robot: "ReachyMini", cfg: dict[str, Any]) -> None:
    """Small nod to acknowledge end of utterance."""
    if not cfg.get("robot", {}).get("head_nod_on_listen", True):
        return
    _goto(robot, head=_head_pose(pitch_deg=-8), duration=0.3)


def on_thinking(robot: "ReachyMini", cfg: dict[str, Any]) -> None:
    """Slight sideways tilt – 'thinking' pose."""
    _goto(robot, head=_head_pose(roll_deg=8), duration=0.5)


def on_speaking_start(robot: "ReachyMini", cfg: dict[str, Any]) -> None:
    """Straighten head and optionally wiggle antennas."""
    _goto(robot, head=_head_pose(), duration=0.3)
    if cfg.get("robot", {}).get("antenna_wiggle_on_speak", True):
        threading.Thread(target=_antenna_wiggle, args=(robot,), daemon=True).start()


def on_error(robot: "ReachyMini", cfg: dict[str, Any]) -> None:
    """Brief head shake on error."""
    if not cfg.get("robot", {}).get("head_shake_on_error", True):
        return
    def _shake():
        for yaw in (10, -10, 8, -8, 0):
            try:
                robot.goto_target(head=_head_pose(yaw_deg=yaw), duration=0.15)
                time.sleep(0.15)
            except Exception:
                break
    threading.Thread(target=_shake, daemon=True).start()


def _antenna_wiggle(robot: "ReachyMini") -> None:
    """Gentle antenna animation while speaking."""
    try:
        positions = [0.3, -0.3, 0.2, -0.2, 0.0]
        for pos in positions:
            robot.goto_target(antennas=[pos, -pos], duration=0.25)
            time.sleep(0.25)
    except Exception as exc:
        logger.debug("Antenna wiggle error: %s", exc)
