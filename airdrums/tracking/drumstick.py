"""
airdrums.tracking.drumstick
===========================
Virtual drumstick model. The drumstick extends from the wrist along the
forearm direction (elbow -> wrist). Only the TIP is used as the strike
landmark for upper body drums.
"""
from __future__ import annotations

import logging
import math
import time
from collections import deque
from typing import Deque, Optional, Tuple

import numpy as np

from .. import config
from .skeleton import Joint

log = logging.getLogger(__name__)


class Drumstick:
    """Virtual drumstick originating at the wrist."""

    def __init__(self, side: str, length: float = config.DRUMSTICK_LENGTH_DEFAULT):
        assert side in ("left", "right")
        self.side = side
        self.length = float(length)
        self._history: Deque[Tuple[float, float, float, float]] = deque(
            maxlen=config.VELOCITY_HISTORY_FRAMES
        )   # (t, x, y, z)
        self.tip_x = 0.0
        self.tip_y = 0.0
        self.tip_z = 0.0
        self.tip_vx = 0.0
        self.tip_vy = 0.0
        self.tip_vz = 0.0
        self.tip_speed = 0.0
        self.visible = False
        self._wrist_joint: Optional[Joint] = None
        self._elbow_joint: Optional[Joint] = None

    # ------------------------------------------------------------------
    def set_length(self, length: float) -> None:
        self.length = float(max(config.DRUMSTICK_LENGTH_MIN,
                                min(config.DRUMSTICK_LENGTH_MAX, length)))

    # ------------------------------------------------------------------
    def update(self, wrist: Joint, elbow: Optional[Joint]) -> None:
        """Recompute tip position and velocity from wrist + elbow joints."""
        self._wrist_joint = wrist
        self._elbow_joint = elbow
        if wrist is None or not wrist.visible:
            self.visible = False
            return

        # Forearm direction; fall back to pointing down if elbow missing
        if elbow is not None and elbow.visible:
            dx, dy, dz = wrist.x - elbow.x, wrist.y - elbow.y, wrist.z_depth - elbow.z_depth
            norm = math.sqrt(dx * dx + dy * dy + dz * dz) + 1e-6
            nx, ny, nz = dx / norm, dy / norm, dz / norm
        else:
            # Graceful fallback: tip == wrist
            nx = ny = nz = 0.0

        self.tip_x = wrist.x + nx * self.length
        self.tip_y = wrist.y + ny * self.length
        self.tip_z = wrist.z_depth + nz * self.length
        self.visible = True

        now = time.time()
        if self._history:
            pt, px, py, pz = self._history[-1]
            dt = max(now - pt, 1e-3)
            self.tip_vx = (self.tip_x - px) / dt
            self.tip_vy = (self.tip_y - py) / dt
            self.tip_vz = (self.tip_z - pz) / dt
            self.tip_speed = float(np.sqrt(self.tip_vx ** 2 + self.tip_vy ** 2 + self.tip_vz ** 2))
        self._history.append((now, self.tip_x, self.tip_y, self.tip_z))

    # ------------------------------------------------------------------
    def to_joint(self) -> Joint:
        """Convert the tip to a Joint so downstream code can treat it uniformly."""
        return Joint(
            name=f"{self.side}_tip",
            x=self.tip_x, y=self.tip_y,
            z_depth=self.tip_z,
            z_pose=0.0,
            visibility=1.0 if self.visible else 0.0,
            vx=self.tip_vx, vy=self.tip_vy, vz=self.tip_vz,
            speed=self.tip_speed,
            timestamp=time.time(),
        )

    # ------------------------------------------------------------------
    def draw(self, frame: np.ndarray, strike_flash: float = 0.0) -> None:
        """Render a tapered drumstick + tip on the given BGR frame.

        ``strike_flash`` in [0, 1] controls post-strike visual intensity;
        the UI layer decays it over time.
        """
        import cv2
        if not self.visible or self._wrist_joint is None:
            return
        h, w = frame.shape[:2]
        wrist_px = (int(self._wrist_joint.x * w), int(self._wrist_joint.y * h))
        tip_px = (int(self.tip_x * w), int(self.tip_y * h))

        # Base color per hand
        base_bgr = (255, 120, 60) if self.side == "left" else (60, 60, 255)
        # Brightness scales with speed
        boost = min(1.0, self.tip_speed / 4.0)
        color = tuple(int(base_bgr[i] + (255 - base_bgr[i]) * boost) for i in range(3))

        # Shadow first
        shadow_offset = 4
        cv2.line(frame,
                 (wrist_px[0] + shadow_offset, wrist_px[1] + shadow_offset),
                 (tip_px[0] + shadow_offset, tip_px[1] + shadow_offset),
                 (30, 30, 30),
                 config.DRUMSTICK_GRIP_PX, lineType=cv2.LINE_AA)

        # Tapered stick drawn as 3 overlapping segments of decreasing thickness
        segs = 5
        for i in range(segs):
            t0 = i / segs
            t1 = (i + 1) / segs
            p0 = (int(wrist_px[0] + (tip_px[0] - wrist_px[0]) * t0),
                  int(wrist_px[1] + (tip_px[1] - wrist_px[1]) * t0))
            p1 = (int(wrist_px[0] + (tip_px[0] - wrist_px[0]) * t1),
                  int(wrist_px[1] + (tip_px[1] - wrist_px[1]) * t1))
            thickness = int(config.DRUMSTICK_GRIP_PX - (config.DRUMSTICK_GRIP_PX - config.DRUMSTICK_TIP_PX) * t1)
            cv2.line(frame, p0, p1, color, max(1, thickness), lineType=cv2.LINE_AA)

        # Grip band
        grip_end = (int(wrist_px[0] + (tip_px[0] - wrist_px[0]) * 0.12),
                    int(wrist_px[1] + (tip_px[1] - wrist_px[1]) * 0.12))
        band_color = (40, 200, 40) if self.side == "left" else (40, 40, 200)
        cv2.line(frame, wrist_px, grip_end, band_color,
                 config.DRUMSTICK_GRIP_PX + 2, lineType=cv2.LINE_AA)

        # Tip: filled circle with outline
        cv2.circle(frame, tip_px, config.DRUMSTICK_TIP_RADIUS_PX, color, -1, lineType=cv2.LINE_AA)
        cv2.circle(frame, tip_px, config.DRUMSTICK_TIP_RADIUS_PX + 1, (255, 255, 255), 1, lineType=cv2.LINE_AA)

        # Strike burst
        if strike_flash > 0.01:
            ring_r = int(8 + 40 * strike_flash)
            cv2.circle(frame, tip_px, ring_r, (255, 255, 255),
                       max(1, int(3 * strike_flash)), lineType=cv2.LINE_AA)
            # Sparkle particles
            for ang in range(0, 360, 45):
                rad = math.radians(ang)
                pr = ring_r + 6
                px = int(tip_px[0] + math.cos(rad) * pr)
                py = int(tip_px[1] + math.sin(rad) * pr)
                cv2.circle(frame, (px, py), 2, (255, 255, 255), -1)
