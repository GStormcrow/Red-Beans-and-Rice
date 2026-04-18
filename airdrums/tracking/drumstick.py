"""
airdrums.tracking.drumstick
===========================
V2 virtual drumstick model.  The tip is computed by projecting from the wrist
in a direction perpendicular to the wrist-to-MCP axis, so it mimics a stick
held in the fist with the tip pointing outward to the side of the hand.
"""
from __future__ import annotations

import logging
import math
from collections import deque
from typing import Deque, Optional, Tuple

import numpy as np

from .. import config
from .hands import Joint

log = logging.getLogger(__name__)


class Drumstick:
    """Virtual drumstick for one hand; tracks tip position and renders the stick.

    The tip vector is obtained by rotating the wrist-to-MCP axis 90° (CW for
    left hand, CCW for right hand) and scaling by ``stick_length``.  A rolling
    deque of tip positions is kept to compute smooth tip velocity.
    """

    def __init__(
        self,
        hand_id: str,
        stick_length: float = config.STICK_LENGTH_DEFAULT,
    ) -> None:
        """Create a Drumstick for the given hand.

        Args:
            hand_id: ``"left"`` or ``"right"``.
            stick_length: Initial stick length in normalised units.
        """
        if hand_id not in ("left", "right"):
            raise ValueError(f"hand_id must be 'left' or 'right', got {hand_id!r}")
        self.hand_id = hand_id
        self.stick_length: float = float(
            np.clip(stick_length, config.STICK_LENGTH_MIN, config.STICK_LENGTH_MAX)
        )

        # Rolling tip history: (x, y, z, t)
        self._deque_len: int = max(2, int(config.VELOCITY_DEQUE_FACTOR * 1.0))
        self._history: Deque[Tuple[float, float, float, float]] = deque(
            maxlen=self._deque_len
        )

        # Public tip state
        self._tip_x: float = 0.0
        self._tip_y: float = 0.0
        self._tip_z: float = 0.0
        self.tip_vx: float = 0.0
        self.tip_vy: float = 0.0
        self.tip_vz: float = 0.0
        self.tip_speed: float = 0.0

        # Wrist position kept for draw()
        self._wrist_x: float = 0.0
        self._wrist_y: float = 0.0
        self._visible: bool = False

        log.debug("Drumstick created (hand=%s, length=%.3f)", hand_id, stick_length)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def tip_x(self) -> float:
        """Normalised x coordinate of the stick tip."""
        return self._tip_x

    @property
    def tip_y(self) -> float:
        """Normalised y coordinate of the stick tip."""
        return self._tip_y

    @property
    def tip(self) -> Tuple[float, float]:
        """Normalised (x, y) of the stick tip."""
        return (self._tip_x, self._tip_y)

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------
    def update(
        self,
        wrist: Joint,
        index_dip: Optional[Joint],
        fps: float,
    ) -> None:
        """Recompute tip position and velocity for the current frame.

        The stick base is placed at landmark 7 (INDEX_FINGER_DIP) and extends
        further in the wrist→index_dip direction, so it follows the natural
        pointing direction of the index finger.  Falls back to wrist when
        index_dip is unavailable.

        Args:
            wrist: Wrist Joint (landmark 0).
            index_dip: INDEX_FINGER_DIP Joint (landmark 7), may be None.
            fps: Current capture FPS — used to keep the velocity deque at the
                 right length.
        """
        import time  # local to avoid module-level import cost

        if wrist is None:
            self._visible = False
            return

        # Update deque length when FPS changes
        new_len = max(2, int(config.VELOCITY_DEQUE_FACTOR * fps / 30))
        if new_len != self._deque_len:
            self._deque_len = new_len
            old = list(self._history)
            self._history = deque(old[-new_len:], maxlen=new_len)

        wx, wy = wrist.x, wrist.y

        use_dip = (
            index_dip is not None
            and (index_dip.visibility >= 0.5 or index_dip.visibility == 0.0)
        )
        if use_dip:
            bx, by = index_dip.x, index_dip.y   # base = landmark 7
            dx, dy = bx - wx, by - wy
            norm = math.sqrt(dx * dx + dy * dy) + 1e-8
            # Unit direction wrist -> index_dip; stick extends further from base
            ux, uy = dx / norm, dy / norm
            tx = bx + ux * self.stick_length
            ty = by + uy * self.stick_length
        else:
            # Fallback: base and tip both at wrist
            bx, by = wx, wy
            tx, ty = wx, wy

        # Store base for draw()
        self._wrist_x, self._wrist_y = bx, by

        tz = wrist.z_depth

        self._tip_x = tx
        self._tip_y = ty
        self._tip_z = tz
        self._visible = True

        now = time.monotonic()
        if self._history:
            px2, py2, pz2, pt = self._history[-1]
            dt = max(now - pt, 1e-3)
            self.tip_vx = (tx - px2) / dt
            self.tip_vy = (ty - py2) / dt
            self.tip_vz = (tz - pz2) / dt
            self.tip_speed = float(
                math.sqrt(self.tip_vx ** 2 + self.tip_vy ** 2 + self.tip_vz ** 2)
            )
        self._history.append((tx, ty, tz, now))

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------
    def draw(self, frame: np.ndarray, on_strike: bool = False) -> None:
        """Render the drumstick, tip trail, drop shadow, and strike effect.

        The stick is drawn as a tapered line (6px grip, 2px tip) using five
        overlapping segments.  A trail of the last four tip positions is drawn
        on a semi-transparent overlay at decreasing opacities.  On strike, an
        expanding ring and 8 radiating particle dots are drawn at the tip.

        Args:
            frame: BGR ``uint8`` numpy array; drawn in-place.
            on_strike: When ``True``, draw the strike burst effect.
        """
        import cv2  # local import

        if not self._visible:
            return

        h, w = frame.shape[:2]

        wrist_px = (int(self._wrist_x * w), int(self._wrist_y * h))
        tip_px = (int(self._tip_x * w), int(self._tip_y * h))

        # Base colour: left=blue-tint BGR (255,100,50), right=red-tint BGR (50,100,255)
        if self.hand_id == "left":
            base = (255, 100, 50)
        else:
            base = (50, 100, 255)

        # Lerp toward white as speed increases
        t_speed = min(self.tip_speed / 0.8, 1.0)
        color: Tuple[int, int, int] = tuple(
            int(base[i] + (255 - base[i]) * t_speed) for i in range(3)
        )  # type: ignore[assignment]

        # ---- Drop shadow (offset +3, +3) in dark semi-transparent colour ----
        shadow_overlay = frame.copy()
        shadow_color = (20, 20, 20)
        sw, sh = wrist_px[0] + 3, wrist_px[1] + 3
        tw, th = tip_px[0] + 3, tip_px[1] + 3
        cv2.line(shadow_overlay, (sw, sh), (tw, th), shadow_color, 4, lineType=cv2.LINE_AA)
        cv2.addWeighted(shadow_overlay, 0.4, frame, 0.6, 0, frame)

        # ---- Grip band (first 20% of stick, 8px) ----------------------------
        grip_frac = 0.20
        grip_end_px = (
            int(wrist_px[0] + (tip_px[0] - wrist_px[0]) * grip_frac),
            int(wrist_px[1] + (tip_px[1] - wrist_px[1]) * grip_frac),
        )
        grip_color = (60, 220, 60) if self.hand_id == "left" else (60, 60, 220)
        cv2.line(frame, wrist_px, grip_end_px, grip_color, 8, lineType=cv2.LINE_AA)

        # ---- Tapered body (5 segments, 6px -> 2px) ---------------------------
        n_segs = 5
        thick_grip = 6
        thick_tip = 2
        for i in range(n_segs):
            t0 = i / n_segs
            t1 = (i + 1) / n_segs
            p0 = (
                int(wrist_px[0] + (tip_px[0] - wrist_px[0]) * t0),
                int(wrist_px[1] + (tip_px[1] - wrist_px[1]) * t0),
            )
            p1 = (
                int(wrist_px[0] + (tip_px[0] - wrist_px[0]) * t1),
                int(wrist_px[1] + (tip_px[1] - wrist_px[1]) * t1),
            )
            thickness = max(1, int(thick_grip - (thick_grip - thick_tip) * t1))
            cv2.line(frame, p0, p1, color, thickness, lineType=cv2.LINE_AA)

        # ---- Stick trail (last 4 tip positions) -----------------------------
        trail_pts = list(self._history)
        if len(trail_pts) >= 2:
            trail_overlay = frame.copy()
            opacities = [0.6, 0.4, 0.2, 0.1]
            # Take the 4 positions just before the current tip
            recent = trail_pts[-5:-1]  # up to 4 positions
            for j, (tx, ty, _tz, _t) in enumerate(reversed(recent)):
                if j >= len(opacities):
                    break
                tpx = (int(tx * w), int(ty * h))
                cv2.circle(trail_overlay, tpx, 3, color, -1, lineType=cv2.LINE_AA)
            cv2.addWeighted(trail_overlay, opacities[0], frame, 1 - opacities[0], 0, frame)

        # ---- Tip: filled circle + 1px white outline --------------------------
        cv2.circle(frame, tip_px, 5, color, -1, lineType=cv2.LINE_AA)
        cv2.circle(frame, tip_px, 5, (255, 255, 255), 1, lineType=cv2.LINE_AA)

        # ---- Strike effect ---------------------------------------------------
        if on_strike:
            self._draw_strike(frame, tip_px)

    def _draw_strike(self, frame: np.ndarray, tip_px: Tuple[int, int]) -> None:
        """Draw expanding ring and 8 radiating particle dots at the tip.

        Args:
            frame: BGR ``uint8`` numpy array; drawn in-place.
            tip_px: Pixel position of the stick tip.
        """
        import cv2

        ring_r = 22
        cv2.circle(frame, tip_px, ring_r, (255, 255, 255), 2, lineType=cv2.LINE_AA)

        n_particles = 8
        particle_dist = ring_r + 10
        for i in range(n_particles):
            angle = math.radians(i * (360.0 / n_particles))
            px = int(tip_px[0] + math.cos(angle) * particle_dist)
            py = int(tip_px[1] + math.sin(angle) * particle_dist)
            cv2.circle(frame, (px, py), 3, (255, 255, 255), -1, lineType=cv2.LINE_AA)
