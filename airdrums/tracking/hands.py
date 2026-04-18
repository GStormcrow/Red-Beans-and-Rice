"""
airdrums.tracking.hands
=======================
V2 MediaPipe Hands wrapper.  Provides HandSkeleton which detects up to two
hands per frame, enriches landmarks with rolling-window velocities, and
optionally samples z-depth from a DepthEngine depth map.
"""
from __future__ import annotations

import collections
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np

from .. import config

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Joint dataclass
# ---------------------------------------------------------------------------
@dataclass
class Joint:
    """One MediaPipe hand landmark at a single point in time.

    Coordinates ``x``, ``y`` are normalized to [0, 1] (fraction of frame).
    ``z_depth`` is populated from the DepthEngine depth map when available
    (5x5 median kernel); otherwise 0.0.  Velocity components are computed
    over a rolling history deque.
    """

    x: float = 0.0
    y: float = 0.0
    z_depth: float = 0.0
    visibility: float = 0.0
    vx: float = 0.0
    vy: float = 0.0
    vz: float = 0.0
    speed: float = 0.0


# ---------------------------------------------------------------------------
# HandSkeleton
# ---------------------------------------------------------------------------
class HandSkeleton:
    """Wraps ``mp.solutions.hands.Hands`` and returns per-landmark Joint objects.

    Usage::

        skel = HandSkeleton(fps=30)
        hands = skel.update(frame_bgr, depth_map)
        right_wrist = skel.get_wrist("right")
    """

    def __init__(self, fps: float = 30.0) -> None:
        """Initialise MediaPipe Hands and rolling velocity deques.

        Args:
            fps: Expected capture frame rate; controls the velocity deque length.
        """
        import mediapipe as mp  # type: ignore

        self._mp = mp
        self._hands = mp.solutions.hands.Hands(
            max_num_hands=config.MP_MAX_HANDS,
            min_detection_confidence=config.MP_DETECTION_CONFIDENCE,
            min_tracking_confidence=config.MP_TRACKING_CONFIDENCE,
        )

        self._fps: float = max(1.0, float(fps))
        self._deque_len: int = max(2, int(config.VELOCITY_DEQUE_FACTOR * self._fps / 30))

        # hand_id -> list[deque[Tuple[x, y, z, t]]]  (one deque per landmark)
        self._history: Dict[str, List[Deque[Tuple[float, float, float, float]]]] = {
            "left": [deque(maxlen=self._deque_len) for _ in range(21)],
            "right": [deque(maxlen=self._deque_len) for _ in range(21)],
        }

        # Most recent result (absent key = hand not in frame)
        self._latest: Dict[str, List[Joint]] = {}

        self._last_frame_time: float = time.monotonic()

        log.info(
            "HandSkeleton initialised (fps=%.1f, deque_len=%d)",
            self._fps, self._deque_len,
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------
    @property
    def fps(self) -> float:
        """Current estimated frame rate (updated each call to update())."""
        return self._fps

    def update(
        self,
        frame_bgr: np.ndarray,
        depth_map: Optional[np.ndarray] = None,
    ) -> Dict[str, List[Joint]]:
        """Process one BGR frame and return detected hands.

        Args:
            frame_bgr: ``uint8`` BGR image from the capture device.
            depth_map: Optional ``float32`` depth array from
                       ``DepthEngine.latest_depth``.  When provided, each
                       landmark's ``z_depth`` is sampled with a 5x5 median
                       kernel.

        Returns:
            Mapping of ``"left"`` / ``"right"`` to a list of 21 :class:`Joint`
            objects.  A key is absent when that hand is not detected.
        """
        import cv2  # local import keeps module light

        # Measure FPS
        now = time.monotonic()
        dt = now - self._last_frame_time
        if dt > 0:
            alpha = 0.1  # exponential smoothing
            self._fps = alpha * (1.0 / dt) + (1 - alpha) * self._fps
        self._last_frame_time = now

        # Update deque length if FPS changed significantly
        new_len = max(2, int(config.VELOCITY_DEQUE_FACTOR * self._fps / 30))
        if new_len != self._deque_len:
            self._deque_len = new_len
            for hand_id_key in list(self._history.keys()):
                self._history[hand_id_key] = [
                    collections.deque(dq, maxlen=new_len)
                    for dq in self._history[hand_id_key]
                ]

        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        result = self._hands.process(rgb)

        detected: Dict[str, List[Joint]] = {}

        if not result.multi_hand_landmarks:
            self._latest = detected
            return detected

        handedness_list = result.multi_handedness or []
        for hand_idx, hand_lms in enumerate(result.multi_hand_landmarks):
            # MediaPipe reports handedness from the *mirror* perspective;
            # we flip to match anatomical labelling for a normal selfie view.
            if hand_idx < len(handedness_list):
                mp_label = handedness_list[hand_idx].classification[0].label  # "Left"/"Right"
                hand_id = "right" if mp_label == "Left" else "left"
            else:
                hand_id = "right"

            joints: List[Joint] = []
            lm_list = hand_lms.landmark

            for lm_idx, lm in enumerate(lm_list):
                vis = float(getattr(lm, "visibility", 1.0))

                x = float(lm.x)
                y = float(lm.y)

                # Depth from depth map
                z_depth = 0.0
                if depth_map is not None:
                    z_depth = self._sample_depth(depth_map, x, y)

                # Velocity from rolling history
                vx, vy, vz, speed = self._compute_velocity(
                    hand_id, lm_idx, x, y, z_depth, now
                )

                joint = Joint(
                    x=x, y=y,
                    z_depth=z_depth,
                    visibility=vis,
                    vx=vx, vy=vy, vz=vz,
                    speed=speed,
                )
                joints.append(joint)

                # Update history deque
                self._history[hand_id][lm_idx].append((x, y, z_depth, now))

            detected[hand_id] = joints

        self._latest = detected
        return detected

    def get_wrist(self, hand_id: str) -> Optional[Joint]:
        """Return the wrist Joint (landmark 0) for the given hand, or None.

        Args:
            hand_id: ``"left"`` or ``"right"``.

        Returns:
            :class:`Joint` if the hand is detected and the landmark is
            sufficiently visible, otherwise ``None``.
        """
        joints = self._latest.get(hand_id)
        if not joints or len(joints) <= config.LM_WRIST:
            return None
        j = joints[config.LM_WRIST]
        if j.visibility < 0.5 and j.visibility != 0.0:
            # visibility==0.0 means MediaPipe didn't report it; treat as visible
            return None
        return j

    def get_index_tip(self, hand_id: str) -> Optional[Joint]:
        """Return the INDEX_FINGER_TIP Joint (landmark 8) — primary trigger point.

        Args:
            hand_id: ``"left"`` or ``"right"``.
        """
        joints = self._latest.get(hand_id)
        if not joints or len(joints) <= config.LM_INDEX_TIP:
            return None
        j = joints[config.LM_INDEX_TIP]
        if j.visibility < 0.5 and j.visibility != 0.0:
            return None
        return j

    def get_index_dip(self, hand_id: str) -> Optional[Joint]:
        """Return the INDEX_FINGER_DIP Joint (landmark 7) — drumstick origin.

        Args:
            hand_id: ``"left"`` or ``"right"``.
        """
        joints = self._latest.get(hand_id)
        if not joints or len(joints) <= config.LM_INDEX_DIP:
            return None
        j = joints[config.LM_INDEX_DIP]
        if j.visibility < 0.5 and j.visibility != 0.0:
            return None
        return j

    def get_mcp(self, hand_id: str) -> Optional[Joint]:
        """Return the middle-finger MCP Joint (landmark 9) for the given hand.

        Args:
            hand_id: ``"left"`` or ``"right"``.

        Returns:
            :class:`Joint` if detected and visible, otherwise ``None``.
        """
        joints = self._latest.get(hand_id)
        if not joints or len(joints) <= config.LM_MIDDLE_MCP:
            return None
        j = joints[config.LM_MIDDLE_MCP]
        if j.visibility < 0.5 and j.visibility != 0.0:
            return None
        return j

    def close(self) -> None:
        """Release MediaPipe resources."""
        try:
            self._hands.close()
        except Exception:  # noqa: BLE001
            pass
        log.info("HandSkeleton closed")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _sample_depth(
        self, depth_map: np.ndarray, x_norm: float, y_norm: float
    ) -> float:
        """Sample depth_map at a normalised position using a 5x5 median kernel.

        Args:
            depth_map: Float32 array (H x W).
            x_norm: Normalised x in [0, 1].
            y_norm: Normalised y in [0, 1].

        Returns:
            Median depth value within the 5x5 neighbourhood, or 0.0 on error.
        """
        if depth_map is None or depth_map.size == 0:
            return 0.0
        dh, dw = depth_map.shape[:2]
        px = int(np.clip(x_norm * dw, 0, dw - 1))
        py = int(np.clip(y_norm * dh, 0, dh - 1))
        k = config.DEPTH_KERNEL_SIZE // 2  # 2 for a 5x5 kernel
        x0 = max(0, px - k)
        x1 = min(dw, px + k + 1)
        y0 = max(0, py - k)
        y1 = min(dh, py + k + 1)
        patch = depth_map[y0:y1, x0:x1]
        if patch.size == 0:
            return 0.0
        return float(np.median(patch))

    def _compute_velocity(
        self,
        hand_id: str,
        lm_idx: int,
        x: float,
        y: float,
        z: float,
        t: float,
    ) -> Tuple[float, float, float, float]:
        """Finite-difference velocity from the rolling deque.

        Args:
            hand_id: ``"left"`` or ``"right"``.
            lm_idx: Landmark index 0-20.
            x, y, z: Current normalised position.
            t: Current timestamp (monotonic seconds).

        Returns:
            Tuple of ``(vx, vy, vz, speed)``.
        """
        dq = self._history[hand_id][lm_idx]
        if not dq:
            return 0.0, 0.0, 0.0, 0.0
        px, py, pz, pt = dq[-1]
        dt = max(t - pt, 1e-3)
        vx = (x - px) / dt
        vy = (y - py) / dt
        vz = (z - pz) / dt
        speed = float(np.sqrt(vx * vx + vy * vy + vz * vz))
        return vx, vy, vz, speed
