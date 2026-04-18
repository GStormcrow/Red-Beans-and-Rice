"""
airdrums.tracking.skeleton
==========================
Joint dataclass + DrumSkeleton: MediaPipe Pose wrapper that samples depth,
tracks visibility, computes per-joint velocities, and repositions drum zones
relative to the player's hip center / shoulder width.
"""
from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np

from .. import config

log = logging.getLogger(__name__)

# MediaPipe Pose landmark indices we care about. MediaPipe Pose provides 33.
LANDMARK_NAMES: Dict[str, int] = {
    "nose": 0,
    "left_shoulder": 11, "right_shoulder": 12,
    "left_elbow": 13,    "right_elbow": 14,
    "left_wrist": 15,    "right_wrist": 16,
    "left_hip": 23,      "right_hip": 24,
    "left_knee": 25,     "right_knee": 26,
    "left_ankle": 27,    "right_ankle": 28,
    "left_heel": 29,     "right_heel": 30,
    "left_foot_index": 31, "right_foot_index": 32,  # toes
}


@dataclass
class Joint:
    """One named joint at one moment in time.

    x, y are normalized to the frame (0..1). z_depth comes from Depth Anything
    V2 (centimeters after depth_scale calibration), z_pose comes from
    MediaPipe's own pseudo-depth. Velocities are finite-differenced across the
    rolling history deque.
    """
    name: str = ""
    x: float = 0.0
    y: float = 0.0
    z_depth: float = 0.0
    z_pose: float = 0.0
    visibility: float = 0.0
    vx: float = 0.0
    vy: float = 0.0
    vz: float = 0.0
    speed: float = 0.0
    timestamp: float = 0.0

    @property
    def visible(self) -> bool:
        return self.visibility >= config.VISIBILITY_THRESHOLD

    def as_tuple_xy(self) -> Tuple[float, float]:
        return (self.x, self.y)


class DrumSkeleton:
    """Runs MediaPipe Pose on each frame and exposes Joint objects with
    depth-sampled z and rolling-window velocities."""

    def __init__(self, pose_complexity: int = 1):
        # Lazy import: mediapipe is heavy and not always installed during tests.
        import mediapipe as mp  # type: ignore
        self._mp = mp
        self._pose = mp.solutions.pose.Pose(
            model_complexity=pose_complexity,
            enable_segmentation=False,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        # Rolling history: name -> deque of Joints
        self._history: Dict[str, Deque[Joint]] = {
            name: deque(maxlen=config.VELOCITY_HISTORY_FRAMES)
            for name in LANDMARK_NAMES
        }
        self.joints: Dict[str, Joint] = {name: Joint(name=name) for name in LANDMARK_NAMES}
        self.depth_scale: float = config.DEPTH_DEFAULT_SCALE
        self._frame_wh: Tuple[int, int] = (config.CAMERA_WIDTH, config.CAMERA_HEIGHT)
        log.info("DrumSkeleton initialised (pose_complexity=%d)", pose_complexity)

    # ------------------------------------------------------------------
    # Main entry
    # ------------------------------------------------------------------
    def update(self, frame_bgr: np.ndarray, depth_map: Optional[np.ndarray]) -> None:
        """Process one BGR frame, updating self.joints in place."""
        import cv2  # local import to keep module lightweight
        h, w = frame_bgr.shape[:2]
        self._frame_wh = (w, h)
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        result = self._pose.process(rgb)

        now = time.time()
        if not result.pose_landmarks:
            return

        lms = result.pose_landmarks.landmark
        for name, idx in LANDMARK_NAMES.items():
            lm = lms[idx]
            vis = float(lm.visibility)
            # Gracefully handle invisible landmarks: keep last known x/y but
            # drop visibility so downstream logic ignores them.
            if vis < config.VISIBILITY_THRESHOLD:
                self.joints[name].visibility = vis
                continue

            x = float(lm.x)
            y = float(lm.y)
            z_pose = float(lm.z)
            z_depth = self._sample_depth(depth_map, x, y) if depth_map is not None else 0.0

            vx, vy, vz, speed = self._compute_velocity(name, x, y, z_depth, now)

            joint = Joint(
                name=name,
                x=x, y=y,
                z_depth=z_depth,
                z_pose=z_pose,
                visibility=vis,
                vx=vx, vy=vy, vz=vz,
                speed=speed,
                timestamp=now,
            )
            self.joints[name] = joint
            self._history[name].append(joint)

    # ------------------------------------------------------------------
    # Depth sampling
    # ------------------------------------------------------------------
    def _sample_depth(self, depth_map: np.ndarray, x_norm: float, y_norm: float) -> float:
        """Sample the depth map with a 5x5 median kernel for stability."""
        if depth_map is None or depth_map.size == 0:
            return 0.0
        dh, dw = depth_map.shape[:2]
        px = int(np.clip(x_norm * dw, 0, dw - 1))
        py = int(np.clip(y_norm * dh, 0, dh - 1))
        k = config.DEPTH_SAMPLE_KERNEL // 2
        x0, x1 = max(0, px - k), min(dw, px + k + 1)
        y0, y1 = max(0, py - k), min(dh, py + k + 1)
        patch = depth_map[y0:y1, x0:x1]
        if patch.size == 0:
            return 0.0
        return float(np.median(patch)) * self.depth_scale

    # ------------------------------------------------------------------
    # Velocity computation
    # ------------------------------------------------------------------
    def _compute_velocity(self, name: str, x: float, y: float, z: float,
                          t: float) -> Tuple[float, float, float, float]:
        hist = self._history[name]
        if not hist:
            return 0.0, 0.0, 0.0, 0.0
        prev = hist[-1]
        dt = max(t - prev.timestamp, 1e-3)
        vx = (x - prev.x) / dt
        vy = (y - prev.y) / dt
        vz = (z - prev.z_depth) / dt
        speed = float(np.sqrt(vx * vx + vy * vy + vz * vz))
        return vx, vy, vz, speed

    # ------------------------------------------------------------------
    # Zone repositioning
    # ------------------------------------------------------------------
    def player_frame(self) -> Optional[Dict[str, float]]:
        """Return a reference frame (hip center + shoulder width) so zones
        can be repositioned relative to the player."""
        ls = self.joints.get("left_shoulder")
        rs = self.joints.get("right_shoulder")
        lh = self.joints.get("left_hip")
        rh = self.joints.get("right_hip")
        if not all(j and j.visible for j in (ls, rs, lh, rh)):
            return None
        hip_cx = (lh.x + rh.x) / 2
        hip_cy = (lh.y + rh.y) / 2
        shoulder_w = abs(ls.x - rs.x)
        torso_h = abs(((ls.y + rs.y) / 2) - hip_cy)
        return {
            "hip_cx": hip_cx, "hip_cy": hip_cy,
            "shoulder_w": max(shoulder_w, 0.15),
            "torso_h": max(torso_h, 0.15),
        }

    def rescale_zones(self, zones: List[config.DrumZone]) -> List[config.DrumZone]:
        """Translate / scale configured zones to follow the player."""
        pf = self.player_frame()
        if pf is None:
            return zones
        sx = pf["shoulder_w"] / 0.35     # baseline shoulder width in config
        sy = pf["torso_h"] / 0.25        # baseline torso height
        dx = pf["hip_cx"] - 0.5
        dy = pf["hip_cy"] - 0.7
        rescaled: List[config.DrumZone] = []
        for z in zones:
            x0 = (z.x_range[0] - 0.5) * sx + 0.5 + dx
            x1 = (z.x_range[1] - 0.5) * sx + 0.5 + dx
            y0 = (z.y_range[0] - 0.7) * sy + 0.7 + dy
            y1 = (z.y_range[1] - 0.7) * sy + 0.7 + dy
            rescaled.append(config.DrumZone(
                name=z.name,
                x_range=(float(np.clip(x0, 0, 1)), float(np.clip(x1, 0, 1))),
                y_range=(float(np.clip(y0, 0, 1)), float(np.clip(y1, 0, 1))),
                z_range=z.z_range,
                midi_note=z.midi_note,
                color_bgr=z.color_bgr,
                is_pedal=z.is_pedal,
                stem_group=z.stem_group,
            ))
        return rescaled

    # ------------------------------------------------------------------
    # Misc
    # ------------------------------------------------------------------
    def close(self) -> None:
        try:
            self._pose.close()
        except Exception:  # noqa: BLE001
            pass
