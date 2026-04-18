"""
airdrums.recording.bvh
======================
Export skeleton keyframes to BVH motion capture.

We map MediaPipe's 33-landmark skeleton onto a simplified standard humanoid
BVH hierarchy (Hips -> Spine -> Neck+Head, Hips -> LeftUpLeg/RightUpLeg
chains, Spine -> LeftShoulder/RightShoulder chains). Rotations are derived
from pairwise landmark direction vectors; offsets come from the first frame
so the skeleton scales to the captured player.
"""
from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from .. import config

log = logging.getLogger(__name__)


# Bone chain: (bvh_joint_name, parent_bvh_joint, landmark_from, landmark_to)
# ``landmark_from`` is the parent landmark that sits at this bone's origin;
# ``landmark_to`` is the child landmark used to compute the bone vector.
BONES: List[Tuple[str, str, str, str]] = [
    # Root + spine
    ("Hips",          "",              "hip_center",    "spine"),
    ("Spine",         "Hips",          "spine",         "chest"),
    ("Chest",         "Spine",         "chest",         "neck"),
    ("Neck",          "Chest",         "neck",          "nose"),
    ("Head",          "Neck",          "nose",          "nose"),
    # Left arm
    ("LeftShoulder",  "Chest",         "left_shoulder", "left_elbow"),
    ("LeftArm",       "LeftShoulder",  "left_elbow",    "left_wrist"),
    ("LeftHand",      "LeftArm",       "left_wrist",    "left_wrist"),
    # Right arm
    ("RightShoulder", "Chest",         "right_shoulder","right_elbow"),
    ("RightArm",      "RightShoulder", "right_elbow",   "right_wrist"),
    ("RightHand",     "RightArm",      "right_wrist",   "right_wrist"),
    # Left leg
    ("LeftUpLeg",     "Hips",          "left_hip",      "left_knee"),
    ("LeftLeg",       "LeftUpLeg",     "left_knee",     "left_ankle"),
    ("LeftFoot",      "LeftLeg",       "left_ankle",    "left_foot_index"),
    # Right leg
    ("RightUpLeg",    "Hips",          "right_hip",     "right_knee"),
    ("RightLeg",      "RightUpLeg",    "right_knee",    "right_ankle"),
    ("RightFoot",     "RightLeg",      "right_ankle",   "right_foot_index"),
]


def _derived_points(joints: Dict[str, Tuple[float, float, float, float]]
                    ) -> Dict[str, Tuple[float, float, float]]:
    """Compute synthetic landmarks (hip_center, spine, chest, neck)."""
    def g(n): return joints.get(n)
    out: Dict[str, Tuple[float, float, float]] = {
        k: (v[0], v[1], v[2]) for k, v in joints.items() if v[3] >= config.VISIBILITY_THRESHOLD
    }
    lh, rh = g("left_hip"), g("right_hip")
    ls, rs = g("left_shoulder"), g("right_shoulder")
    if lh and rh:
        out["hip_center"] = ((lh[0] + rh[0]) / 2, (lh[1] + rh[1]) / 2, (lh[2] + rh[2]) / 2)
    if ls and rs:
        out["neck"] = ((ls[0] + rs[0]) / 2, (ls[1] + rs[1]) / 2, (ls[2] + rs[2]) / 2)
    if "hip_center" in out and "neck" in out:
        hc, nk = out["hip_center"], out["neck"]
        out["spine"] = ((hc[0] * 2 + nk[0]) / 3, (hc[1] * 2 + nk[1]) / 3, (hc[2] * 2 + nk[2]) / 3)
        out["chest"] = ((hc[0] + nk[0] * 2) / 3, (hc[1] + nk[1] * 2) / 3, (hc[2] + nk[2] * 2) / 3)
    return out


def _direction(a: Tuple[float, float, float], b: Tuple[float, float, float]
               ) -> Tuple[float, float, float]:
    dx, dy, dz = b[0] - a[0], b[1] - a[1], b[2] - a[2]
    n = math.sqrt(dx * dx + dy * dy + dz * dz) + 1e-9
    return (dx / n, dy / n, dz / n)


def _euler_from_dir(d: Tuple[float, float, float]) -> Tuple[float, float, float]:
    """Approximate ZXY Euler angles from a unit direction vector."""
    z_rot = math.degrees(math.atan2(d[0], -d[1]))
    x_rot = math.degrees(math.asin(max(-1.0, min(1.0, d[2]))))
    y_rot = 0.0
    return (z_rot, x_rot, y_rot)


def _offsets(first_frame: Dict[str, Tuple[float, float, float]]) -> Dict[str, Tuple[float, float, float]]:
    """Compute per-bone offsets (in cm-ish units) from the first frame."""
    out: Dict[str, Tuple[float, float, float]] = {}
    scale = 100.0   # normalized -> pseudo-cm
    for bone, parent, a_name, b_name in BONES:
        a = first_frame.get(a_name)
        b = first_frame.get(b_name)
        if a is None or b is None:
            out[bone] = (0.0, 10.0, 0.0)
            continue
        out[bone] = ((b[0] - a[0]) * scale, (b[1] - a[1]) * scale, (b[2] - a[2]) * scale)
    return out


def _write_hierarchy(lines: List[str], offsets: Dict[str, Tuple[float, float, float]]) -> None:
    """Emit the HIERARCHY block. 6 channels for the root, 3 for the rest."""
    # Build parent -> children map
    children: Dict[str, List[str]] = {"": []}
    for bone, parent, _, _ in BONES:
        children.setdefault(parent, []).append(bone)
        children.setdefault(bone, [])

    def emit(bone: str, depth: int) -> None:
        indent = "  " * depth
        is_root = depth == 0
        ox, oy, oz = offsets.get(bone, (0.0, 0.0, 0.0))
        header = "ROOT" if is_root else "JOINT"
        lines.append(f"{indent}{header} {bone}")
        lines.append(f"{indent}{{")
        lines.append(f"{indent}  OFFSET {ox:.4f} {oy:.4f} {oz:.4f}")
        if is_root:
            lines.append(f"{indent}  CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation")
        else:
            lines.append(f"{indent}  CHANNELS 3 Zrotation Xrotation Yrotation")
        kids = children.get(bone, [])
        if not kids:
            lines.append(f"{indent}  End Site")
            lines.append(f"{indent}  {{")
            lines.append(f"{indent}    OFFSET 0.0000 5.0000 0.0000")
            lines.append(f"{indent}  }}")
        else:
            for c in kids:
                emit(c, depth + 1)
        lines.append(f"{indent}}}")

    lines.append("HIERARCHY")
    emit("Hips", 0)


def export_bvh(session, path: str | Path) -> Path:
    """Write a .bvh file for the session's recorded skeleton keyframes.

    Importable into Blender, Maya and MotionBuilder.
    """
    path = Path(path)
    keyframes = session.skeleton_keyframes
    if not keyframes:
        raise ValueError("Session has no skeleton keyframes to export")

    first = _derived_points(keyframes[0].joints)
    offsets = _offsets(first)

    lines: List[str] = []
    _write_hierarchy(lines, offsets)

    # MOTION block
    n_frames = len(keyframes)
    if n_frames >= 2:
        dt_ms = (keyframes[-1].timestamp_ms - keyframes[0].timestamp_ms) / (n_frames - 1)
    else:
        dt_ms = 1000.0 / config.SKELETON_KEYFRAME_FPS
    frame_time = max(dt_ms / 1000.0, 1.0 / config.SKELETON_KEYFRAME_FPS)

    lines.append("MOTION")
    lines.append(f"Frames: {n_frames}")
    lines.append(f"Frame Time: {frame_time:.6f}")

    for kf in keyframes:
        pts = _derived_points(kf.joints)
        hip = pts.get("hip_center", (0.5, 0.5, 0.0))
        row: List[str] = [
            f"{(hip[0] - 0.5) * 100:.4f}",
            f"{(0.5 - hip[1]) * 100:.4f}",
            f"{hip[2] * 100:.4f}",
        ]
        # Root rotation uses spine direction
        spine_dir = _direction(pts.get("hip_center", (0, 0, 0)), pts.get("neck", (0, 1, 0)))
        zr, xr, yr = _euler_from_dir(spine_dir)
        row.extend([f"{zr:.3f}", f"{xr:.3f}", f"{yr:.3f}"])

        # Remaining joints in declared order (skipping Hips we already emitted root channels for)
        for bone, parent, a_name, b_name in BONES[1:]:
            a = pts.get(a_name)
            b = pts.get(b_name)
            if a is None or b is None or a == b:
                row.extend(["0.000", "0.000", "0.000"])
                continue
            zr, xr, yr = _euler_from_dir(_direction(a, b))
            row.extend([f"{zr:.3f}", f"{xr:.3f}", f"{yr:.3f}"])

        lines.append(" ".join(row))

    path.write_text("\n".join(lines) + "\n")
    log.info("Wrote BVH: %s (%d frames, dt=%.4fs)", path, n_frames, frame_time)
    return path
