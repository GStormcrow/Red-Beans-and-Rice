"""
airdrums.ui.overlay
===================
Renders the live HUD on top of the webcam feed using OpenCV:
  * Both virtual drumsticks (tapered, colored tip, velocity-reactive)
  * Semi-transparent drum zone pads with per-hit glow + ghost outlines
  * Full 33-point pose skeleton with velocity color shift
  * HUD text: BPM, BPM stability, hi-hat state, loop progress, status,
    take counter, MIDI indicator, FPS, depth toggle, drumstick slider.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .. import config
from ..tracking.skeleton import LANDMARK_NAMES, Joint

log = logging.getLogger(__name__)


# MediaPipe pose edge list (33-landmark model) as a compact graph.
POSE_EDGES: List[Tuple[int, int]] = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),     # arms
    (11, 23), (12, 24), (23, 24),                         # torso
    (23, 25), (25, 27), (27, 29), (27, 31),               # left leg
    (24, 26), (26, 28), (28, 30), (28, 32),               # right leg
    (9, 10), (0, 1), (1, 2), (2, 3), (3, 7),              # head
    (0, 4), (4, 5), (5, 6), (6, 8),
]


@dataclass
class FlashState:
    """Per-zone hit flash and per-stick strike flash timers."""
    zone_flash: Dict[str, float] = field(default_factory=dict)     # name -> end_time
    zone_velocity: Dict[str, int] = field(default_factory=dict)
    zone_rim: Dict[str, bool] = field(default_factory=dict)
    stick_flash: Dict[str, float] = field(default_factory=dict)    # side -> end_time


class Overlay:
    """The live HUD renderer."""

    FLASH_DURATION_S = 0.35

    def __init__(self, theme: str = config.DEFAULT_HUD_THEME):
        self.theme = theme if theme in config.HUD_THEMES else "dark"
        self.flash = FlashState()
        self.show_depth = False
        self.fps = 0.0
        self._fps_last = time.time()
        self._fps_count = 0
        self._palette = self._build_palette()

    # ------------------------------------------------------------------
    def _build_palette(self) -> Dict[str, Tuple[int, int, int]]:
        if self.theme == "light":
            return {"bg": (235, 235, 235), "fg": (20, 20, 20),
                    "accent": (20, 120, 255), "dim": (120, 120, 120)}
        if self.theme == "neon":
            return {"bg": (25, 0, 40), "fg": (220, 255, 240),
                    "accent": (220, 50, 255), "dim": (120, 180, 120)}
        return {"bg": (20, 20, 25), "fg": (230, 230, 230),
                "accent": (50, 180, 255), "dim": (110, 110, 120)}

    def set_theme(self, theme: str) -> None:
        if theme in config.HUD_THEMES:
            self.theme = theme
            self._palette = self._build_palette()

    # ------------------------------------------------------------------
    # Event hooks
    # ------------------------------------------------------------------
    def register_hit(self, zone_name: str, velocity: int, is_rimshot: bool = False) -> None:
        end = time.time() + self.FLASH_DURATION_S
        self.flash.zone_flash[zone_name] = end
        self.flash.zone_velocity[zone_name] = velocity
        self.flash.zone_rim[zone_name] = is_rimshot

    def register_stick_strike(self, side: str) -> None:
        self.flash.stick_flash[side] = time.time() + self.FLASH_DURATION_S

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------
    def render(self, frame: np.ndarray,
               skeleton,
               drumsticks,
               zones: List[config.DrumZone],
               hud_state: Dict[str, object],
               depth_map: Optional[np.ndarray] = None) -> np.ndarray:
        """Main render entry. ``hud_state`` aggregates BPM, status, counts, etc.

        Returns the annotated frame (edits in place, but also returned)."""
        import cv2
        self._tick_fps()

        if self.show_depth and depth_map is not None:
            frame = self._composite_depth(frame, depth_map)

        self._draw_zones(frame, zones)
        self._draw_skeleton(frame, skeleton)
        for stick in drumsticks:
            flash = max(0.0, self.flash.stick_flash.get(stick.side, 0.0) - time.time())
            stick.draw(frame, strike_flash=min(1.0, flash / self.FLASH_DURATION_S))
        self._draw_hud(frame, hud_state)
        return frame

    # ------------------------------------------------------------------
    def _draw_zones(self, frame: np.ndarray, zones: List[config.DrumZone]) -> None:
        import cv2
        h, w = frame.shape[:2]
        overlay = frame.copy()
        now = time.time()
        for z in zones:
            x0 = int(z.x_range[0] * w); x1 = int(z.x_range[1] * w)
            y0 = int(z.y_range[0] * h); y1 = int(z.y_range[1] * h)
            flash_end = self.flash.zone_flash.get(z.name, 0.0)
            is_flashing = now < flash_end
            vel = self.flash.zone_velocity.get(z.name, 80)
            alpha = 0.22
            if is_flashing:
                t = (flash_end - now) / self.FLASH_DURATION_S
                alpha = 0.22 + 0.55 * t * (vel / 127.0)
            # Ghost outline when no flash
            color = z.color_bgr
            cv2.rectangle(overlay, (x0, y0), (x1, y1), color, thickness=-1)
            if is_flashing:
                if self.flash.zone_rim.get(z.name, False):
                    cv2.rectangle(overlay, (x0 - 4, y0 - 4), (x1 + 4, y1 + 4),
                                  (255, 255, 255), thickness=3)
                else:
                    cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
                    r = int(max(x1 - x0, y1 - y0) * 0.6 * (1.0 - (flash_end - now) / self.FLASH_DURATION_S))
                    cv2.circle(overlay, (cx, cy), max(1, r), (255, 255, 255), 2)

            cv2.rectangle(frame, (x0, y0), (x1, y1), color, 2)
            # Label
            cv2.putText(frame, z.name, (x0 + 4, y0 + 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (240, 240, 240), 1, cv2.LINE_AA)
        cv2.addWeighted(overlay, alpha := 0.35, frame, 1 - alpha, 0, dst=frame)

    # ------------------------------------------------------------------
    def _draw_skeleton(self, frame: np.ndarray, skeleton) -> None:
        import cv2
        if skeleton is None:
            return
        h, w = frame.shape[:2]
        joints = skeleton.joints
        name_by_idx = {idx: name for name, idx in LANDMARK_NAMES.items()}

        # Edges
        for a, b in POSE_EDGES:
            na = name_by_idx.get(a); nb = name_by_idx.get(b)
            if not na or not nb:
                continue
            ja = joints.get(na); jb = joints.get(nb)
            if ja is None or jb is None or not ja.visible or not jb.visible:
                continue
            pa = (int(ja.x * w), int(ja.y * h))
            pb = (int(jb.x * w), int(jb.y * h))
            cv2.line(frame, pa, pb, (180, 180, 180), 2, lineType=cv2.LINE_AA)

        # Nodes colored by speed
        for name, j in joints.items():
            if not j.visible:
                continue
            p = (int(j.x * w), int(j.y * h))
            c = _speed_color(j.speed)
            r = 6 if name in ("left_wrist", "right_wrist") else 3
            cv2.circle(frame, p, r, c, -1, lineType=cv2.LINE_AA)

    # ------------------------------------------------------------------
    def _draw_hud(self, frame: np.ndarray, state: Dict[str, object]) -> None:
        import cv2
        h, w = frame.shape[:2]
        pal = self._palette

        # Top bar
        bar_h = 44
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, bar_h), pal["bg"], -1)
        cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, dst=frame)

        bpm = float(state.get("bpm", 0.0))
        stab = float(state.get("stability", 0.0))
        drift = str(state.get("drift", "steady"))
        status = str(state.get("status", "idle"))
        take = int(state.get("take", 0))
        hihat = str(state.get("hihat", "open"))
        midi_ok = bool(state.get("midi_connected", False))
        stick_len = float(state.get("stick_length", config.DRUMSTICK_LENGTH_DEFAULT))
        loop_pct = float(state.get("loop_progress", 0.0))

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, f"BPM {bpm:0.1f}", (10, 28), font, 0.7, pal["accent"], 2, cv2.LINE_AA)
        cv2.putText(frame, f"+/-{stab:0.1f} {drift}", (130, 28), font, 0.55, pal["dim"], 1, cv2.LINE_AA)
        cv2.putText(frame, f"HiHat:{hihat}", (290, 28), font, 0.55, pal["fg"], 1, cv2.LINE_AA)
        cv2.putText(frame, f"Status:{status}", (420, 28), font, 0.55, pal["fg"], 1, cv2.LINE_AA)
        cv2.putText(frame, f"Take {take}", (560, 28), font, 0.55, pal["fg"], 1, cv2.LINE_AA)

        dot_color = (80, 220, 80) if midi_ok else (80, 80, 220)
        cv2.circle(frame, (w - 50, 22), 6, dot_color, -1)
        cv2.putText(frame, "MIDI", (w - 40, 28), font, 0.5, pal["fg"], 1, cv2.LINE_AA)

        # Loop progress bar along bottom
        lp_y = h - 28
        cv2.rectangle(frame, (10, lp_y), (w - 10, lp_y + 10), pal["bg"], -1)
        cv2.rectangle(frame, (10, lp_y), (10 + int((w - 20) * max(0.0, min(1.0, loop_pct))), lp_y + 10),
                      pal["accent"], -1)

        # FPS + stick length slider
        cv2.putText(frame, f"FPS {self.fps:0.1f}", (10, h - 40), font, 0.5, pal["fg"], 1, cv2.LINE_AA)
        cv2.putText(frame, f"stick {stick_len:0.2f}", (120, h - 40), font, 0.5, pal["fg"], 1, cv2.LINE_AA)
        # Slider track
        sx0, sx1 = 220, 360
        sy = h - 43
        cv2.line(frame, (sx0, sy), (sx1, sy), pal["dim"], 2)
        t = ((stick_len - config.DRUMSTICK_LENGTH_MIN)
             / (config.DRUMSTICK_LENGTH_MAX - config.DRUMSTICK_LENGTH_MIN))
        sx = int(sx0 + (sx1 - sx0) * max(0.0, min(1.0, t)))
        cv2.circle(frame, (sx, sy), 6, pal["accent"], -1)

        if self.show_depth:
            cv2.putText(frame, "DEPTH", (w - 90, h - 40), font, 0.5, pal["accent"], 2, cv2.LINE_AA)

    # ------------------------------------------------------------------
    def _composite_depth(self, frame: np.ndarray, depth: np.ndarray) -> np.ndarray:
        import cv2
        d = cv2.resize(depth, (frame.shape[1] // 3, frame.shape[0] // 3))
        d = (d * 255).clip(0, 255).astype(np.uint8)
        d = cv2.applyColorMap(d, cv2.COLORMAP_INFERNO)
        y0 = 50
        x0 = frame.shape[1] - d.shape[1] - 10
        roi = frame[y0:y0 + d.shape[0], x0:x0 + d.shape[1]]
        if roi.shape == d.shape:
            frame[y0:y0 + d.shape[0], x0:x0 + d.shape[1]] = cv2.addWeighted(roi, 0.1, d, 0.9, 0)
        return frame

    # ------------------------------------------------------------------
    def _tick_fps(self) -> None:
        self._fps_count += 1
        now = time.time()
        if now - self._fps_last >= 0.5:
            self.fps = self._fps_count / (now - self._fps_last)
            self._fps_count = 0
            self._fps_last = now


def _speed_color(speed: float) -> Tuple[int, int, int]:
    """Green -> yellow -> red gradient based on scalar speed."""
    s = max(0.0, min(1.0, speed / 4.0))
    if s < 0.5:
        t = s / 0.5
        return (0, int(255), int(255 * t))
    t = (s - 0.5) / 0.5
    return (0, int(255 * (1 - t)), 255)
