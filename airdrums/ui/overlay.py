"""
airdrums.ui.overlay
===================
Full OpenCV HUD / drum overlay renderer for AirDrums V2.

Optimized specifically for 640 x 480 resolution while preserving:
- handle_mouse()
- edit mode dragging
- controls panel
- hit flashes
- HUD indicators
- loop bar
- FPS counter
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import List, Optional

import cv2
import numpy as np

from .. import config

log = logging.getLogger(__name__)


@dataclass
class FlashState:
    drum_name: str
    triggered_at_ms: float
    band_idx: int


class Overlay:
    def __init__(
        self,
        drum_lines: List[config.DrumLine],
        frame_w: int = 640,
        frame_h: int = 480,
    ) -> None:

        self.drum_lines = drum_lines
        self.frame_w = frame_w
        self.frame_h = frame_h

        self._flashes: List[FlashState] = []

        self._edit_mode = False
        self._controls_visible = True
        self._controls_start = time.monotonic()

        self._drag_line: Optional[str] = None
        self._drag_handle: Optional[str] = None

    # ==========================================================
    # PUBLIC API
    # ==========================================================

    def register_hit(self, drum_name: str, band_idx: int) -> None:
        self._flashes.append(
            FlashState(
                drum_name,
                time.monotonic() * 1000.0,
                band_idx,
            )
        )

    def toggle_edit_mode(self) -> None:
        self._edit_mode = not self._edit_mode

    def toggle_controls(self) -> None:
        self._controls_visible = not self._controls_visible
        self._controls_start = time.monotonic()

    def update_drum_lines(self, drum_lines) -> None:
        self.drum_lines = drum_lines

    # ==========================================================
    # MOUSE INPUT
    # ==========================================================

    def handle_mouse(self, event, x, y, flags, param) -> None:
        if not self._edit_mode:
            return

        w = self.frame_w
        h = self.frame_h
        radius = 8

        if event == cv2.EVENT_LBUTTONDOWN:

            for dl in self.drum_lines:
                px = int(dl.x_center * w)
                py = int(dl.y_position * h)
                half = int(dl.half_width * w)

                handles = {
                    "center": (px, py),
                    "left": (px - half, py),
                    "right": (px + half, py),
                }

                for name, (hx, hy) in handles.items():
                    if abs(x - hx) <= radius and abs(y - hy) <= radius:
                        self._drag_line = dl.name
                        self._drag_handle = name
                        return

        elif event == cv2.EVENT_MOUSEMOVE and self._drag_line:

            dl = next(
                (d for d in self.drum_lines if d.name == self._drag_line),
                None,
            )

            if dl is None:
                return

            nx = max(0.0, min(1.0, x / w))
            ny = max(0.0, min(1.0, y / h))

            if self._drag_handle == "center":
                dl.x_center = nx
                dl.y_position = ny

            elif self._drag_handle == "left":
                dl.half_width = max(0.02, min(dl.x_center - nx, 0.45))

            elif self._drag_handle == "right":
                dl.half_width = max(0.02, min(nx - dl.x_center, 0.45))

        elif event == cv2.EVENT_LBUTTONUP:
            self._drag_line = None
            self._drag_handle = None

    # ==========================================================
    # MAIN DRAW
    # ==========================================================

    def draw(self, frame: np.ndarray, fingertips: dict, hud: dict):
        out = frame.copy()

        now_ms = time.monotonic() * 1000.0

        self._draw_drum_lines(out, fingertips, now_ms)
        self._draw_fingertips(out, fingertips)
        self._draw_hud(out, hud)
        self._draw_loop_bar(out, hud.get("loop_progress", 0.0))

        if self._controls_visible:
            elapsed = time.monotonic() - self._controls_start
            if elapsed < config.CONTROLS_AUTO_HIDE_S:
                self._draw_controls_panel(out)

        fps = float(hud.get("fps", 0.0))
        cv2.putText(
            out,
            f"{fps:.1f} FPS",
            (540, 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (220, 220, 220),
            1,
            cv2.LINE_AA,
        )

        if self._edit_mode:
            self._draw_edit_indicator(out)

        return out

    # ==========================================================
    # DRAW HELPERS
    # ==========================================================

    def _draw_fingertips(self, frame, fingertips):
        colors = {
            "left": (255, 120, 60),
            "right": (60, 120, 255),
        }

        for hand, (nx, ny) in fingertips.items():
            px = int(nx * self.frame_w)
            py = int(ny * self.frame_h)

            cv2.circle(frame, (px, py), 6, (255, 255, 255), 2)
            cv2.circle(frame, (px, py), 4, colors.get(hand, (255, 255, 255)), -1)

    def _draw_drum_lines(self, frame, fingertips, now_ms):

        expired = []

        for dl in self.drum_lines:

            px = int(dl.x_center * self.frame_w)
            py = int(dl.y_position * self.frame_h)
            half = int(dl.half_width * self.frame_w)

            near = any(
                abs(ny - dl.y_position) < config.NEAR_THRESHOLD
                for (_, ny) in fingertips.values()
            )

            opacity = (
                config.OVERLAY_OPACITY_NEAR
                if near
                else config.OVERLAY_OPACITY_REST
            )

            active = None

            for fs in self._flashes:
                if fs.drum_name == dl.name:
                    dur = config.FLASH_DURATION_MS * (1 + fs.band_idx * 0.5)

                    if now_ms - fs.triggered_at_ms <= dur:
                        active = fs
                    else:
                        expired.append(fs)

            if active:
                dur = config.FLASH_DURATION_MS * (1 + active.band_idx * 0.5)
                age = now_ms - active.triggered_at_ms
                t = max(0.0, 1.0 - age / dur)

                glow = frame.copy()

                cv2.line(
                    glow,
                    (px - half - 4, py),
                    (px + half + 4, py),
                    (255, 255, 255),
                    4,
                    cv2.LINE_AA,
                )

                cv2.addWeighted(glow, t * 0.65, frame, 1 - t * 0.65, 0, frame)

            overlay = frame.copy()

            cv2.line(
                overlay,
                (px - half, py),
                (px + half, py),
                dl.color_bgr,
                2,
                cv2.LINE_AA,
            )

            cv2.addWeighted(
                overlay,
                opacity,
                frame,
                1 - opacity,
                0,
                frame,
            )

            cv2.putText(
                frame,
                dl.label,
                (max(4, px - half), py - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                dl.color_bgr,
                1,
                cv2.LINE_AA,
            )

            if self._edit_mode:
                handles = {
                    "left": (px - half, py),
                    "center": (px, py),
                    "right": (px + half, py),
                }

                for name, pos in handles.items():
                    fill = (
                        (255, 255, 0)
                        if self._drag_line == dl.name
                        and self._drag_handle == name
                        else (200, 200, 200)
                    )

                    cv2.circle(frame, pos, 6, fill, -1, cv2.LINE_AA)
                    cv2.circle(frame, pos, 6, (50, 50, 50), 1, cv2.LINE_AA)

        self._flashes = [f for f in self._flashes if f not in expired]

    def _draw_hud(self, frame, hud):
        w = self.frame_w
        h = 28

        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (12, 12, 18), -1)
        cv2.addWeighted(overlay, 0.74, frame, 0.26, 0, frame)

        font = cv2.FONT_HERSHEY_SIMPLEX

        bpm = float(hud.get("bpm", 0.0))

        cv2.putText(
            frame,
            f"BPM {bpm:.1f}",
            (8, 18),
            font,
            0.46,
            (90, 210, 255),
            1,
            cv2.LINE_AA,
        )

        x = 118

        items = [
            ("REC", hud.get("recording", False), (50, 50, 220)),
            ("OVR", hud.get("overdub", False), (50, 180, 220)),
            ("PLY", hud.get("playing", False), (50, 220, 80)),
        ]

        for txt, state, color in items:
            c = color if state else (55, 55, 60)

            cv2.circle(frame, (x, 14), 4, c, -1)

            cv2.putText(
                frame,
                txt,
                (x + 7, 18),
                font,
                0.34,
                (220, 220, 220),
                1,
                cv2.LINE_AA,
            )

            x += 52

        takes = int(hud.get("take_count", 0))

        cv2.putText(
            frame,
            f"T:{takes}",
            (280, 18),
            font,
            0.36,
            (220, 220, 220),
            1,
            cv2.LINE_AA,
        )

        midi = hud.get("midi_connected", False)
        cv2.circle(
            frame,
            (335, 14),
            4,
            (60, 220, 60) if midi else (80, 80, 160),
            -1,
        )

        cv2.putText(
            frame,
            "MIDI",
            (343, 18),
            font,
            0.34,
            (220, 220, 220),
            1,
            cv2.LINE_AA,
        )

        depth = hud.get("depth_active", False)
        cv2.putText(
            frame,
            "D:ON" if depth else "D:OFF",
            (410, 18),
            font,
            0.34,
            (90, 210, 255) if depth else (120, 120, 120),
            1,
            cv2.LINE_AA,
        )

        mirror = hud.get("mirror_mode", False)

        cv2.putText(
            frame,
            "L-MODE" if mirror else "R-MODE",
            (475, 18),
            font,
            0.34,
            (220, 220, 220),
            1,
            cv2.LINE_AA,
        )

    def _draw_loop_bar(self, frame, progress):
        progress = max(0.0, min(1.0, progress))

        y = 472
        h = 5
        w = self.frame_w

        cv2.rectangle(frame, (0, y), (w, y + h), (30, 30, 40), -1)

        fill = int(w * progress)

        if fill > 0:
            b = int(180 * (1 - progress))
            g = int(60 + 180 * progress)

            cv2.rectangle(
                frame,
                (0, y),
                (fill, y + h),
                (b, g, 30),
                -1,
            )

    def _draw_controls_panel(self, frame):
        x = 430
        y = 36
        w = 205

        rows = [
            "H Controls",
            "D Depth",
            "M Mirror",
            "E Edit",
            "R Record",
            "O Overdub",
            "Space Play",
            "Z Undo",
            "Esc Exit",
        ]

        h = 18 * len(rows) + 18

        overlay = frame.copy()
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (12, 12, 18), -1)
        cv2.addWeighted(overlay, 0.82, frame, 0.18, 0, frame)

        cv2.putText(
            frame,
            "KEYS",
            (x + 8, y + 14),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            (90, 210, 255),
            1,
            cv2.LINE_AA,
        )

        for i, row in enumerate(rows):
            cv2.putText(
                frame,
                row,
                (x + 8, y + 32 + i * 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.34,
                (220, 220, 220),
                1,
                cv2.LINE_AA,
            )

    def _draw_edit_indicator(self, frame):
        cv2.putText(
            frame,
            "-- EDIT MODE --",
            (8, 458),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 215, 255),
            1,
            cv2.LINE_AA,
        )