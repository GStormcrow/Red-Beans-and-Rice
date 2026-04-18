"""
airdrums.ui.overlay
===================
OpenCV-based HUD and drum line renderer for AirDrums V2.

Draws all UI elements directly onto numpy frames using cv2:
  - Semi-transparent drum lines with hit flash animations
  - Both virtual drumsticks
  - HUD bar (BPM, status, indicators)
  - Loop progress bar
  - Controls panel with keyboard shortcuts
  - FPS counter
  - Edit mode with drag handles for repositioning drum lines
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
    """Tracks an active hit flash for one drum line.

    Attributes:
        drum_name: Name of the drum line that was hit.
        triggered_at_ms: Monotonic time in milliseconds when the hit occurred.
        band_idx: Velocity band index (0=ghost … 4=accent) for scaling duration.
    """

    drum_name: str
    triggered_at_ms: float
    band_idx: int


class Overlay:
    """OpenCV-based HUD renderer for the AirDrums live view.

    Renders all UI elements onto a copy of the webcam frame using pure
    OpenCV + numpy — no PyQt5 or other GUI toolkit required.
    """

    def __init__(
        self,
        drum_lines: List[config.DrumLine],
        frame_w: int,
        frame_h: int,
    ) -> None:
        """Initialise the overlay renderer.

        Args:
            drum_lines: Ordered list of :class:`config.DrumLine` objects to render.
            frame_w: Frame width in pixels.
            frame_h: Frame height in pixels.
        """
        self.drum_lines: List[config.DrumLine] = drum_lines
        self.frame_w = frame_w
        self.frame_h = frame_h

        self._flashes: List[FlashState] = []
        self._edit_mode: bool = False
        self._controls_visible: bool = True
        self._controls_start: float = time.monotonic()
        self._drag_line: Optional[str] = None   # name of line being dragged
        self._drag_handle: Optional[str] = None  # "center", "left", or "right"

        log.debug(
            "Overlay initialised (%dx%d, %d lines)",
            frame_w,
            frame_h,
            len(drum_lines),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def register_hit(self, drum_name: str, band_idx: int) -> None:
        """Record a hit flash for the named drum line.

        Args:
            drum_name: Must match a :attr:`config.DrumLine.name` in drum_lines.
            band_idx: Velocity band index (0–4) used to scale flash duration.
        """
        now_ms = time.monotonic() * 1000.0
        self._flashes.append(FlashState(drum_name, now_ms, band_idx))
        log.debug("register_hit: %s (band=%d)", drum_name, band_idx)

    def draw(
        self,
        frame: np.ndarray,
        fingertips: dict,
        hud_data: dict,
    ) -> np.ndarray:
        """Render all HUD elements onto *frame* and return the annotated copy.

        Rendering order:
        1. Drum lines (semi-transparent + flash effects)
        2. Fingertip dots (landmark 8, one per detected hand)
        3. HUD bar (top strip)
        4. Loop progress bar (bottom)
        5. Controls panel (if visible and not auto-hidden)
        6. FPS counter (top-right corner)
        7. Edit mode indicator

        Args:
            frame: Raw BGR ``uint8`` webcam frame.
            fingertips: Dict mapping hand_id (``"left"``/``"right"``) to
                normalised ``(x, y)`` of INDEX_FINGER_TIP (landmark 8).
            hud_data: Dictionary with optional keys: ``bpm``, ``stability``,
                ``drift``, ``recording``, ``overdub``, ``playing``,
                ``take_count``, ``midi_connected``, ``depth_active``,
                ``mirror_mode``, ``fps``, ``loop_progress`` (float 0-1).

        Returns:
            Annotated BGR ``uint8`` numpy array (same shape as *frame*).
        """
        out = frame.copy()
        now_ms = time.monotonic() * 1000.0

        self._draw_drum_lines(out, fingertips, now_ms)
        self._draw_fingertips(out, fingertips)
        self._draw_hud(out, hud_data)
        self._draw_loop_bar(out, float(hud_data.get("loop_progress", 0.0)))

        elapsed_controls = time.monotonic() - self._controls_start
        if self._controls_visible and elapsed_controls < config.CONTROLS_AUTO_HIDE_S:
            self._draw_controls_panel(out)

        fps_val = float(hud_data.get("fps", 0.0))
        cv2.putText(
            out,
            f"FPS {fps_val:.1f}",
            (self.frame_w - 110, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (200, 200, 200),
            1,
            cv2.LINE_AA,
        )

        if self._edit_mode:
            self._draw_edit_indicator(out)

        return out

    def toggle_edit_mode(self) -> None:
        """Toggle the interactive edit mode on or off."""
        self._edit_mode = not self._edit_mode
        log.debug("Edit mode: %s", self._edit_mode)

    def toggle_controls(self) -> None:
        """Toggle the controls panel visibility and reset the auto-hide timer."""
        self._controls_visible = not self._controls_visible
        self._controls_start = time.monotonic()
        log.debug("Controls visible: %s", self._controls_visible)

    def handle_mouse(
        self,
        event: int,
        x: int,
        y: int,
        flags: int,
        param: object,
    ) -> None:
        """Handle OpenCV mouse events for edit-mode drag interactions.

        In edit mode, clicking near a drag handle (left edge, center, or
        right edge) of a drum line starts a drag operation.  Moving the mouse
        while dragging updates the line geometry in real time.

        Args:
            event: OpenCV mouse event constant (e.g. ``cv2.EVENT_LBUTTONDOWN``).
            x: Mouse x position in pixels.
            y: Mouse y position in pixels.
            flags: OpenCV event flags (unused).
            param: OpenCV event param (unused).
        """
        if not self._edit_mode:
            return

        w, h = self.frame_w, self.frame_h
        handle_radius = 10  # pixels

        if event == cv2.EVENT_LBUTTONDOWN:
            for dl in self.drum_lines:
                px = int(dl.x_center * w)
                py = int(dl.y_position * h)
                half_px = int(dl.half_width * w)
                handles = {
                    "center": (px, py),
                    "left": (px - half_px, py),
                    "right": (px + half_px, py),
                }
                for handle_name, (hx, hy) in handles.items():
                    if abs(x - hx) <= handle_radius and abs(y - hy) <= handle_radius:
                        self._drag_line = dl.name
                        self._drag_handle = handle_name
                        log.debug(
                            "Drag start: line=%s handle=%s", dl.name, handle_name
                        )
                        return

        elif event == cv2.EVENT_MOUSEMOVE and self._drag_line is not None:
            dl = next(
                (d for d in self.drum_lines if d.name == self._drag_line), None
            )
            if dl is None:
                return
            nx = max(0.0, min(1.0, x / w))
            ny = max(0.0, min(1.0, y / h))
            if self._drag_handle == "center":
                dl.x_center = nx
                dl.y_position = ny
            elif self._drag_handle == "left":
                # Moving left handle changes x_center and half_width together
                new_half = max(0.01, dl.x_center - nx)
                dl.half_width = min(new_half, 0.49)
            elif self._drag_handle == "right":
                new_half = max(0.01, nx - dl.x_center)
                dl.half_width = min(new_half, 0.49)

        elif event == cv2.EVENT_LBUTTONUP:
            log.debug(
                "Drag end: line=%s handle=%s", self._drag_line, self._drag_handle
            )
            self._drag_line = None
            self._drag_handle = None

    def update_drum_lines(self, drum_lines: List[config.DrumLine]) -> None:
        """Replace the current drum line list.

        Args:
            drum_lines: New list of :class:`config.DrumLine` objects.
        """
        self.drum_lines = drum_lines
        log.debug("Drum lines updated (%d lines)", len(drum_lines))

    # ------------------------------------------------------------------
    # Private rendering helpers
    # ------------------------------------------------------------------

    def _draw_fingertips(self, frame: np.ndarray, fingertips: dict) -> None:
        """Draw a small coloured dot at each detected INDEX_FINGER_TIP position.

        Args:
            frame: BGR ``uint8`` array; drawn in-place.
            fingertips: Dict mapping ``"left"``/``"right"`` to normalised ``(x, y)``.
        """
        colors = {"left": (255, 100, 50), "right": (50, 100, 255)}
        for hand_id, (nx, ny) in fingertips.items():
            px = int(nx * self.frame_w)
            py = int(ny * self.frame_h)
            color = colors.get(hand_id, (255, 255, 255))
            cv2.circle(frame, (px, py), 8, (255, 255, 255), 2)
            cv2.circle(frame, (px, py), 5, color, -1)

    def _draw_drum_lines(
        self,
        frame: np.ndarray,
        fingertips: dict,
        now_ms: float,
    ) -> None:
        """Render each drum line with opacity, flash, and optional edit handles.

        Args:
            frame: BGR ``uint8`` array; drawn in-place.
            fingertips: Dict mapping hand_id to normalised ``(x, y)`` of landmark 8.
            now_ms: Current monotonic time in milliseconds.
        """
        w, h = self.frame_w, self.frame_h
        expired_names: List[str] = []

        for dl in self.drum_lines:
            px = int(dl.x_center * w)
            py = int(dl.y_position * h)
            half_px = int(dl.half_width * w)

            # --- Opacity: near -> full, rest -> dimmed ----------------------
            is_near = any(
                abs(ny - dl.y_position) < config.NEAR_THRESHOLD
                for (_, ny) in fingertips.values()
            )
            opacity = (
                config.OVERLAY_OPACITY_NEAR if is_near else config.OVERLAY_OPACITY_REST
            )

            # --- Flash state ------------------------------------------------
            active_flash: Optional[FlashState] = None
            for fs in self._flashes:
                if fs.drum_name == dl.name:
                    flash_dur = config.FLASH_DURATION_MS * (1 + fs.band_idx * 0.5)
                    if now_ms - fs.triggered_at_ms <= flash_dur:
                        active_flash = fs
                    else:
                        expired_names.append(dl.name)

            # --- Draw flash glow (wider, brighter) --------------------------
            if active_flash is not None:
                flash_dur = config.FLASH_DURATION_MS * (1 + active_flash.band_idx * 0.5)
                age = now_ms - active_flash.triggered_at_ms
                flash_t = max(0.0, 1.0 - age / flash_dur)

                glow_overlay = frame.copy()
                b, g, r = dl.color_bgr
                bright = (
                    min(255, int(b + (255 - b) * flash_t * 0.6)),
                    min(255, int(g + (255 - g) * flash_t * 0.6)),
                    min(255, int(r + (255 - r) * flash_t * 0.6)),
                )
                glow_thickness = 3 + int(flash_t * 6)
                extend = int(flash_t * half_px * 0.3)
                cv2.line(
                    glow_overlay,
                    (px - half_px - extend, py),
                    (px + half_px + extend, py),
                    bright,
                    glow_thickness,
                    lineType=cv2.LINE_AA,
                )
                cv2.addWeighted(glow_overlay, flash_t * 0.9, frame, 1 - flash_t * 0.9, 0, frame)

            # --- Semi-transparent base line ---------------------------------
            line_overlay = frame.copy()
            cv2.line(
                line_overlay,
                (px - half_px, py),
                (px + half_px, py),
                dl.color_bgr,
                thickness=3,
                lineType=cv2.LINE_AA,
            )
            cv2.addWeighted(line_overlay, opacity, frame, 1 - opacity, 0, frame)

            # --- Label text -------------------------------------------------
            label_x = max(0, px - half_px)
            label_y = max(12, py - 6)
            cv2.putText(
                frame,
                dl.label,
                (label_x, label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                dl.color_bgr,
                1,
                cv2.LINE_AA,
            )

            # --- Edit handles -----------------------------------------------
            if self._edit_mode:
                handle_positions = {
                    "left": (px - half_px, py),
                    "center": (px, py),
                    "right": (px + half_px, py),
                }
                for handle_name, (hx, hy) in handle_positions.items():
                    fill = (255, 255, 0) if self._drag_line == dl.name and self._drag_handle == handle_name else (200, 200, 200)
                    cv2.circle(frame, (hx, hy), 8, fill, -1, lineType=cv2.LINE_AA)
                    cv2.circle(frame, (hx, hy), 8, (50, 50, 50), 1, lineType=cv2.LINE_AA)

        # Remove expired flashes
        self._flashes = [
            fs for fs in self._flashes if fs.drum_name not in expired_names
        ]

    def _draw_hud(self, frame: np.ndarray, hud_data: dict) -> None:
        """Render the top HUD bar with BPM, status, and mode indicators.

        The bar is a semi-transparent dark rectangle occupying the top 40px of
        the frame.  Items rendered left-to-right:
        - BPM value, stability badge, drift arrow
        - REC / OVR / PLAY indicators (coloured dots)
        - Take count
        - MIDI connectivity dot
        - Depth toggle ("D:ON" / "D:OFF")
        - Mirror mode ("L-MODE" / "R-MODE")
        - Edit mode lock icon

        Args:
            frame: BGR ``uint8`` array; drawn in-place.
            hud_data: See :meth:`draw` for key documentation.
        """
        w, h = self.frame_w, self.frame_h
        bar_h = 40

        bar_overlay = frame.copy()
        cv2.rectangle(bar_overlay, (0, 0), (w, bar_h), (15, 15, 20), -1)
        cv2.addWeighted(bar_overlay, 0.70, frame, 0.30, 0, frame)

        font = cv2.FONT_HERSHEY_SIMPLEX
        fg = (220, 220, 220)
        dim = (120, 120, 130)
        accent = (80, 200, 255)

        # BPM
        bpm = float(hud_data.get("bpm", 0.0))
        stab = float(hud_data.get("stability", 0.0))
        drift_raw = str(hud_data.get("drift", "steady"))
        drift_arrow = {"rushing": "^", "dragging": "v", "steady": ">"}.get(
            drift_raw.lower(), ">"
        )
        cv2.putText(
            frame, f"BPM {bpm:.1f}", (10, 27), font, 0.70, accent, 2, cv2.LINE_AA
        )
        cv2.putText(
            frame,
            f"+/-{stab:.1f} {drift_arrow}",
            (135, 27),
            font,
            0.50,
            dim,
            1,
            cv2.LINE_AA,
        )

        # REC / OVR / PLAY coloured dots
        dot_x = 285
        indicators = [
            ("REC", bool(hud_data.get("recording", False)), (50, 50, 220)),
            ("OVR", bool(hud_data.get("overdub", False)), (50, 180, 220)),
            ("PLAY", bool(hud_data.get("playing", False)), (50, 220, 80)),
        ]
        for label, active, color in indicators:
            dot_color = color if active else (55, 55, 65)
            cv2.circle(frame, (dot_x, 20), 6, dot_color, -1)
            cv2.putText(
                frame, label, (dot_x + 10, 25), font, 0.40, fg if active else dim, 1, cv2.LINE_AA
            )
            dot_x += 55

        # Take count
        takes = int(hud_data.get("take_count", 0))
        cv2.putText(
            frame, f"T:{takes}", (dot_x + 5, 27), font, 0.50, fg, 1, cv2.LINE_AA
        )
        dot_x += 55

        # MIDI indicator
        midi_ok = bool(hud_data.get("midi_connected", False))
        midi_color = (60, 220, 60) if midi_ok else (60, 60, 180)
        cv2.circle(frame, (dot_x + 5, 20), 6, midi_color, -1)
        cv2.putText(
            frame, "MIDI", (dot_x + 14, 25), font, 0.40, fg if midi_ok else dim, 1, cv2.LINE_AA
        )
        dot_x += 60

        # Depth indicator
        depth_on = bool(hud_data.get("depth_active", False))
        depth_label = "D:ON" if depth_on else "D:OFF"
        depth_color = accent if depth_on else dim
        cv2.putText(frame, depth_label, (dot_x + 5, 27), font, 0.45, depth_color, 1, cv2.LINE_AA)
        dot_x += 60

        # Mirror mode indicator
        mirrored = bool(hud_data.get("mirror_mode", False))
        mode_label = "L-MODE" if mirrored else "R-MODE"
        cv2.putText(frame, mode_label, (dot_x + 5, 27), font, 0.45, fg, 1, cv2.LINE_AA)

        # Edit mode lock icon (right side)
        if self._edit_mode:
            cv2.putText(
                frame, "[EDIT]", (w - 230, 27), font, 0.50, (0, 215, 255), 1, cv2.LINE_AA
            )

    def _draw_loop_bar(self, frame: np.ndarray, loop_progress: float) -> None:
        """Render a thin loop-progress bar at the bottom of the frame.

        The bar fills from left to right.  Color interpolates from blue (empty)
        to green (full) as *loop_progress* increases from 0 to 1.

        Args:
            frame: BGR ``uint8`` array; drawn in-place.
            loop_progress: Fill proportion in [0.0, 1.0].
        """
        w, h = self.frame_w, self.frame_h
        bar_y = h - 10
        bar_h = 6
        progress = max(0.0, min(1.0, loop_progress))

        # Background track
        cv2.rectangle(frame, (0, bar_y), (w, bar_y + bar_h), (30, 30, 40), -1)

        # Filled portion: blue -> green
        fill_w = int(w * progress)
        if fill_w > 0:
            t = progress
            b = int(200 * (1.0 - t))
            g = int(60 + 160 * t)
            r = 30
            cv2.rectangle(frame, (0, bar_y), (fill_w, bar_y + bar_h), (b, g, r), -1)

    def _draw_controls_panel(self, frame: np.ndarray) -> None:
        """Render the keyboard shortcuts reference panel on the right side.

        The panel is a semi-transparent dark rectangle approximately 220px wide
        listing all keyboard shortcuts available in the main loop.

        Args:
            frame: BGR ``uint8`` array; drawn in-place.
        """
        w, h = self.frame_w, self.frame_h
        panel_w = 220
        panel_x = w - panel_w - 4
        panel_y = 50

        shortcuts = [
            ("H", "Controls panel"),
            ("D", "Depth toggle"),
            ("M", "Handedness"),
            ("E", "Edit mode"),
            ("R", "Record"),
            ("O", "Overdub"),
            ("Space", "Play/pause"),
            ("Z", "Undo take"),
            ("S", "Settings"),
            ("Esc", "Exit"),
        ]

        row_h = 22
        panel_h = 14 + len(shortcuts) * row_h + 8
        panel_overlay = frame.copy()
        cv2.rectangle(
            panel_overlay,
            (panel_x, panel_y),
            (panel_x + panel_w, panel_y + panel_h),
            (12, 12, 20),
            -1,
        )
        cv2.addWeighted(panel_overlay, 0.78, frame, 0.22, 0, frame)

        cv2.putText(
            frame,
            "CONTROLS",
            (panel_x + 8, panel_y + 14),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (100, 180, 255),
            1,
            cv2.LINE_AA,
        )

        for i, (key, desc) in enumerate(shortcuts):
            ry = panel_y + 14 + (i + 1) * row_h
            cv2.putText(
                frame,
                key,
                (panel_x + 8, ry),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (80, 200, 255),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                desc,
                (panel_x + 58, ry),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                (190, 190, 200),
                1,
                cv2.LINE_AA,
            )

    def _draw_edit_indicator(self, frame: np.ndarray) -> None:
        """Draw a bottom-left edit mode status badge.

        Renders a small coloured badge so the user always knows edit mode is
        active, even when the controls panel is hidden.

        Args:
            frame: BGR ``uint8`` array; drawn in-place.
        """
        h = self.frame_h
        label = "-- EDIT MODE --"
        cv2.putText(
            frame,
            label,
            (10, h - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 215, 255),
            2,
            cv2.LINE_AA,
        )
