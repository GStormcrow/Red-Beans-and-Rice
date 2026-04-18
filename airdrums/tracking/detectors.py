"""
airdrums.tracking.detectors
===========================
V2 strike detection via horizontal line-crossing.

``LineCrossDetector`` monitors whether a drumstick tip has crossed downward
through each drum-line's y threshold while within its x bounds, then maps
tip speed to a MIDI velocity band and applies per-line cooldowns.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from .. import config

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# HitResult
# ---------------------------------------------------------------------------
@dataclass
class HitResult:
    """Encapsulates a single detected drum hit.

    Attributes:
        drum_line: The :class:`config.DrumLine` that was crossed.
        velocity:  MIDI velocity in [1, 127].
        band_idx:  Index into ``config.VELOCITY_BANDS`` (0 = ghost … 4 = accent).
        hand_id:   ``"left"`` or ``"right"``.
    """

    drum_line: config.DrumLine
    velocity: int
    band_idx: int
    hand_id: str


# ---------------------------------------------------------------------------
# LineCrossDetector
# ---------------------------------------------------------------------------
class LineCrossDetector:
    """Detects downward crossings of drum-line thresholds by a stick tip.

    One instance is created per hand.  Both hands operate independently with
    no shared state; the caller is responsible for handling results from both.

    The detection algorithm for each drum line per frame:

    1. Confirm tip_x is within the line's x band.
    2. Confirm tip_vy > 0 (downward motion — upstrokes never trigger).
    3. Confirm tip_y crossed from above (prev_y < y_position) to at-or-below
       (tip_y >= y_position) since the last frame.
    4. Classify tip_speed with :func:`config.classify_velocity`.
    5. Check the per-band cooldown has elapsed since the last hit on this line.
    6. Emit :class:`HitResult` and update internal state.
    """

    def __init__(self, hand_id: str) -> None:
        """Create a detector for one hand.

        Args:
            hand_id: ``"left"`` or ``"right"``.
        """
        if hand_id not in ("left", "right"):
            raise ValueError(f"hand_id must be 'left' or 'right', got {hand_id!r}")
        self.hand_id = hand_id

        # Per drum-line state: name -> (last_tip_y, last_trigger_time)
        self._state: Dict[str, Tuple[float, float]] = {}

        log.debug("LineCrossDetector created (hand=%s)", hand_id)

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------
    def check(
        self,
        tip_x: float,
        tip_y: float,
        tip_vy: float,
        tip_speed: float,
        drum_lines: list,  # list[config.DrumLine]
    ) -> Optional[HitResult]:
        """Check whether the tip crossed any drum line downward this frame.

        Args:
            tip_x: Normalised x of the stick tip.
            tip_y: Normalised y of the stick tip (0 = top, 1 = bottom).
            tip_vy: Vertical velocity of the tip (positive = moving downward).
            tip_speed: Scalar tip speed in normalised units per second.
            drum_lines: List of :class:`config.DrumLine` objects to test.

        Returns:
            A :class:`HitResult` for the first line crossed this frame, or
            ``None`` if no crossing occurred.
        """
        now = time.monotonic()

        for dl in drum_lines:
            line_name = dl.name

            # Retrieve previous tip_y for this line (default: tip_y so no
            # crossing is reported before we have a previous position).
            prev_y, last_trigger_time = self._state.get(
                line_name, (tip_y, 0.0)
            )

            # --- Gate 1: x must be within the line's horizontal band ----------
            x_lo = dl.x_center - dl.half_width
            x_hi = dl.x_center + dl.half_width
            within_x = x_lo <= tip_x <= x_hi

            # --- Gate 2: downward motion only ---------------------------------
            moving_down = tip_vy > 0.0

            # --- Gate 3: tip crossed from above to at/below the line ----------
            crossed = (prev_y < dl.y_position) and (tip_y >= dl.y_position)

            if within_x and moving_down and crossed:
                # --- Gate 4: velocity band ------------------------------------
                # Normalise speed to [0, 1] by clamping at the highest band max
                max_speed = config.VELOCITY_BANDS[-1].speed_max
                norm_speed = min(tip_speed / max_speed, 1.0) if max_speed > 0 else 0.0
                band_idx = config.classify_velocity(norm_speed)
                band = config.VELOCITY_BANDS[band_idx]

                # --- Gate 5: per-band cooldown --------------------------------
                elapsed_s = now - last_trigger_time
                cooldown_s = band.cooldown_ms / 1000.0
                if elapsed_s >= cooldown_s:
                    midi_vel = config.band_to_midi_velocity(band_idx, norm_speed)
                    midi_vel = max(1, min(127, midi_vel))

                    # Update trigger time; tip_y updated below after the loop
                    self._state[line_name] = (tip_y, now)

                    log.debug(
                        "HIT hand=%s line=%s band=%s vel=%d speed=%.3f",
                        self.hand_id, line_name, band.name, midi_vel, tip_speed,
                    )
                    return HitResult(
                        drum_line=dl,
                        velocity=midi_vel,
                        band_idx=band_idx,
                        hand_id=self.hand_id,
                    )

            # Always update last_tip_y for the next frame so we can detect
            # the crossing direction correctly.
            _, last_trigger_time = self._state.get(line_name, (tip_y, 0.0))
            self._state[line_name] = (tip_y, last_trigger_time)

        return None
