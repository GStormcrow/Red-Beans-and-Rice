"""
airdrums.tracking.detectors
===========================
Strike detection.

VelocitySpikeDetector: one per drumstick tip. Looks for the canonical
accelerate -> peak -> decelerate profile and emits ("strike", midi_velocity).

FootPedalDetector: one per foot. Uses raw heel / toe / ankle landmarks (NOT
the drumstick tip) plus a hi-hat state machine.
"""
from __future__ import annotations

import logging
import math
import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional, Tuple

from .. import config
from .skeleton import Joint

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# VelocitySpikeDetector
# ---------------------------------------------------------------------------
class VelocitySpikeDetector:
    """Detects a drum strike from a drumstick tip's velocity profile.

    The detector fires when the tip has been accelerating strongly, crosses
    `spike_threshold`, reaches a local peak, then decelerates by at least
    `peak_drop_ratio` of that peak. MIDI velocity is mapped from peak speed.
    """

    def __init__(self,
                 spike_threshold: float = config.STRIKE_SPIKE_THRESHOLD,
                 cooldown_ms: int = config.STRIKE_COOLDOWN_MS,
                 peak_drop_ratio: float = config.STRIKE_PEAK_DROP_RATIO,
                 side: str = "right"):
        self.spike_threshold = float(spike_threshold)
        self.cooldown_ms = int(cooldown_ms)
        self.peak_drop_ratio = float(peak_drop_ratio)
        self.side = side
        self._history: Deque[Tuple[float, float]] = deque(maxlen=16)   # (t, speed)
        self._last_strike_ms: float = 0.0
        self._peak_speed: float = 0.0
        self._velocity_curve: str = "linear"

    def set_velocity_curve(self, curve: str) -> None:
        if curve in ("linear", "logarithmic", "exponential"):
            self._velocity_curve = curve

    def update(self, tip: Joint, elbow: Optional[Joint] = None
               ) -> Tuple[Optional[str], int, bool]:
        """Return (event, midi_velocity, is_rimshot).

        ``event`` is ``"strike"`` or ``None``. ``is_rimshot`` is only
        meaningful when a strike is returned.
        """
        now_ms = time.time() * 1000.0
        speed = float(tip.speed)
        self._history.append((now_ms, speed))

        if now_ms - self._last_strike_ms < self.cooldown_ms:
            return None, 0, False

        # Track the running peak speed over the short window
        if speed > self._peak_speed:
            self._peak_speed = speed

        # Need at least a threshold-crossing peak before we can fire
        if self._peak_speed < self.spike_threshold:
            return None, 0, False

        # Drop detected: fire
        if speed < self._peak_speed * self.peak_drop_ratio:
            peak = self._peak_speed
            self._peak_speed = 0.0
            self._last_strike_ms = now_ms
            velocity = self._speed_to_midi(peak)
            rim = self._is_rimshot(tip, elbow)
            log.debug("Strike(%s) peak=%.2f vel=%d rim=%s", self.side, peak, velocity, rim)
            return "strike", velocity, rim
        return None, 0, False

    def _speed_to_midi(self, speed: float) -> int:
        # Map speed in [threshold, 4*threshold] to MIDI [40, 127]
        norm = (speed - self.spike_threshold) / (3 * self.spike_threshold + 1e-6)
        norm = max(0.0, min(1.0, norm))
        if self._velocity_curve == "logarithmic":
            norm = math.log1p(norm * (math.e - 1)) / 1.0
        elif self._velocity_curve == "exponential":
            norm = norm * norm
        vel = int(40 + norm * (127 - 40))
        return max(1, min(127, vel))

    def _is_rimshot(self, tip: Joint, elbow: Optional[Joint]) -> bool:
        """Classify rimshot by forearm angle. Low elbow relative to wrist
        tends to produce rimshots (glancing hits)."""
        if elbow is None or not elbow.visible:
            return False
        # Vertical drop from elbow to wrist (and wrist to tip) approximated
        # by y component relative to forearm length.
        dx = tip.x - elbow.x
        dy = tip.y - elbow.y
        length = math.hypot(dx, dy) + 1e-6
        angle_deg = math.degrees(math.atan2(abs(dy), abs(dx)))
        return angle_deg < (180.0 - config.RIMSHOT_ELBOW_ANGLE_DEG)


# ---------------------------------------------------------------------------
# FootPedalDetector
# ---------------------------------------------------------------------------
@dataclass
class PedalEvent:
    """Event emitted by FootPedalDetector.update()."""
    kind: str            # "strike" | "hihat_state"
    velocity: int = 0
    hihat_state: str = "open"


class FootPedalDetector:
    """Heel+toe pedal detector. Uses raw landmark joints, not drumsticks.

    * Right foot -> kick drum: fires on downward heel velocity spike.
    * Left foot  -> hi-hat pedal: drives a state machine and fires the
      hi-hat chick on ``closed`` transitions.
    """

    def __init__(self, foot: str = "right"):
        assert foot in ("left", "right")
        self.foot = foot
        self.cooldown_ms = (config.PEDAL_KICK_COOLDOWN_MS if foot == "right"
                            else config.PEDAL_HIHAT_COOLDOWN_MS)
        self.heel_threshold = config.PEDAL_HEEL_THRESHOLD
        self._last_strike_ms: float = 0.0
        self._peak_speed: float = 0.0
        self._hihat_state: str = "open"
        self._last_state: str = "open"

    @property
    def hihat_state(self) -> str:
        return self._hihat_state

    def set_threshold(self, v: float) -> None:
        self.heel_threshold = float(v)

    def update(self, heel: Optional[Joint], toe: Optional[Joint],
               ankle: Optional[Joint]) -> Optional[PedalEvent]:
        now_ms = time.time() * 1000.0
        if heel is None or not heel.visible:
            return None

        # Downward velocity in normalized units / s. A pedal stomp spikes vy
        # positive (image y grows downward).
        down_speed = max(0.0, heel.vy)

        event: Optional[PedalEvent] = None

        if self.foot == "right":
            # Kick drum: straight velocity-spike detection on heel
            if down_speed > self._peak_speed:
                self._peak_speed = down_speed
            if (self._peak_speed > self.heel_threshold and
                    down_speed < self._peak_speed * 0.5 and
                    now_ms - self._last_strike_ms > self.cooldown_ms):
                velocity = self._speed_to_midi(self._peak_speed)
                self._peak_speed = 0.0
                self._last_strike_ms = now_ms
                event = PedalEvent(kind="strike", velocity=velocity)
        else:
            # Hi-hat pedal: state machine driven by heel-toe ratio.
            ratio = self._heel_toe_ratio(heel, toe, ankle)
            new_state = self._advance_state(ratio)
            if new_state != self._last_state:
                self._last_state = new_state
                event = PedalEvent(kind="hihat_state", hihat_state=new_state)
                if new_state == "closed" and now_ms - self._last_strike_ms > self.cooldown_ms:
                    # Pedal chick fires an audio/MIDI event separately
                    self._last_strike_ms = now_ms
                    event = PedalEvent(kind="strike",
                                       velocity=self._speed_to_midi(down_speed),
                                       hihat_state=new_state)
            self._hihat_state = new_state

        return event

    def _heel_toe_ratio(self, heel: Joint, toe: Optional[Joint],
                        ankle: Optional[Joint]) -> float:
        if toe is None or not toe.visible:
            return 0.5
        # When toe is below heel (pedal pressed), ratio drops
        dy = toe.y - heel.y
        return float(max(0.0, min(1.0, 0.5 - dy * 3.0)))

    def _advance_state(self, ratio: float) -> str:
        s = self._hihat_state
        if s in ("open", "opening"):
            if ratio > config.HIHAT_CLOSE_PEDAL_THRESHOLD:
                return "closing"
            return s
        if s == "closing":
            if ratio > config.HIHAT_CLOSE_PEDAL_THRESHOLD + 0.05:
                return "closed"
            if ratio < config.HIHAT_OPEN_PEDAL_THRESHOLD:
                return "opening"
            return s
        if s == "closed":
            if ratio < config.HIHAT_CLOSE_PEDAL_THRESHOLD:
                return "opening"
            return s
        return "open"

    def _speed_to_midi(self, speed: float) -> int:
        norm = max(0.0, min(1.0, speed / (2.0 * self.heel_threshold + 1e-6)))
        return max(30, min(127, int(50 + norm * 77)))
