"""
airdrums.recording.session
==========================
Session recording, loop, overdub, and persistence for AirDrums V2.

Provides HitEvent, Take, and Session — the core data layer for a drumming
session.  All state mutations are handled through Session methods; no
direct field access to internal state should be needed by callers.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from airdrums import config
from airdrums.config import DrumLine

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class HitEvent:
    """One recorded drum hit, with velocity band and hand attribution."""

    timestamp_ms: float
    drum_name: str
    midi_note: int
    velocity: int           # MIDI 0–127
    velocity_band: str      # "ghost" | "soft" | "medium" | "hard" | "accent"
    hand_side: str          # "left" | "right"
    take_id: int


@dataclass
class Take:
    """A named, mutable collection of HitEvents supporting mute and solo."""

    take_id: int
    name: str
    events: List[HitEvent] = field(default_factory=list)
    muted: bool = False
    solo: bool = False


# ---------------------------------------------------------------------------
# Session
# ---------------------------------------------------------------------------

class Session:
    """Owns all recorded data and drives the record/play/loop state machine.

    Parameters
    ----------
    drum_lines:
        The ordered list of :class:`~airdrums.config.DrumLine` objects that
        define the kit layout for this session.
    """

    def __init__(self, drum_lines: List[DrumLine]) -> None:
        """Initialise an empty session for the given drum layout."""
        self.drum_lines: List[DrumLine] = drum_lines

        self.takes: List[Take] = []
        self.current_take: Optional[Take] = None

        # BPM tracking — list of (timestamp_ms, bpm)
        self.bpm_history: List[Tuple[float, float]] = []

        # Internal recording state
        self._recording: bool = False
        self._overdub: bool = False
        self._playing: bool = False
        self._loop_bars: int = 4
        self._loop_length_ms: float = 0.0
        self._playback_speed: float = 1.0
        self._start_time_ms: float = 0.0

        self._last_autosave: float = time.monotonic()

        log.debug("Session initialised with %d drum lines.", len(drum_lines))

    # ------------------------------------------------------------------
    # Recording lifecycle
    # ------------------------------------------------------------------

    def start_recording(self, loop_bars: int = 4, detected_bpm: float = 120.0) -> None:
        """Begin a new recording take.

        Creates a fresh :class:`Take`, marks the session as recording, and
        computes the loop length from *loop_bars* and *detected_bpm*.

        Parameters
        ----------
        loop_bars:
            Number of bars in the loop (must be one of
            :data:`~airdrums.config.LOOP_BAR_OPTIONS`).
        detected_bpm:
            Current tempo estimate; used to derive the loop length in ms.
        """
        if loop_bars not in config.LOOP_BAR_OPTIONS:
            log.warning(
                "loop_bars=%d not in LOOP_BAR_OPTIONS %s; defaulting to 4.",
                loop_bars, config.LOOP_BAR_OPTIONS,
            )
            loop_bars = 4

        take_id = len(self.takes) + 1
        take = Take(take_id=take_id, name=f"Take {take_id}")
        self.current_take = take

        self._loop_bars = loop_bars
        self._loop_length_ms = self._compute_loop_length_ms(loop_bars, detected_bpm)
        self._start_time_ms = time.monotonic() * 1000.0
        self._recording = True

        log.info(
            "Recording started — take=%d, bars=%d, bpm=%.1f, loop_ms=%.1f",
            take_id, loop_bars, detected_bpm, self._loop_length_ms,
        )

    def stop_recording(self) -> None:
        """Finalise the current take and append it to :attr:`takes`."""
        if not self._recording:
            log.debug("stop_recording called while not recording; ignored.")
            return

        self._recording = False

        if self.current_take is not None:
            self.takes.append(self.current_take)
            log.info(
                "Recording stopped — take=%d, %d events captured.",
                self.current_take.take_id, len(self.current_take.events),
            )
            self.current_take = None
        else:
            log.warning("stop_recording: current_take was None.")

    # ------------------------------------------------------------------
    # Hit recording
    # ------------------------------------------------------------------

    def record_hit(
        self,
        drum_name: str,
        midi_note: int,
        velocity: int,
        velocity_band: str,
        hand_side: str,
    ) -> Optional[HitEvent]:
        """Record a single drum hit into the active take.

        In overdub mode the timestamp is wrapped modulo the loop length so
        that the hit slots into the correct position within the loop.

        Parameters
        ----------
        drum_name:
            Canonical name matching a :class:`~airdrums.config.DrumLine`.
        midi_note:
            MIDI note number (0–127).
        velocity:
            MIDI velocity (0–127).
        velocity_band:
            Human-readable band label, e.g. ``"medium"``.
        hand_side:
            ``"left"`` or ``"right"``.

        Returns
        -------
        HitEvent or None
            The newly created event, or *None* if not currently recording.
        """
        if not self._recording or self.current_take is None:
            return None

        now_ms = time.monotonic() * 1000.0 - self._start_time_ms

        if self._overdub and self._loop_length_ms > 0:
            now_ms = now_ms % self._loop_length_ms

        event = HitEvent(
            timestamp_ms=now_ms,
            drum_name=drum_name,
            midi_note=midi_note,
            velocity=velocity,
            velocity_band=velocity_band,
            hand_side=hand_side,
            take_id=self.current_take.take_id,
        )
        self.current_take.events.append(event)
        return event

    # ------------------------------------------------------------------
    # Loop / overdub controls
    # ------------------------------------------------------------------

    def toggle_overdub(self) -> None:
        """Flip the overdub flag on or off."""
        self._overdub = not self._overdub
        log.info("Overdub %s.", "enabled" if self._overdub else "disabled")

    def set_loop_bars(self, bars: int) -> None:
        """Update the loop length, recomputing from the most recent BPM.

        Parameters
        ----------
        bars:
            Desired loop length in bars.
        """
        self._loop_bars = bars
        current_bpm = self._latest_bpm()
        self._loop_length_ms = self._compute_loop_length_ms(bars, current_bpm)
        log.info(
            "Loop bars set to %d — loop_ms=%.1f (bpm=%.1f).",
            bars, self._loop_length_ms, current_bpm,
        )

    # ------------------------------------------------------------------
    # Playback
    # ------------------------------------------------------------------

    def toggle_playback(self) -> None:
        """Flip playback on or off."""
        self._playing = not self._playing
        log.info("Playback %s.", "started" if self._playing else "stopped")

    def set_playback_speed(self, speed: float) -> None:
        """Set playback speed, clamped to the configured min/max.

        Parameters
        ----------
        speed:
            Desired speed multiplier (e.g. ``1.5`` for 50 % faster).
        """
        clamped = max(config.PLAYBACK_SPEED_MIN, min(config.PLAYBACK_SPEED_MAX, speed))
        if abs(clamped - speed) > 1e-4:
            log.debug(
                "Playback speed %.2f clamped to %.2f.", speed, clamped
            )
        self._playback_speed = clamped
        log.info("Playback speed set to %.2f.", self._playback_speed)

    # ------------------------------------------------------------------
    # Take management
    # ------------------------------------------------------------------

    def undo_last_take(self) -> None:
        """Remove the most recently completed take from :attr:`takes`."""
        if not self.takes:
            log.debug("undo_last_take: no takes to remove.")
            return
        removed = self.takes.pop()
        log.info("Undid take %d ('%s').", removed.take_id, removed.name)

    def get_active_events(self) -> List[HitEvent]:
        """Return events from all non-muted takes (or soloed takes if any).

        If one or more takes have ``solo=True``, only those takes contribute
        events.  Otherwise every un-muted take contributes.

        Returns
        -------
        list[HitEvent]
            Merged, time-sorted event list.
        """
        soloed = [t for t in self.takes if t.solo]
        if soloed:
            source_takes = soloed
        else:
            source_takes = [t for t in self.takes if not t.muted]

        events: List[HitEvent] = []
        for take in source_takes:
            events.extend(take.events)

        events.sort(key=lambda e: e.timestamp_ms)
        return events

    # ------------------------------------------------------------------
    # BPM tracking
    # ------------------------------------------------------------------

    def update_bpm(self, bpm: float) -> None:
        """Append a BPM sample stamped with the current time.

        Parameters
        ----------
        bpm:
            Detected tempo in beats per minute.
        """
        now_ms = time.monotonic() * 1000.0
        self.bpm_history.append((now_ms, float(bpm)))
        log.debug("BPM updated: %.1f", bpm)

    # ------------------------------------------------------------------
    # Autosave
    # ------------------------------------------------------------------

    def maybe_autosave(self, session_path: Path) -> None:
        """Write a recovery snapshot to disk if the autosave interval has elapsed.

        The recovery file is always written to
        :data:`~airdrums.config.RECOVERY_DIR` ``/ "recovery.json"``,
        regardless of *session_path*.  Errors are swallowed silently so that
        a broken filesystem does not interrupt the player.

        Parameters
        ----------
        session_path:
            Provided for context (currently unused) but kept in the
            signature for future delta-save support.
        """
        now = time.monotonic()
        if now - self._last_autosave < config.AUTOSAVE_INTERVAL_S:
            return

        recovery_path = config.RECOVERY_DIR / "recovery.json"
        try:
            self.export_json(recovery_path)
            log.debug("Autosaved recovery snapshot to %s.", recovery_path)
        except Exception:  # noqa: BLE001
            log.debug("Autosave failed silently.", exc_info=True)

        self._last_autosave = now

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def export_json(self, path: Path) -> None:
        """Serialise the full session to a JSON file.

        The file contains drum line metadata, all takes with their events,
        the BPM history, loop/playback settings, and a version tag so that
        future readers can migrate old formats.

        Parameters
        ----------
        path:
            Destination file path.  Parent directories are created if needed.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data: Dict = {
            "version": "2.0",
            "timestamp": time.time(),
            "loop_bars": self._loop_bars,
            "loop_length_ms": self._loop_length_ms,
            "playback_speed": self._playback_speed,
            "bpm_history": self.bpm_history,
            "drum_lines": [
                {
                    "name": dl.name,
                    "x_center": dl.x_center,
                    "y_position": dl.y_position,
                    "half_width": dl.half_width,
                    "midi_note": dl.midi_note,
                    "color_bgr": list(dl.color_bgr),
                    "is_cymbal": dl.is_cymbal,
                    "label": dl.label,
                }
                for dl in self.drum_lines
            ],
            "takes": [
                {
                    "take_id": t.take_id,
                    "name": t.name,
                    "muted": t.muted,
                    "solo": t.solo,
                    "events": [asdict(e) for e in t.events],
                }
                for t in self.takes
            ],
        }

        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        log.info("Session exported to %s (%d takes).", path, len(self.takes))

    @classmethod
    def from_json(cls, path: Path) -> "Session":
        """Reconstruct a :class:`Session` from a JSON file produced by
        :meth:`export_json`.

        Parameters
        ----------
        path:
            Source file path.

        Returns
        -------
        Session
            A fully populated session instance.

        Raises
        ------
        FileNotFoundError
            If *path* does not exist.
        ValueError
            If the JSON structure is missing required keys.
        """
        path = Path(path)
        data = json.loads(path.read_text(encoding="utf-8"))

        drum_lines = [
            DrumLine(
                name=dl["name"],
                x_center=dl["x_center"],
                y_position=dl["y_position"],
                half_width=dl["half_width"],
                midi_note=dl["midi_note"],
                color_bgr=tuple(dl["color_bgr"]),
                is_cymbal=dl["is_cymbal"],
                label=dl["label"],
            )
            for dl in data.get("drum_lines", [])
        ]

        session = cls(drum_lines=drum_lines)
        session._loop_bars = data.get("loop_bars", 4)
        session._loop_length_ms = data.get("loop_length_ms", 0.0)
        session._playback_speed = data.get("playback_speed", 1.0)
        session.bpm_history = [tuple(x) for x in data.get("bpm_history", [])]

        takes: List[Take] = []
        for td in data.get("takes", []):
            take = Take(
                take_id=td["take_id"],
                name=td["name"],
                muted=td.get("muted", False),
                solo=td.get("solo", False),
            )
            take.events = [HitEvent(**e) for e in td.get("events", [])]
            takes.append(take)

        session.takes = takes
        log.info(
            "Session loaded from %s — %d drum lines, %d takes.",
            path, len(drum_lines), len(takes),
        )
        return session

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _compute_loop_length_ms(self, bars: int, bpm: float) -> float:
        """Return loop length in milliseconds for *bars* bars at *bpm* BPM."""
        if bpm <= 0:
            bpm = 120.0
        beat_ms = 60000.0 / bpm
        bar_ms = beat_ms * 4.0        # 4/4 time
        return bar_ms * bars

    def _latest_bpm(self) -> float:
        """Return the most recently recorded BPM, or 120.0 if unknown."""
        if self.bpm_history:
            return self.bpm_history[-1][1]
        return 120.0
