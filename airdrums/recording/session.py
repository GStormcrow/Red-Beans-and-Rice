"""
airdrums.recording.session
==========================
Session state: timestamped hit events, skeleton keyframes at a fixed rate,
named takes with mute/solo/undo, loop + overdub recording, and variable-speed
playback with pitch-corrected audio via librosa.
"""
from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .. import config

log = logging.getLogger(__name__)


@dataclass
class HitEvent:
    """One recorded drum hit."""
    timestamp_ms: float
    drum_name: str
    midi_note: int
    velocity: int
    hand_side: str              # "left", "right", "foot_left", "foot_right"
    take_id: int = 0
    is_rimshot: bool = False


@dataclass
class SkeletonKeyframe:
    """One keyframe of the full-body skeleton at a particular time."""
    timestamp_ms: float
    joints: Dict[str, Tuple[float, float, float, float]]   # name -> (x,y,z,vis)


@dataclass
class Take:
    """A named collection of events, supporting mute + solo."""
    take_id: int
    name: str
    events: List[HitEvent] = field(default_factory=list)
    muted: bool = False
    solo: bool = False


class Session:
    """Owns all recorded data and the record/play state machine."""

    def __init__(self, sample_pack: str = "default"):
        self.started_at = time.time()
        self.sample_pack = sample_pack
        self.bpm_history: List[Tuple[float, float]] = []     # (ts_ms, bpm)
        self.calibration: Dict[str, float] = {}
        self.skeleton_keyframes: List[SkeletonKeyframe] = []
        self.takes: Dict[int, Take] = {}
        self._active_take_id: int = 0
        self._next_take_id: int = 1
        self._record_start_ms: Optional[float] = None
        self._keyframe_interval = 1.0 / config.SKELETON_KEYFRAME_FPS
        self._last_keyframe_ms: float = 0.0
        self._lock = threading.Lock()

        # Loop state
        self.loop_enabled: bool = False
        self.loop_bars: int = 4
        self.overdub: bool = False
        self.playback_speed: float = 1.0

        self.new_take("Take 1")

    # ------------------------------------------------------------------
    # Take management
    # ------------------------------------------------------------------
    def new_take(self, name: Optional[str] = None) -> Take:
        tid = self._next_take_id
        self._next_take_id += 1
        take = Take(take_id=tid, name=name or f"Take {tid}")
        with self._lock:
            self.takes[tid] = take
            self._active_take_id = tid
        log.info("New take: %s", take.name)
        return take

    def undo_last_take(self) -> None:
        with self._lock:
            if self.takes:
                last = max(self.takes)
                del self.takes[last]
                if not self.takes:
                    self._next_take_id = 1
                    self.new_take("Take 1")
                else:
                    self._active_take_id = max(self.takes)
        log.info("Undid last take, active=%d", self._active_take_id)

    def mute(self, take_id: int, muted: bool = True) -> None:
        if take_id in self.takes:
            self.takes[take_id].muted = muted

    def solo(self, take_id: int, solo: bool = True) -> None:
        if take_id in self.takes:
            self.takes[take_id].solo = solo

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------
    @property
    def events(self) -> List[HitEvent]:
        """Flat audible event list, respecting mute + solo."""
        solos = [t for t in self.takes.values() if t.solo]
        visible = solos or [t for t in self.takes.values() if not t.muted]
        out: List[HitEvent] = []
        for t in sorted(visible, key=lambda x: x.take_id):
            out.extend(t.events)
        return sorted(out, key=lambda e: e.timestamp_ms)

    def start_recording(self) -> None:
        self._record_start_ms = time.time() * 1000.0
        log.info("Recording started")

    def stop_recording(self) -> None:
        self._record_start_ms = None
        log.info("Recording stopped")

    @property
    def is_recording(self) -> bool:
        return self._record_start_ms is not None

    def record_hit(self, drum_name: str, midi_note: int, velocity: int,
                   hand_side: str, is_rimshot: bool = False) -> Optional[HitEvent]:
        if not self.is_recording:
            return None
        now_ms = time.time() * 1000.0 - (self._record_start_ms or 0)
        # Loop wrap-around
        if self.loop_enabled:
            loop_len = self._loop_length_ms()
            if loop_len > 0:
                now_ms = now_ms % loop_len
                if not self.overdub:
                    # Clear events in this loop region from the active take
                    self._clear_active_loop_region()
        ev = HitEvent(
            timestamp_ms=now_ms,
            drum_name=drum_name,
            midi_note=midi_note,
            velocity=velocity,
            hand_side=hand_side,
            take_id=self._active_take_id,
            is_rimshot=is_rimshot,
        )
        with self._lock:
            self.takes[self._active_take_id].events.append(ev)
        return ev

    def record_skeleton(self, joints: Dict[str, Tuple[float, float, float, float]]) -> None:
        if not self.is_recording:
            return
        now_ms = time.time() * 1000.0 - (self._record_start_ms or 0)
        if (now_ms - self._last_keyframe_ms) * 0.001 < self._keyframe_interval:
            return
        self._last_keyframe_ms = now_ms
        self.skeleton_keyframes.append(SkeletonKeyframe(timestamp_ms=now_ms, joints=joints))

    def _loop_length_ms(self) -> float:
        bpm = self._current_bpm()
        bar_ms = (60000.0 / bpm) * 4.0
        return bar_ms * self.loop_bars

    def _clear_active_loop_region(self) -> None:
        take = self.takes[self._active_take_id]
        take.events = [e for e in take.events if e.timestamp_ms < self._loop_length_ms() - 10]

    def _current_bpm(self) -> float:
        if self.bpm_history:
            return self.bpm_history[-1][1]
        return config.MIDI_DEFAULT_BPM

    # ------------------------------------------------------------------
    # Analysis hooks
    # ------------------------------------------------------------------
    def record_bpm(self, bpm: float) -> None:
        now_ms = time.time() * 1000.0 - (self._record_start_ms or time.time() * 1000.0)
        self.bpm_history.append((max(0.0, now_ms), float(bpm)))

    # ------------------------------------------------------------------
    # Export / import
    # ------------------------------------------------------------------
    def export_json(self, path: str | Path) -> Path:
        path = Path(path)
        data = {
            "started_at": self.started_at,
            "sample_pack": self.sample_pack,
            "calibration": self.calibration,
            "bpm_history": self.bpm_history,
            "takes": {
                str(tid): {
                    "take_id": t.take_id,
                    "name": t.name,
                    "muted": t.muted,
                    "solo": t.solo,
                    "events": [asdict(e) for e in t.events],
                } for tid, t in self.takes.items()
            },
            "skeleton_keyframes": [
                {"timestamp_ms": kf.timestamp_ms, "joints": kf.joints}
                for kf in self.skeleton_keyframes
            ],
            "loop": {
                "enabled": self.loop_enabled,
                "bars": self.loop_bars,
                "overdub": self.overdub,
                "playback_speed": self.playback_speed,
            },
        }
        path.write_text(json.dumps(data, indent=2))
        log.info("Wrote session JSON: %s", path)
        return path

    @classmethod
    def from_json(cls, path: str | Path) -> "Session":
        path = Path(path)
        data = json.loads(path.read_text())
        s = cls(sample_pack=data.get("sample_pack", "default"))
        s.takes.clear()
        s.started_at = data.get("started_at", time.time())
        s.calibration = data.get("calibration", {})
        s.bpm_history = [tuple(x) for x in data.get("bpm_history", [])]
        for tid_s, tdata in data.get("takes", {}).items():
            t = Take(take_id=int(tid_s), name=tdata["name"],
                     muted=tdata.get("muted", False),
                     solo=tdata.get("solo", False))
            t.events = [HitEvent(**e) for e in tdata["events"]]
            s.takes[t.take_id] = t
        s._next_take_id = (max(s.takes) + 1) if s.takes else 1
        s._active_take_id = max(s.takes) if s.takes else 0
        s.skeleton_keyframes = [
            SkeletonKeyframe(timestamp_ms=kf["timestamp_ms"],
                             joints={k: tuple(v) for k, v in kf["joints"].items()})
            for kf in data.get("skeleton_keyframes", [])
        ]
        loop = data.get("loop", {})
        s.loop_enabled = loop.get("enabled", False)
        s.loop_bars = loop.get("bars", 4)
        s.overdub = loop.get("overdub", False)
        s.playback_speed = loop.get("playback_speed", 1.0)
        return s

    # ------------------------------------------------------------------
    # Variable-speed playback with pitch correction (librosa)
    # ------------------------------------------------------------------
    def timestretch_buffer(self, audio, sr: int, speed: Optional[float] = None):
        """Return a pitch-corrected time-stretched audio buffer."""
        import librosa  # type: ignore
        import numpy as np
        speed = float(speed if speed is not None else self.playback_speed)
        if abs(speed - 1.0) < 1e-3:
            return audio
        stretched = librosa.effects.time_stretch(
            np.ascontiguousarray(audio.T) if audio.ndim == 2 else audio,
            rate=speed,
        )
        return stretched.T if audio.ndim == 2 else stretched
