"""
airdrums.config
===============
Central configuration for the AirDrums system. All tunable constants, zone
coordinates, MIDI mappings, paths, and hardware profile settings live here.
No logic module should hardcode values that could appear in this file.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
HOME_DIR = Path.home() / ".airdrums"
PROFILES_DIR = HOME_DIR / "profiles"
SESSIONS_DIR = HOME_DIR / "sessions"
EXPORTS_DIR = HOME_DIR / "exports"
SAMPLE_PACKS_DIR = Path(os.environ.get("AIRDRUMS_PACKS", Path(__file__).parent / "packs"))
DEFAULT_PROFILE_PATH = PROFILES_DIR / "default.json"

for _d in (HOME_DIR, PROFILES_DIR, SESSIONS_DIR, EXPORTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
LOG_LEVEL = os.environ.get("AIRDRUMS_LOG", "INFO")
LOG_FORMAT = "[%(asctime)s] %(levelname)s %(name)s: %(message)s"

# -----------------------------------------------------------------------------
# Camera
# -----------------------------------------------------------------------------
CAMERA_INDEX = 0
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
CAMERA_FPS = 30

# -----------------------------------------------------------------------------
# Hardware profiles
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class HardwareProfile:
    """Describes a runtime configuration appropriate for a class of hardware."""
    name: str
    depth_model: str          # HuggingFace repo id
    depth_target_fps: int
    pose_complexity: int      # 0 | 1 | 2
    depth_enabled: bool = True

PROFILE_HIGH = HardwareProfile(
    name="high",
    depth_model="depth-anything/Depth-Anything-V2-Base-hf",
    depth_target_fps=28,
    pose_complexity=1,
)
PROFILE_MEDIUM = HardwareProfile(
    name="medium",
    depth_model="depth-anything/Depth-Anything-V2-Small-hf",
    depth_target_fps=18,
    pose_complexity=1,
)
PROFILE_LOW = HardwareProfile(
    name="low",
    depth_model="depth-anything/Depth-Anything-V2-Small-hf",
    depth_target_fps=6,
    pose_complexity=0,
)
PROFILES = {"high": PROFILE_HIGH, "medium": PROFILE_MEDIUM, "low": PROFILE_LOW}

# -----------------------------------------------------------------------------
# Depth engine
# -----------------------------------------------------------------------------
DEPTH_INPUT_SIZE: Tuple[int, int] = (518, 392)  # (W, H) native DA-V2 input
DEPTH_QUEUE_SIZE = 2
DEPTH_DEFAULT_SCALE = 1.0   # recalibrated at startup
DEPTH_SAMPLE_KERNEL = 5     # 5x5 median sampling around each landmark

# -----------------------------------------------------------------------------
# Tracking
# -----------------------------------------------------------------------------
POSE_LANDMARK_COUNT = 33
VISIBILITY_THRESHOLD = 0.4
VELOCITY_HISTORY_FRAMES = 8

# -----------------------------------------------------------------------------
# Virtual drumstick
# -----------------------------------------------------------------------------
DRUMSTICK_LENGTH_DEFAULT = 0.18     # normalized to frame height
DRUMSTICK_LENGTH_MIN = 0.10
DRUMSTICK_LENGTH_MAX = 0.30
DRUMSTICK_GRIP_PX = 6
DRUMSTICK_TIP_PX = 2
DRUMSTICK_TIP_RADIUS_PX = 5

# -----------------------------------------------------------------------------
# Detection
# -----------------------------------------------------------------------------
STRIKE_COOLDOWN_MS = 130
STRIKE_PEAK_DROP_RATIO = 0.5
STRIKE_SPIKE_THRESHOLD = 0.9      # normalized speed units / s, calibrated
PEDAL_KICK_COOLDOWN_MS = 120
PEDAL_HIHAT_COOLDOWN_MS = 100
PEDAL_HEEL_THRESHOLD = 0.8        # calibrated
RIMSHOT_ELBOW_ANGLE_DEG = 150.0   # >=150 treated as center, <150 rimshot

# -----------------------------------------------------------------------------
# MIDI
# -----------------------------------------------------------------------------
MIDI_PORT_NAME = "AirDrums"
MIDI_CHANNEL = 9              # 0-indexed channel 10 (GM drums)
MIDI_NOTE_OFF_MS = 50
MIDI_CLOCK_PPQN = 24
MIDI_DEFAULT_BPM = 120.0

# -----------------------------------------------------------------------------
# Audio
# -----------------------------------------------------------------------------
AUDIO_SAMPLE_RATE = 44100
AUDIO_BIT_DEPTH = 16
AUDIO_CHANNELS = 2
AUDIO_BUFFER_SIZE = 256
VELOCITY_LAYERS = ("pp", "mp", "mf", "f", "ff")

# -----------------------------------------------------------------------------
# Drum kit
# -----------------------------------------------------------------------------
@dataclass
class DrumZone:
    """A rectangular hit zone in normalized player-relative screen space."""
    name: str
    x_range: Tuple[float, float]
    y_range: Tuple[float, float]
    z_range: Tuple[float, float]
    midi_note: int
    color_bgr: Tuple[int, int, int]
    is_pedal: bool = False
    # A group for stem export: kick, snare, hihat, toms, cymbals
    stem_group: str = "toms"

# All coords are normalized to player frame (0..1) where hip center is (0.5, 0.7)
# and shoulder width ~ 0.35 of frame width. Runtime rescales with player.
DRUM_ZONES = [
    DrumZone("Hi-Hat Closed", (0.12, 0.32), (0.35, 0.50), (0.55, 1.0), 42,
             (255, 255, 0), False, "hihat"),
    DrumZone("Snare",         (0.40, 0.60), (0.45, 0.60), (0.55, 1.0), 38,
             (0, 200, 255),   False, "snare"),
    DrumZone("Tom 1",         (0.30, 0.48), (0.25, 0.40), (0.55, 1.0), 45,
             (0, 120, 255),   False, "toms"),
    DrumZone("Tom 2",         (0.52, 0.70), (0.25, 0.40), (0.55, 1.0), 47,
             (0, 80,  255),   False, "toms"),
    DrumZone("Floor Tom",     (0.68, 0.88), (0.45, 0.62), (0.55, 1.0), 41,
             (0, 40,  200),   False, "toms"),
    DrumZone("Crash Cymbal",  (0.02, 0.20), (0.10, 0.28), (0.55, 1.0), 49,
             (255, 220, 100), False, "cymbals"),
    DrumZone("Ride Cymbal",   (0.80, 0.98), (0.10, 0.28), (0.55, 1.0), 51,
             (220, 180, 80),  False, "cymbals"),
    DrumZone("Bass Drum",     (0.40, 0.60), (0.82, 0.98), (0.0, 1.0),  36,
             (150, 80, 255),  True,  "kick"),
    DrumZone("Hi-Hat Pedal",  (0.12, 0.30), (0.82, 0.98), (0.0, 1.0),  44,
             (255, 180, 80),  True,  "hihat"),
]
DRUM_ZONES_BY_NAME = {z.name: z for z in DRUM_ZONES}
DRUM_ZONES_BY_NOTE = {z.midi_note: z for z in DRUM_ZONES}

# Rimshot variant for snare
SNARE_RIMSHOT_NOTE = 37

# -----------------------------------------------------------------------------
# Hi-hat state machine
# -----------------------------------------------------------------------------
HIHAT_STATES = ("open", "closing", "closed", "opening")
HIHAT_CLOSE_PEDAL_THRESHOLD = 0.65      # heel-toe ratio threshold
HIHAT_OPEN_PEDAL_THRESHOLD = 0.35

# -----------------------------------------------------------------------------
# Recording
# -----------------------------------------------------------------------------
SKELETON_KEYFRAME_FPS = 10
LOOP_BAR_CHOICES = (1, 2, 4, 8)

# -----------------------------------------------------------------------------
# Analytics
# -----------------------------------------------------------------------------
BPM_ROLLING_WINDOW = 8
QUANTIZE_GRIDS = ("none", "1/8", "1/16", "1/32")

# -----------------------------------------------------------------------------
# UI
# -----------------------------------------------------------------------------
HUD_THEMES = ("dark", "light", "neon")
DEFAULT_HUD_THEME = "dark"
