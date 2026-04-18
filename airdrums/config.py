"""Central configuration for AirDrums V2 — all constants live here, never in logic modules."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
HOME_DIR = Path.home() / ".airdrums"
PROFILES_DIR = HOME_DIR / "profiles"
SESSIONS_DIR = HOME_DIR / "sessions"
EXPORTS_DIR = HOME_DIR / "exports"
RECOVERY_DIR = HOME_DIR / "recovery"

for _d in (HOME_DIR, PROFILES_DIR, SESSIONS_DIR, EXPORTS_DIR, RECOVERY_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_LEVEL = logging.DEBUG
LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"

# ---------------------------------------------------------------------------
# Camera
# ---------------------------------------------------------------------------
CAMERA_INDEX = 0
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
CAMERA_FPS = 30

# ---------------------------------------------------------------------------
# Hardware Profiles
# ---------------------------------------------------------------------------
@dataclass
class HardwareProfile:
    """Resource limits for a hardware tier."""

    name: str
    depth_model: str   # HuggingFace model id
    depth_fps: int     # target depth inference FPS
    device: str        # 'cuda', 'mps', or 'cpu'


PROFILE_HIGH = HardwareProfile(
    name="high",
    depth_model="depth-anything/Depth-Anything-V2-Base-hf",
    depth_fps=28,
    device="cuda",
)
PROFILE_MEDIUM = HardwareProfile(
    name="medium",
    depth_model="depth-anything/Depth-Anything-V2-Small-hf",
    depth_fps=17,
    device="mps",
)
PROFILE_LOW = HardwareProfile(
    name="low",
    depth_model="depth-anything/Depth-Anything-V2-Small-hf",
    depth_fps=6,
    device="cpu",
)
PROFILES: dict = {p.name: p for p in (PROFILE_HIGH, PROFILE_MEDIUM, PROFILE_LOW)}

# ---------------------------------------------------------------------------
# Depth Engine
# ---------------------------------------------------------------------------
DEPTH_INPUT_WIDTH = 518
DEPTH_INPUT_HEIGHT = 392
DEPTH_QUEUE_SIZE = 2
DEPTH_KERNEL_SIZE = 5   # 5x5 median kernel for landmark depth sampling

# ---------------------------------------------------------------------------
# MediaPipe Hands
# ---------------------------------------------------------------------------
MP_MAX_HANDS = 2
MP_DETECTION_CONFIDENCE = 0.7
MP_TRACKING_CONFIDENCE = 0.6

# Key landmark indices
LM_WRIST = 0
LM_INDEX_MCP = 5
LM_INDEX_DIP = 7
LM_INDEX_TIP = 8   # INDEX_FINGER_TIP — primary trigger point
LM_MIDDLE_MCP = 9

# ---------------------------------------------------------------------------
# Virtual Drumstick
# ---------------------------------------------------------------------------
STICK_LENGTH_DEFAULT = 0.18   # normalized units
STICK_LENGTH_MIN = 0.10
STICK_LENGTH_MAX = 0.30
VELOCITY_DEQUE_FACTOR = 8     # deque_len = int(factor * fps / 30)

# ---------------------------------------------------------------------------
# Velocity Bands
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class VelocityBand:
    """Maps a speed range to sample-layer suffixes, cooldown, and MIDI velocity range."""

    name: str
    speed_min: float
    speed_max: float
    suffix: str        # primary: _ghost / _soft / _medium / _hard / _accent
    alt_suffix: str    # legacy:  _pp / _mp / _mf / _f / _ff
    cooldown_ms: int
    midi_lo: int
    midi_hi: int


VELOCITY_BANDS: List[VelocityBand] = [
    VelocityBand("ghost",  0.00, 0.25, "_ghost",  "_pp",  60,   1,  31),
    VelocityBand("soft",   0.25, 0.45, "_soft",   "_mp",  90,  32,  63),
    VelocityBand("medium", 0.45, 0.65, "_medium", "_mf", 120,  64,  95),
    VelocityBand("hard",   0.65, 0.85, "_hard",   "_f",  160,  96, 111),
    VelocityBand("accent", 0.85, 1.00, "_accent", "_ff", 200, 112, 127),
]


def classify_velocity(normalized_speed: float) -> int:
    """Return band index 0-4 for a normalized tip speed in [0, 1]."""
    for i, band in enumerate(VELOCITY_BANDS):
        if normalized_speed <= band.speed_max:
            return i
    return len(VELOCITY_BANDS) - 1


def band_to_midi_velocity(band_idx: int, normalized_speed: float) -> int:
    """Map a speed within its band to a MIDI velocity in [1, 127]."""
    band = VELOCITY_BANDS[band_idx]
    span = band.speed_max - band.speed_min
    t = (normalized_speed - band.speed_min) / span if span > 0 else 0.5
    t = max(0.0, min(1.0, t))
    return int(band.midi_lo + t * (band.midi_hi - band.midi_lo))

# ---------------------------------------------------------------------------
# Drum Lines
# ---------------------------------------------------------------------------
@dataclass
class DrumLine:
    """A flat horizontal trigger line representing one drum surface."""

    name: str
    x_center: float                  # normalized 0.0-1.0
    y_position: float                # normalized 0.0-1.0 (0 = top)
    half_width: float                # half trigger width, normalized
    midi_note: int
    color_bgr: Tuple[int, int, int]
    is_cymbal: bool
    label: str


DEFAULT_DRUM_LINES: List[DrumLine] = [
    DrumLine("crash",    x_center=0.12, y_position=0.20, half_width=0.10,
             midi_note=49, color_bgr=(255, 200, 80),  is_cymbal=True,  label="Crash"),
    DrumLine("hihat",    x_center=0.28, y_position=0.30, half_width=0.10,
             midi_note=42, color_bgr=(100, 220, 200), is_cymbal=False, label="Hi-Hat"),
    DrumLine("snare",    x_center=0.50, y_position=0.47, half_width=0.12,
             midi_note=38, color_bgr=(60, 140, 255),  is_cymbal=False, label="Snare"),
    DrumLine("tom1",     x_center=0.36, y_position=0.38, half_width=0.10,
             midi_note=45, color_bgr=(120, 255, 160), is_cymbal=False, label="Tom 1"),
    DrumLine("tom2",     x_center=0.64, y_position=0.38, half_width=0.10,
             midi_note=47, color_bgr=(160, 120, 255), is_cymbal=False, label="Tom 2"),
    DrumLine("floortom", x_center=0.72, y_position=0.57, half_width=0.10,
             midi_note=41, color_bgr=(255, 100, 180), is_cymbal=False, label="Floor Tom"),
    DrumLine("ride",     x_center=0.88, y_position=0.20, half_width=0.10,
             midi_note=51, color_bgr=(80, 200, 255),  is_cymbal=True,  label="Ride"),
]


def get_drum_lines(
    lines: Optional[List[DrumLine]] = None,
    mirrored: bool = False,
) -> List[DrumLine]:
    """Return drum lines, optionally mirrored for left-handed mode."""
    src = lines if lines is not None else DEFAULT_DRUM_LINES
    if not mirrored:
        return src
    return [
        DrumLine(
            name=dl.name,
            x_center=1.0 - dl.x_center,
            y_position=dl.y_position,
            half_width=dl.half_width,
            midi_note=dl.midi_note,
            color_bgr=dl.color_bgr,
            is_cymbal=dl.is_cymbal,
            label=dl.label,
        )
        for dl in src
    ]

# ---------------------------------------------------------------------------
# MIDI
# ---------------------------------------------------------------------------
MIDI_PORT_NAME = "AirDrums"
MIDI_CHANNEL = 9             # 0-indexed -> GM channel 10
MIDI_NOTE_OFF_DELAY_MS = 50
MIDI_PPQN = 24

# ---------------------------------------------------------------------------
# Audio
# ---------------------------------------------------------------------------
AUDIO_SAMPLE_RATE = 44100
AUDIO_CHANNELS = 2
AUDIO_BUFFER = 256
AUDIO_BIT_DEPTH = 16
PACKS_DIR = Path(__file__).parent.parent / "packs"

# ---------------------------------------------------------------------------
# Recording
# ---------------------------------------------------------------------------
AUTOSAVE_INTERVAL_S = 60
LOOP_BAR_OPTIONS = (1, 2, 4, 8)
PLAYBACK_SPEED_MIN = 0.5
PLAYBACK_SPEED_MAX = 2.0
KEYFRAME_FPS = 10

# ---------------------------------------------------------------------------
# Analytics
# ---------------------------------------------------------------------------
BPM_SMOOTHING_WINDOW = 8    # rolling median over N snare hits
BPM_STABILITY_BARS = 4      # std-dev window in bars
DEFAULT_QUANTIZATION = "1/16"

# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
OVERLAY_OPACITY_REST = 0.55
OVERLAY_OPACITY_NEAR = 1.0
NEAR_THRESHOLD = 0.05        # normalized vertical distance to line for full opacity
FLASH_DURATION_MS = 200      # base flash duration in ms; scaled per band
CONTROLS_AUTO_HIDE_S = 5     # seconds before controls panel auto-hides on launch
