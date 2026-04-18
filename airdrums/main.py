"""
airdrums.main
=============
Entry point for AirDrums V2. Parses CLI, auto-detects hardware, opens camera,
and runs the main loop using MediaPipe Hands + flat drum-line triggers.

Keyboard controls:
    H        toggle controls panel
    D        toggle Depth Anything V2
    M        toggle left/right handed mode
    E        toggle drum line edit mode
    R        start / stop recording
    O        toggle overdub
    Space    play / pause session playback
    Z        undo last take
    S        open settings panel
    Esc      exit
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

from . import config
from .analytics.stats import SessionAnalytics
from .audio.engine import AudioEngine
from .audio.export import export_mp3, export_stems, export_wav
from .midi.export import export_mid
from .midi.output import MidiOutput
from .recording.daw import (export_als, export_flp, export_garageband_folder,
                             export_logicx, export_rpp)
from .recording.session import HitEvent, Session
from .tracking.depth import DepthEngine
from .tracking.detectors import HitResult, LineCrossDetector
from .tracking.hands import HandSkeleton, Joint
from .ui.overlay import Overlay
from .ui.welcome import launch_welcome

log = logging.getLogger("airdrums.main")


# ---------------------------------------------------------------------------
# Hardware profile detection
# ---------------------------------------------------------------------------
def auto_detect_profile() -> config.HardwareProfile:
    """Return the best hardware profile for the current machine."""
    try:
        import torch  # type: ignore
        if torch.cuda.is_available():
            return config.PROFILE_HIGH
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return config.PROFILE_MEDIUM
    except Exception:
        pass
    return config.PROFILE_LOW


# ---------------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------------
class AirDrumsApp:
    """Orchestrates all AirDrums subsystems through the main capture loop."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.profile = self._pick_profile(args.profile)
        log.info("Running with profile=%s", self.profile.name)

        # Depth is disabled when --no-depth is passed
        self._depth_active = not args.no_depth

        # Mirror / handedness
        self._mirror = getattr(args, "mirror", False)

        # Drum lines (static, settable via settings)
        self._drum_lines = config.get_drum_lines(mirrored=self._mirror)

        # Core subsystems
        self.hands = HandSkeleton()
        self.depth = DepthEngine(self.profile)
        self.audio = AudioEngine(pack_name="default")
        self.audio.set_cymbal_lines(self._drum_lines)
        self.midi = MidiOutput()
        self.session = Session(drum_lines=self._drum_lines)
        self.analytics = SessionAnalytics(drum_lines=self._drum_lines)
        self.overlay = Overlay(
            drum_lines=self._drum_lines,
            frame_w=config.CAMERA_WIDTH,
            frame_h=config.CAMERA_HEIGHT,
        )

        # One LineCrossDetector per hand (landmark 8 used directly — no drumstick)
        self.detectors: Dict[str, LineCrossDetector] = {
            "left":  LineCrossDetector("left"),
            "right": LineCrossDetector("right"),
        }

        # Load saved session if requested
        if args.session:
            self._load_session(Path(args.session))

        # Calibration: load saved or force
        self._calibration: Dict = {}
        self._load_calibration()
        if args.calibrate:
            self._calibration_pending = True
        else:
            self._calibration_pending = (
                not (config.PROFILES_DIR / "default.json").exists()
            )

    # ------------------------------------------------------------------
    def _pick_profile(self, flag: str) -> config.HardwareProfile:
        """Resolve --profile flag to a HardwareProfile."""
        if flag in config.PROFILES:
            return config.PROFILES[flag]
        return auto_detect_profile()

    def _load_session(self, path: Path) -> None:
        """Load an existing session JSON, logging any failure."""
        try:
            self.session = Session.from_json(path)
            log.info("Loaded session: %s", path)
        except Exception as exc:
            log.warning("Could not load session %s: %s", path, exc)

    def _load_calibration(self) -> None:
        """Read saved calibration from default profile JSON if it exists."""
        path = config.PROFILES_DIR / "default.json"
        if path.exists():
            try:
                with open(path) as fh:
                    self._calibration = json.load(fh)
                log.info("Loaded calibration profile: %s", path)
            except Exception as exc:
                log.warning("Could not read calibration: %s", exc)

    def _save_calibration(self) -> None:
        """Persist calibration dict to default.json."""
        path = config.PROFILES_DIR / "default.json"
        try:
            with open(path, "w") as fh:
                json.dump(self._calibration, fh, indent=2)
            log.info("Saved calibration to %s", path)
        except Exception as exc:
            log.warning("Could not save calibration: %s", exc)

    # ------------------------------------------------------------------
    def run(self) -> None:
        """Open the webcam and run the main capture loop."""
        cap = cv2.VideoCapture(config.CAMERA_INDEX)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, config.CAMERA_FPS)
        # Natural lighting: let the camera manage exposure automatically
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)   # 1 = auto on most drivers
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
        if not cap.isOpened():
            log.error("Could not open camera %d", config.CAMERA_INDEX)
            return

        # Offer to restore recovery session
        self._maybe_restore_recovery()

        # Check for pending calibration
        if self._calibration_pending:
            self._run_calibration(cap)

        # Start MIDI clock
        self.midi.start_clock()

        window = "AirDrums"
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window, config.CAMERA_WIDTH, config.CAMERA_HEIGHT)
        cv2.setMouseCallback(window, self.overlay.handle_mouse)

        fps_times: List[float] = []

        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    log.warning("Camera frame read failed")
                    break

                frame = cv2.flip(frame, 1)   # mirror for natural drumming perspective
                t_frame = time.monotonic()

                # Depth inference (non-blocking)
                depth_map: Optional[np.ndarray] = None
                if self._depth_active:
                    self.depth.submit_frame(frame)
                    depth_map = self.depth.latest_depth

                # Hand tracking
                hands_data = self.hands.update(
                    frame, depth_map if self._depth_active else None
                )
                # Per-hand: use landmark 8 (INDEX_FINGER_TIP) directly
                hits: List[HitResult] = []
                fingertips: Dict[str, tuple] = {}   # hand_id -> (x, y) for overlay
                for hand_id in ("left", "right"):
                    if hands_data.get(hand_id) is None:
                        continue
                    tip = self.hands.get_index_tip(hand_id)
                    if tip is None:
                        continue

                    fingertips[hand_id] = (tip.x, tip.y)
                    hit = self.detectors[hand_id].check(
                        tip.x, tip.y, tip.vy, tip.speed, self._drum_lines,
                    )
                    if hit is not None:
                        hits.append(hit)

                # Fire all hits (both hands can hit simultaneously)
                for hit in hits:
                    self._fire_hit(hit)

                # Auto-save
                self.session.maybe_autosave(config.SESSIONS_DIR / "current.json")

                # FPS tracking
                fps_times.append(t_frame)
                fps_times = [t for t in fps_times if t_frame - t < 1.0]
                display_fps = len(fps_times)

                # Compose HUD data
                hud_data = {
                    "bpm":           self.analytics.bpm,
                    "stability":     self.analytics.bpm_stability,
                    "drift":         self.analytics.drift,
                    "recording":     self.session._recording,
                    "overdub":       self.session._overdub,
                    "playing":       self.session._playing,
                    "take_count":    len(self.session.takes),
                    "midi_connected": self.midi.connected,
                    "depth_active":  self._depth_active,
                    "mirror_mode":   self._mirror,
                    "fps":           display_fps,
                    "loop_progress": self._loop_progress(),
                }

                # Render overlay
                frame = self.overlay.draw(frame, fingertips, hud_data)

                cv2.imshow(window, frame)

                key = cv2.waitKey(1) & 0xFF
                if self._handle_key(key, cap):
                    break

                # Sync MIDI clock BPM every second
                if self.analytics.bpm > 0:
                    self.midi.set_bpm(self.analytics.bpm)

        finally:
            cap.release()
            cv2.destroyAllWindows()
            self._shutdown()

    # ------------------------------------------------------------------
    def _fire_hit(self, hit: HitResult) -> None:
        """Play audio, send MIDI, record event, and register overlay flash."""
        dl = hit.drum_line
        log.debug(
            "HIT %-12s note=%d vel=%d band=%s hand=%s",
            dl.name, dl.midi_note, hit.velocity,
            config.VELOCITY_BANDS[hit.band_idx].name, hit.hand_id,
        )
        self.audio.play_hit(dl.name, hit.band_idx, hit.velocity)
        self.midi.send_hit(dl.midi_note, hit.velocity)

        band_name = config.VELOCITY_BANDS[hit.band_idx].name
        self.session.record_hit(
            drum_name=dl.name,
            midi_note=dl.midi_note,
            velocity=hit.velocity,
            velocity_band=band_name,
            hand_side=hit.hand_id,
        )

        # Analytics always sees the hit even if not recording
        ev = HitEvent(
            timestamp_ms=time.time() * 1000.0,
            drum_name=dl.name,
            midi_note=dl.midi_note,
            velocity=hit.velocity,
            velocity_band=band_name,
            hand_side=hit.hand_id,
            take_id=-1,
        )
        self.analytics.on_hit(ev)
        self.overlay.register_hit(dl.name, hit.band_idx)

    # ------------------------------------------------------------------
    def _loop_progress(self) -> float:
        """Return 0.0-1.0 indicating position within current loop."""
        if not self.session._recording or self.session._loop_length_ms <= 0:
            return 0.0
        elapsed = time.time() * 1000.0 - self.session._start_time_ms
        return (elapsed % self.session._loop_length_ms) / self.session._loop_length_ms

    # ------------------------------------------------------------------
    def _handle_key(self, key: int, cap) -> bool:
        """Return True to exit the main loop."""
        if key == 27:   # Esc
            return True
        if key == ord('r'):
            if self.session._recording:
                self.session.stop_recording()
                log.info("Recording stopped")
            else:
                self.session.start_recording(
                    loop_bars=self.session._loop_bars,
                    detected_bpm=max(self.analytics.bpm, 60.0),
                )
                log.info("Recording started")
        elif key == ord('o'):
            self.session.toggle_overdub()
            log.info("Overdub: %s", self.session._overdub)
        elif key == ord(' '):
            self.session.toggle_playback()
            log.info("Playback: %s", self.session._playing)
        elif key == ord('z'):
            self.session.undo_last_take()
            log.info("Undid last take")
        elif key == ord('d'):
            self._depth_active = not self._depth_active
            log.info("Depth: %s", "ON" if self._depth_active else "OFF")
        elif key == ord('m'):
            self._mirror = not self._mirror
            self._drum_lines = config.get_drum_lines(mirrored=self._mirror)
            self.overlay.update_drum_lines(self._drum_lines)
            log.info("Mirror mode: %s", self._mirror)
        elif key == ord('e'):
            self.overlay.toggle_edit_mode()
            log.info("Edit mode: %s", self.overlay._edit_mode)
        elif key == ord('h'):
            self.overlay.toggle_controls()
        elif key == ord('s'):
            self._open_settings()
        elif key == ord('x'):
            self._export_all()
        return False

    # ------------------------------------------------------------------
    def _open_settings(self) -> None:
        """Launch the PyQt5 settings panel and apply any changes."""
        from .ui.settings import launch_settings_panel

        config_state = {
            "camera_index": config.CAMERA_INDEX,
            "pack_name": "default",
            "mirror": self._mirror,
        }
        result = launch_settings_panel(config_state, self._drum_lines)
        if result is None:
            return

        if "drum_lines" in result:
            self._drum_lines = result["drum_lines"]
            self.overlay.update_drum_lines(self._drum_lines)
            self.audio.set_cymbal_lines(self._drum_lines)

        log.info("Settings applied")

    # ------------------------------------------------------------------
    def _maybe_restore_recovery(self) -> None:
        """If a recovery file exists, offer to restore it (logs only — no UI prompt in V2)."""
        recovery = config.RECOVERY_DIR / "recovery.json"
        if recovery.exists():
            log.info("Recovery file found at %s — pass --session to restore", recovery)

    # ------------------------------------------------------------------
    def _run_calibration(self, cap) -> None:
        """Interactive 3-stage calibration sequence displayed in the OpenCV window."""
        log.info("=== CALIBRATION START ===")
        calibration: Dict = {}

        def grab() -> Optional[np.ndarray]:
            ok, f = cap.read()
            if not ok:
                return None
            return cv2.flip(f, 1)

        def show(frame: np.ndarray, message: str) -> None:
            annotated = frame.copy()
            cv2.putText(annotated, message, (40, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            cv2.imshow("AirDrums", annotated)
            cv2.waitKey(1)

        # --- Stage 1: Depth scale ---
        log.info("Stage 1: Stand 150 cm from camera. Press any key when ready.")
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            f = grab()
            if f is not None:
                show(f, "CALIBRATION 1/3: Stand 150 cm away | waiting 5s...")
                self.depth.submit_frame(f)
        depth_map = self.depth.latest_depth
        hands_data = self.hands.update(grab() or np.zeros((config.CAMERA_HEIGHT, config.CAMERA_WIDTH, 3), np.uint8), depth_map)
        for hand_id in ("left", "right"):
            wrist = self.hands.get_wrist(hand_id)
            if wrist is not None and wrist.z_depth > 0:
                calibration["depth_scale"] = 150.0 / wrist.z_depth
                log.info("Depth scale: %.3f", calibration["depth_scale"])
                break

        # --- Stage 2: Strike threshold ---
        log.info("Stage 2: Hit each drum line 3x at normal force. 10 seconds.")
        speeds: List[float] = []
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            f = grab()
            if f is None:
                continue
            show(f, "CALIBRATION 2/3: Strike drum lines 3x each (10s)")
            dm = self.depth.latest_depth if self._depth_active else None
            self.hands.update(f, dm)
            for hand_id in ("left", "right"):
                tip = self.hands.get_index_tip(hand_id)
                if tip and tip.speed > 0.01:
                    speeds.append(tip.speed)
        if speeds:
            speeds_sorted = sorted(speeds)
            idx = int(len(speeds_sorted) * 0.30)
            calibration["strike_threshold"] = speeds_sorted[idx]
            log.info("Strike threshold (30th pct): %.3f", calibration["strike_threshold"])

        # --- Stage 3: Line fit confirmation ---
        log.info("Stage 3: Hold stick tips at each drum line for 5 seconds.")
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            f = grab()
            if f is not None:
                show(f, "CALIBRATION 3/3: Hold tips at each drum line (5s)")
                cv2.waitKey(1)
        log.info("Line fit confirmed (using default layout)")
        calibration["line_fit_ok"] = True

        self._calibration = calibration
        self._save_calibration()
        self._calibration_pending = False
        log.info("=== CALIBRATION COMPLETE ===")

    # ------------------------------------------------------------------
    def _export_all(self) -> None:
        """Export MIDI, WAV, stems, analytics, and DAW project files."""
        out = config.EXPORTS_DIR / time.strftime("session_%Y%m%d_%H%M%S")
        out.mkdir(parents=True, exist_ok=True)
        log.info("Exporting to %s", out)
        events = self.session.get_active_events()
        bpm_history = self.session.bpm_history

        try:
            self.session.export_json(out / "session.json")
        except Exception as exc:
            log.warning("Session JSON export failed: %s", exc)

        try:
            mid_path = out / "AirDrums.mid"
            export_mid(events, self._drum_lines, bpm_history, mid_path)
        except Exception as exc:
            log.warning("MIDI export failed: %s", exc)
            mid_path = None

        try:
            wav_path = out / "mix.wav"
            export_wav(events, self._drum_lines, "default", wav_path)
        except Exception as exc:
            log.warning("WAV export failed: %s", exc)
            wav_path = None

        if wav_path and wav_path.exists():
            try:
                export_mp3(wav_path, out / "mix.mp3")
            except Exception as exc:
                log.warning("MP3 export failed (ffmpeg installed?): %s", exc)

        stems: List[Path] = []
        try:
            stems = export_stems(events, self._drum_lines, "default", out / "stems")
        except Exception as exc:
            log.warning("Stem export failed: %s", exc)

        try:
            analytics_dir = out / "analytics"
            analytics_dir.mkdir(exist_ok=True)
            self.analytics.hit_heatmap(events, analytics_dir / "heatmap.png")
            self.analytics.velocity_histogram(events, analytics_dir / "velocity.png")
            self.analytics.export_pdf(events, out / "report.pdf")
        except Exception as exc:
            log.warning("Analytics export failed: %s", exc)

        if stems and mid_path:
            try:
                export_als(stems, mid_path, out / "project.als", self._drum_lines)
                export_flp(stems, mid_path, out / "project.flp", self._drum_lines)
                export_logicx(stems, mid_path, out / "project.logicx", self._drum_lines)
                export_rpp(stems, mid_path, out / "project.rpp", self._drum_lines)
                export_garageband_folder(stems, mid_path, out / "GarageBand", self._drum_lines)
            except Exception as exc:
                log.warning("DAW export failed: %s", exc)

        log.info("Export complete: %s", out)

    # ------------------------------------------------------------------
    def _shutdown(self) -> None:
        """Release all resources cleanly."""
        self.midi.close()
        self.depth.stop()
        self.audio.close()
        self.hands.close()
        log.info("AirDrums shut down")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    p = argparse.ArgumentParser(
        prog="airdrums",
        description="AirDrums V2: computer-vision drum controller",
    )
    p.add_argument("--profile", choices=["low", "medium", "high", "auto"],
                   default="auto", help="Force hardware profile or auto-detect")
    p.add_argument("--no-depth", action="store_true",
                   help="Disable Depth Anything V2 on launch")
    p.add_argument("--calibrate", action="store_true",
                   help="Force recalibration even with saved profile")
    p.add_argument("--session", type=str, default=None,
                   help="Load existing session JSON")
    p.add_argument("--mirror", action="store_true",
                   help="Launch in left-handed (mirrored) mode")
    return p


# def main(argv: Optional[List[str]] = None) -> int:
#     """Entry point — configure logging, parse args, run app."""
#     logging.basicConfig(level=config.LOG_LEVEL, format=config.LOG_FORMAT)
#     args = build_parser().parse_args(argv)
#     try:
#         app = AirDrumsApp(args)
#         app.run()
#     except KeyboardInterrupt:
#         log.info("Interrupted by user")
#     except Exception as exc:
#         log.exception("Fatal error: %s", exc)
#         return 1
#     return 0
def main(argv: Optional[list] = None) -> int:
    logging.basicConfig(level=config.LOG_LEVEL, format=config.LOG_FORMAT)
    args = build_parser().parse_args(argv)

    # Show welcome screen first; on_start is called when user clicks Start.
    # launch_welcome blocks until the window is closed.
    # on_start receives the fully-built AirDrumsApp from the loading screen
    def on_start(app, settings: dict) -> None:
        if app is None:
            log.error("Initialisation failed; cannot start.")
            return
        try:
            app.run()
        except KeyboardInterrupt:
            log.info("Interrupted by user")
        except Exception as exc:  # noqa: BLE001
            log.exception("Fatal: %s", exc)
        finally:
            try:
                app.shutdown()
            except Exception:  # noqa: BLE001
                pass

    launch_welcome(on_start=on_start, initial={}, args=args)
    return 0

if __name__ == "__main__":
    sys.exit(main())
