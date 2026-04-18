"""
airdrums.main
=============
Entry point. Sets up logging, parses CLI, auto-detects a hardware profile,
opens the webcam + MIDI + audio, then runs the main loop.

Keyboard controls in the OpenCV window::

    q        quit
    r        toggle recording
    o        toggle overdub
    l        cycle loop length (1/2/4/8 bars)
    t        new take
    u        undo last take
    d        toggle depth map overlay
    c        re-run calibration
    [ / ]    shrink / grow drumstick length
    s        open settings panel (PyQt5)
    e        export session (MIDI + WAV + stems + PDF + DAW projects)
    h        cycle HUD theme
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from . import config
from .analytics.stats import SessionAnalytics
from .audio.engine import AudioEngine
from .audio.export import export_mp3, export_stems, export_wav
from .calibration import Calibrator, load_profile
from .midi.export import export_mid
from .midi.output import MidiOutput
from .recording.bvh import export_bvh
from .recording.daw import (export_als, export_flp, export_garageband_folder,
                            export_logicx, export_rpp)
from .recording.session import Session
from .tracking.depth import DepthEngine
from .tracking.detectors import FootPedalDetector, VelocitySpikeDetector
from .tracking.drumstick import Drumstick
from .tracking.skeleton import DrumSkeleton
from .ui.overlay import Overlay

log = logging.getLogger("airdrums.main")


# ---------------------------------------------------------------------------
# Profile detection
# ---------------------------------------------------------------------------
def auto_detect_profile() -> config.HardwareProfile:
    try:
        import torch  # type: ignore
        if torch.cuda.is_available():
            return config.PROFILE_HIGH
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return config.PROFILE_MEDIUM
    except Exception:  # noqa: BLE001
        pass
    return config.PROFILE_LOW


# ---------------------------------------------------------------------------
# Zone resolution
# ---------------------------------------------------------------------------
def resolve_zone(tip_x: float, tip_y: float, zones: List[config.DrumZone]
                 ) -> Optional[config.DrumZone]:
    for z in zones:
        if z.is_pedal:
            continue
        if z.x_range[0] <= tip_x <= z.x_range[1] and z.y_range[0] <= tip_y <= z.y_range[1]:
            return z
    return None


def resolve_pedal_zone(foot_x: float, foot_y: float, zones: List[config.DrumZone],
                       foot_side: str) -> Optional[config.DrumZone]:
    for z in zones:
        if not z.is_pedal:
            continue
        if foot_side == "right" and z.midi_note != 36:
            continue
        if foot_side == "left" and z.midi_note != 44:
            continue
        if z.x_range[0] <= foot_x <= z.x_range[1] and z.y_range[0] <= foot_y <= z.y_range[1]:
            return z
    return None


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------
class AirDrumsApp:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.profile = self._pick_profile(args.profile)
        if args.no_depth:
            self.profile = config.HardwareProfile(
                name=self.profile.name, depth_model=self.profile.depth_model,
                depth_target_fps=0, pose_complexity=self.profile.pose_complexity,
                depth_enabled=False,
            )
        log.info("Running with profile=%s", self.profile.name)

        # Core components
        self.skeleton = DrumSkeleton(pose_complexity=self.profile.pose_complexity)
        self.depth = DepthEngine(self.profile, enabled=self.profile.depth_enabled)
        self.audio = AudioEngine(sample_pack="default")
        self.midi = MidiOutput()
        self.session = Session(sample_pack="default")
        self.analytics = SessionAnalytics(self.session)
        self.overlay = Overlay()

        # Two drumsticks
        self.sticks = {
            "left": Drumstick("left", args.stick_length),
            "right": Drumstick("right", args.stick_length),
        }
        self.strike_detectors = {
            "left": VelocitySpikeDetector(side="left"),
            "right": VelocitySpikeDetector(side="right"),
        }
        self.pedal_detectors = {
            "right": FootPedalDetector("right"),
            "left": FootPedalDetector("left"),
        }

        # Apply saved profile values
        self._apply_saved_calibration()

    # ------------------------------------------------------------------
    def _pick_profile(self, flag: str) -> config.HardwareProfile:
        if flag in ("low", "medium", "high"):
            return config.PROFILES[flag]
        return auto_detect_profile()

    def _apply_saved_calibration(self) -> None:
        prof = load_profile()
        if not prof:
            return
        self.session.calibration.update(prof)
        if "depth_scale" in prof:
            self.depth.set_depth_scale(prof["depth_scale"])
            self.skeleton.depth_scale = prof["depth_scale"]
        if "spike_threshold" in prof:
            for d in self.strike_detectors.values():
                d.spike_threshold = float(prof["spike_threshold"])
        if "pedal_threshold" in prof:
            for d in self.pedal_detectors.values():
                d.set_threshold(float(prof["pedal_threshold"]))
        log.info("Applied saved calibration: %s", prof)

    # ------------------------------------------------------------------
    def run(self) -> None:
        import cv2

        cap = cv2.VideoCapture(config.CAMERA_INDEX)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, config.CAMERA_FPS)
        if not cap.isOpened():
            log.error("Could not open camera %d", config.CAMERA_INDEX)
            return

        # Load existing session if requested
        if self.args.session:
            try:
                self.session = Session.from_json(self.args.session)
                self.analytics = SessionAnalytics(self.session)
                log.info("Loaded session: %s", self.args.session)
            except Exception as exc:  # noqa: BLE001
                log.warning("Could not load session %s: %s", self.args.session, exc)

        # Forced calibration
        if self.args.calibrate or not config.DEFAULT_PROFILE_PATH.exists():
            self._run_calibration(cap)

        # Start clock
        self.midi.set_bpm(self.analytics.bpm)
        self.midi.start_clock()

        window = "AirDrums"
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window, config.CAMERA_WIDTH, config.CAMERA_HEIGHT)

        last_bpm_sync = time.time()
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    log.warning("Camera frame read failed")
                    break
                frame = cv2.flip(frame, 1)     # mirror for natural interaction

                self.depth.submit(frame)
                depth_map = self.depth.latest_depth
                self.skeleton.update(frame, depth_map)

                # Update drumsticks
                self._update_sticks()
                zones = self.skeleton.rescale_zones(config.DRUM_ZONES)

                # Upper-body strike detection via stick tips
                self._process_strikes(zones)
                # Foot pedals
                self._process_pedals(zones)

                # Record skeleton keyframes
                joints_xyzv = {
                    name: (j.x, j.y, j.z_depth, j.visibility)
                    for name, j in self.skeleton.joints.items()
                }
                self.session.record_skeleton(joints_xyzv)

                # HUD state
                hud_state = {
                    "bpm": self.analytics.bpm,
                    "stability": self.analytics.bpm_stability,
                    "drift": self.analytics.drift,
                    "status": ("REC" if self.session.is_recording else "idle"),
                    "take": self.session._active_take_id,
                    "hihat": self.pedal_detectors["left"].hihat_state,
                    "midi_connected": self.midi.connected,
                    "stick_length": self.sticks["right"].length,
                    "loop_progress": self._loop_progress(),
                }
                self.overlay.render(frame, self.skeleton, list(self.sticks.values()),
                                    zones, hud_state, depth_map)

                cv2.imshow(window, frame)
                key = cv2.waitKey(1) & 0xFF
                if self._handle_key(key, cap):
                    break

                # Sync MIDI clock with live BPM every second
                if time.time() - last_bpm_sync > 1.0:
                    self.midi.set_bpm(self.analytics.bpm)
                    last_bpm_sync = time.time()
        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.shutdown()

    # ------------------------------------------------------------------
    def _update_sticks(self) -> None:
        lw = self.skeleton.joints.get("left_wrist")
        le = self.skeleton.joints.get("left_elbow")
        rw = self.skeleton.joints.get("right_wrist")
        re = self.skeleton.joints.get("right_elbow")
        if lw is not None:
            self.sticks["left"].update(lw, le)
        if rw is not None:
            self.sticks["right"].update(rw, re)

    # ------------------------------------------------------------------
    def _process_strikes(self, zones: List[config.DrumZone]) -> None:
        for side, stick in self.sticks.items():
            if not stick.visible:
                continue
            tip_joint = stick.to_joint()
            elbow = self.skeleton.joints.get(f"{side}_elbow")
            event, velocity, is_rim = self.strike_detectors[side].update(tip_joint, elbow)
            if event != "strike":
                continue
            zone = resolve_zone(stick.tip_x, stick.tip_y, zones)
            if zone is None:
                continue
            note = zone.midi_note
            if is_rim and zone.name == "Snare":
                note = config.SNARE_RIMSHOT_NOTE
            self._fire_hit(zone.name, note, velocity, side, is_rim, (stick.tip_x, stick.tip_y))

    # ------------------------------------------------------------------
    def _process_pedals(self, zones: List[config.DrumZone]) -> None:
        heel_l = self.skeleton.joints.get("left_heel")
        toe_l = self.skeleton.joints.get("left_foot_index")
        ankle_l = self.skeleton.joints.get("left_ankle")
        heel_r = self.skeleton.joints.get("right_heel")
        toe_r = self.skeleton.joints.get("right_foot_index")
        ankle_r = self.skeleton.joints.get("right_ankle")

        ev_r = self.pedal_detectors["right"].update(heel_r, toe_r, ankle_r)
        if ev_r and ev_r.kind == "strike":
            self._fire_hit("Bass Drum", 36, ev_r.velocity, "foot_right", False,
                           (heel_r.x if heel_r else 0.5, heel_r.y if heel_r else 0.9))

        ev_l = self.pedal_detectors["left"].update(heel_l, toe_l, ankle_l)
        if ev_l is not None:
            # Push state into audio engine
            self.audio.set_hihat_state(self.pedal_detectors["left"].hihat_state)
            if ev_l.kind == "strike":
                self._fire_hit("Hi-Hat Pedal", 44, ev_l.velocity, "foot_left", False,
                               (heel_l.x if heel_l else 0.2, heel_l.y if heel_l else 0.9))
                self.audio.play_pedal_chick()

    # ------------------------------------------------------------------
    def _fire_hit(self, drum_name: str, midi_note: int, velocity: int,
                  hand_side: str, is_rim: bool, tip_xy) -> None:
        log.debug("HIT %-12s note=%d vel=%d (%s)", drum_name, midi_note, velocity, hand_side)
        self.audio.play(drum_name, velocity)
        self.midi.send_hit(midi_note, velocity)
        ev = self.session.record_hit(drum_name, midi_note, velocity, hand_side, is_rim)
        if ev is None:
            # Still let analytics see the hit even if we aren't recording
            from .recording.session import HitEvent
            ev = HitEvent(timestamp_ms=time.time() * 1000.0, drum_name=drum_name,
                          midi_note=midi_note, velocity=velocity, hand_side=hand_side,
                          is_rimshot=is_rim)
        self.analytics.on_hit(ev, tip_xy=tip_xy)
        self.overlay.register_hit(drum_name, velocity, is_rim)
        if hand_side in ("left", "right"):
            self.overlay.register_stick_strike(hand_side)

    # ------------------------------------------------------------------
    def _loop_progress(self) -> float:
        if not self.session.loop_enabled or not self.session.is_recording:
            return 0.0
        loop_len = self.session._loop_length_ms()
        if loop_len <= 0:
            return 0.0
        now_ms = time.time() * 1000.0 - (self.session._record_start_ms or 0)
        return (now_ms % loop_len) / loop_len

    # ------------------------------------------------------------------
    def _handle_key(self, key: int, cap) -> bool:
        import cv2
        if key == ord('q'):
            return True
        if key == ord('r'):
            if self.session.is_recording:
                self.session.stop_recording()
            else:
                self.session.start_recording()
        elif key == ord('o'):
            self.session.overdub = not self.session.overdub
            log.info("Overdub: %s", self.session.overdub)
        elif key == ord('l'):
            idx = config.LOOP_BAR_CHOICES.index(self.session.loop_bars) \
                  if self.session.loop_bars in config.LOOP_BAR_CHOICES else 0
            idx = (idx + 1) % len(config.LOOP_BAR_CHOICES)
            self.session.loop_bars = config.LOOP_BAR_CHOICES[idx]
            self.session.loop_enabled = True
            log.info("Loop bars: %d", self.session.loop_bars)
        elif key == ord('t'):
            self.session.new_take()
        elif key == ord('u'):
            self.session.undo_last_take()
        elif key == ord('d'):
            self.overlay.show_depth = not self.overlay.show_depth
        elif key == ord('c'):
            self._run_calibration(cap)
        elif key == ord('['):
            for s in self.sticks.values():
                s.set_length(s.length - 0.01)
        elif key == ord(']'):
            for s in self.sticks.values():
                s.set_length(s.length + 0.01)
        elif key == ord('h'):
            themes = list(config.HUD_THEMES)
            i = themes.index(self.overlay.theme) if self.overlay.theme in themes else 0
            self.overlay.set_theme(themes[(i + 1) % len(themes)])
        elif key == ord('s'):
            from .ui.settings import launch_settings_panel
            launch_settings_panel(on_apply=self._apply_settings,
                                  initial={"stick_length": self.sticks["right"].length})
        elif key == ord('e'):
            self._export_all()
        return False

    # ------------------------------------------------------------------
    def _apply_settings(self, data: Dict) -> None:
        if "stick_length" in data:
            for s in self.sticks.values():
                s.set_length(float(data["stick_length"]))
        if "theme" in data:
            self.overlay.set_theme(data["theme"])
        if "velocity_curve" in data:
            for d in self.strike_detectors.values():
                d.set_velocity_curve(data["velocity_curve"])

    # ------------------------------------------------------------------
    def _run_calibration(self, cap) -> None:
        import cv2

        def poll() -> Dict:
            ok, f = cap.read()
            if not ok:
                return {}
            f = cv2.flip(f, 1)
            self.depth.submit(f)
            self.skeleton.update(f, self.depth.latest_depth)
            self._update_sticks()
            tips = [self.sticks["left"].to_joint(), self.sticks["right"].to_joint()]
            cv2.imshow("AirDrums", f)
            cv2.waitKey(1)
            return {"joints": self.skeleton.joints, "tips": tips}

        def prompt(text: str) -> None:
            log.info(text)

        cal = Calibrator(poll, prompt)
        result = cal.run()
        self.session.calibration.update(result)
        # Propagate
        if "depth_scale" in result:
            self.depth.set_depth_scale(result["depth_scale"])
            self.skeleton.depth_scale = result["depth_scale"]
        if "spike_threshold" in result:
            for d in self.strike_detectors.values():
                d.spike_threshold = result["spike_threshold"]
        if "pedal_threshold" in result:
            for d in self.pedal_detectors.values():
                d.set_threshold(result["pedal_threshold"])

    # ------------------------------------------------------------------
    def _export_all(self) -> None:
        out = config.EXPORTS_DIR / time.strftime("session_%Y%m%d_%H%M%S")
        out.mkdir(parents=True, exist_ok=True)
        bpm = self.analytics.bpm
        try:
            self.session.export_json(out / "session.json")
            mid_path = export_mid(self.session, bpm, out / "AirDrums.mid")
            wav_path = export_wav(self.session, self.session.sample_pack, out / "mix.wav")
            try:
                export_mp3(wav_path, out / "mix.mp3")
            except Exception as exc:  # noqa: BLE001
                log.warning("MP3 export failed (ffmpeg installed?): %s", exc)
            stems = export_stems(self.session, self.session.sample_pack, out / "stems")
            # Analytics
            self.analytics.export_heatmaps(out / "analytics")
            self.analytics.export_velocity_histograms(out / "analytics")
            self.analytics.export_pdf(out / "report.pdf",
                                      include_png_dir=out / "analytics")
            # BVH
            try:
                export_bvh(self.session, out / "motion.bvh")
            except Exception as exc:  # noqa: BLE001
                log.warning("BVH export failed: %s", exc)
            # DAW projects
            export_als(stems, mid_path, out / "project.als")
            export_flp(stems, mid_path, out / "project.flp")
            export_logicx(stems, mid_path, out / "project.logicx")
            export_rpp(stems, mid_path, out / "project.rpp")
            export_garageband_folder(stems, mid_path, out / "GarageBand")
            log.info("Exported full session to %s", out)
        except Exception as exc:  # noqa: BLE001
            log.exception("Export failed: %s", exc)

    # ------------------------------------------------------------------
    def shutdown(self) -> None:
        self.midi.close()
        self.depth.stop()
        self.audio.shutdown()
        self.skeleton.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="airdrums",
                                description="AirDrums: CV drum controller")
    p.add_argument("--profile", choices=["low", "medium", "high", "auto"],
                   default="auto", help="Force a hardware profile or auto-detect")
    p.add_argument("--no-depth", action="store_true",
                   help="Disable Depth Anything V2; use MediaPipe z only")
    p.add_argument("--calibrate", action="store_true",
                   help="Force recalibration even if profile exists")
    p.add_argument("--session", type=str, default=None,
                   help="Load an existing session JSON to resume or replay")
    p.add_argument("--stick-length", type=float,
                   default=config.DRUMSTICK_LENGTH_DEFAULT,
                   dest="stick_length",
                   help="Override drumstick length (default 0.18)")
    return p


def main(argv: Optional[list] = None) -> int:
    logging.basicConfig(level=config.LOG_LEVEL, format=config.LOG_FORMAT)
    args = build_parser().parse_args(argv)
    try:
        app = AirDrumsApp(args)
        app.run()
    except KeyboardInterrupt:
        log.info("Interrupted by user")
    except Exception as exc:  # noqa: BLE001
        log.exception("Fatal: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
