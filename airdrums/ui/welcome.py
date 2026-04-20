"""
airdrums.ui.welcome
===================
Welcome screen shown on startup. Displays the app name, a brief
description, and a start button that transitions to a loading screen.

The loading screen builds AirDrumsApp piece by piece in a background
thread, emitting progress labels as each subsystem initialises. When
complete it immediately calls on_start with the fully built app object.

Usage (from main.py):
    from ui.welcome import launch_welcome

    def on_start(app, settings):
        app.run()

    launch_welcome(on_start=on_start, args=parsed_args)
"""
from __future__ import annotations

import logging
import math
from typing import Callable, Dict, Optional

from .. import config

log = logging.getLogger(__name__)


def launch_welcome(
    on_start: Optional[Callable] = None,
    initial: Optional[Dict] = None,
    args=None,
) -> None:
    """Open the welcome screen. Blocks until the user starts or closes."""
    try:
        from PyQt5 import QtCore, QtGui, QtWidgets  # type: ignore
    except Exception as exc:  # noqa: BLE001
        log.warning("PyQt5 unavailable (%s); skipping welcome screen", exc)
        if on_start is not None:
            on_start(None, initial or {})
        return

    initial = initial or {}

    # ------------------------------------------------------------------
    # Animated drum logo
    # ------------------------------------------------------------------
    class LogoWidget(QtWidgets.QWidget):
        def __init__(self, size: int = 88):
            super().__init__()
            self.setFixedSize(size, size)
            self._phase = 0.0
            t = QtCore.QTimer(self)
            t.timeout.connect(self._tick)
            t.start(16)

        def _tick(self):
            self._phase = (self._phase + 0.045) % (2 * math.pi)
            self.update()

        def paintEvent(self, _):  # noqa: N802
            p = QtGui.QPainter(self)
            p.setRenderHint(QtGui.QPainter.Antialiasing)
            w, h = self.width(), self.height()
            cx, cy = w // 2, h // 2

            pulse = 0.5 + 0.5 * math.sin(self._phase)
            p.setPen(QtGui.QPen(QtGui.QColor(137, 180, 250, int(50 + 50 * pulse)), 1.5))
            p.setBrush(QtCore.Qt.NoBrush)
            p.drawEllipse(4, 4, w - 8, h - 8)

            p.setPen(QtGui.QPen(QtGui.QColor(137, 180, 250), 2))
            p.setBrush(QtGui.QBrush(QtGui.QColor(25, 25, 40)))
            p.drawEllipse(cx - 26, cy - 4, 52, 30)

            p.setPen(QtGui.QPen(QtGui.QColor(180, 210, 255), 1.5))
            p.setBrush(QtGui.QBrush(QtGui.QColor(40, 45, 65)))
            p.drawEllipse(cx - 26, cy - 16, 52, 20)

            ls_y = int(cy - 28 - 7 * max(0.0, math.sin(self._phase * 1.2)))
            p.setPen(QtGui.QPen(QtGui.QColor(205, 214, 244), 2.5,
                                QtCore.Qt.SolidLine, QtCore.Qt.RoundCap))
            p.drawLine(cx - 14, ls_y, cx - 8, cy - 8)

            rs_y = int(cy - 28 - 7 * max(0.0, math.sin(self._phase * 1.2 + 1.6)))
            p.drawLine(cx + 14, rs_y, cx + 8, cy - 8)
            p.end()

    # ------------------------------------------------------------------
    # Shared background painter
    # ------------------------------------------------------------------
    def _paint_bg(widget, painter):
        painter.fillRect(widget.rect(), QtGui.QColor(13, 13, 18))
        painter.setPen(QtGui.QPen(QtGui.QColor(28, 28, 42), 1))
        for x in range(0, widget.width() + 36, 36):
            for y in range(0, widget.height() + 36, 36):
                painter.drawPoint(x, y)
        grad = QtGui.QRadialGradient(
            widget.width() / 2, widget.height() / 2,
            max(widget.width(), widget.height()) * 0.6,
        )
        grad.setColorAt(0, QtGui.QColor(137, 180, 250, 18))
        grad.setColorAt(1, QtGui.QColor(137, 180, 250, 0))
        painter.fillRect(widget.rect(), QtGui.QBrush(grad))

    # ------------------------------------------------------------------
    # Spinner widget
    # ------------------------------------------------------------------
    class SpinnerWidget(QtWidgets.QWidget):
        def __init__(self):
            super().__init__()
            self.setFixedSize(36, 36)
            self._angle = 0
            t = QtCore.QTimer(self)
            t.timeout.connect(self._tick)
            t.start(16)

        def _tick(self):
            self._angle = (self._angle + 6) % 360
            self.update()

        def paintEvent(self, _):  # noqa: N802
            p = QtGui.QPainter(self)
            p.setRenderHint(QtGui.QPainter.Antialiasing)
            p.translate(18, 18)
            p.rotate(self._angle)
            for i in range(8):
                alpha = int(30 + 225 * (i / 8))
                p.setPen(QtCore.Qt.NoPen)
                p.setBrush(QtGui.QColor(137, 180, 250, alpha))
                p.drawEllipse(-3, -14, 6, 6)
                p.rotate(45)
            p.end()

    # ------------------------------------------------------------------
    # Thin progress bar
    # ------------------------------------------------------------------
    class ProgressBar(QtWidgets.QWidget):
        def __init__(self):
            super().__init__()
            self.setFixedHeight(3)
            self._progress = 0.0

        def set_progress(self, value: float):
            self._progress = max(0.0, min(1.0, value))
            self.update()

        def paintEvent(self, _):  # noqa: N802
            p = QtGui.QPainter(self)
            p.fillRect(self.rect(), QtGui.QColor(30, 30, 46))
            fill_w = int(self.width() * self._progress)
            if fill_w > 0:
                p.fillRect(0, 0, fill_w, self.height(), QtGui.QColor(137, 180, 250))
            p.end()

    # ------------------------------------------------------------------
    # Init thread — builds AirDrumsApp subsystem by subsystem
    # ------------------------------------------------------------------
    class InitThread(QtCore.QThread):
        step_done = QtCore.pyqtSignal(str, int)  # (label, steps_done)
        all_done  = QtCore.pyqtSignal(object)    # fully built app (or None on error)

        TOTAL_STEPS = 8

        def __init__(self, build_args, settings: Dict):
            super().__init__()
            self._args = build_args
            self._settings = settings
            self._step = 0

        def _emit(self, label: str):
            self._step += 1
            self.step_done.emit(label, self._step)

        def run(self):  # noqa: C901
            try:
                import cv2
                from pathlib import Path
                from .. import config as cfg
                from ..main import auto_detect_profile, AirDrumsApp
                from ..tracking.hands import HandSkeleton
                from ..tracking.depth import DepthEngine
                from ..audio.engine import AudioEngine
                from ..midi.output import MidiOutput
                from ..recording.session import Session
                from ..analytics.stats import SessionAnalytics
                from ..ui.overlay import Overlay
                from ..tracking.detectors import LineCrossDetector

                # Bypass __init__ — build each subsystem individually
                app = object.__new__(AirDrumsApp)
                app.args = self._args

                # 1 — hardware profile
                self._emit("Detecting hardware profile…")
                app.profile = (
                    cfg.PROFILES[self._args.profile]
                    if self._args.profile in cfg.PROFILES
                    else auto_detect_profile()
                )
                app._depth_active = not getattr(self._args, "no_depth", False)
                app._mirror = getattr(self._args, "mirror", False)
                app._drum_lines = cfg.get_drum_lines(mirrored=app._mirror)
                app._calibration = {}
                app._calibration_pending = False

                # 2 — hand tracking / MediaPipe
                self._emit("Loading hand tracking…")
                app.hands = HandSkeleton()

                # 3 — depth engine
                self._emit("Starting depth engine…")
                app.depth = DepthEngine(app.profile)

                # 4 — audio
                self._emit("Loading audio engine…")
                app.audio = AudioEngine(pack_name="default")
                app.audio.set_cymbal_lines(app._drum_lines)

                # 5 — MIDI
                self._emit("Opening MIDI port…")
                app.midi = MidiOutput()

                # 6 — session / analytics / overlay / detectors
                self._emit("Preparing session…")
                app.session = Session(drum_lines=app._drum_lines)
                app.analytics = SessionAnalytics(drum_lines=app._drum_lines)
                app.overlay = Overlay(
                    drum_lines=app._drum_lines,
                    frame_w=cfg.CAMERA_WIDTH,
                    frame_h=cfg.CAMERA_HEIGHT,
                )
                app.detectors = {
                    "left":  LineCrossDetector("left"),
                    "right": LineCrossDetector("right"),
                }

                # 7 — calibration
                self._emit("Applying calibration…")
                app._load_calibration()
                app._calibration_pending = (
                    getattr(self._args, "calibrate", False)
                    or not (cfg.PROFILES_DIR / "default.json").exists()
                )
                if getattr(self._args, "session", None):
                    app._load_session(Path(self._args.session))

                # 8 — camera (blocking — do last so UI is ready instantly)
                self._emit("Opening camera…")
                cap = cv2.VideoCapture(cfg.CAMERA_INDEX)
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.CAMERA_WIDTH)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.CAMERA_HEIGHT)
                cap.set(cv2.CAP_PROP_FPS, cfg.CAMERA_FPS)
                cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
                cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
                app._preopen_cap = cap  # run() will use this instead of reopening

                self.all_done.emit(app)

            except Exception as exc:  # noqa: BLE001
                log.exception("Init thread failed: %s", exc)
                self.all_done.emit(None)

    # ------------------------------------------------------------------
    # Loading screen
    # ------------------------------------------------------------------
    class LoadingScreen(QtWidgets.QWidget):
        def __init__(self, settings: Dict, build_args):
            super().__init__()
            self.setWindowTitle("Not Quite My Tempo...")
            self.setFixedSize(620, 480)
            self._settings = settings
            self._build_args = build_args
            self._build()
            # Start thread after the window has painted at least once
            QtCore.QTimer.singleShot(50, self._start_thread)

        def paintEvent(self, _):  # noqa: N802
            p = QtGui.QPainter(self)
            _paint_bg(self, p)
            p.end()

        def _build(self):
            root = QtWidgets.QVBoxLayout(self)
            root.setContentsMargins(64, 0, 64, 52)
            root.setSpacing(0)
            root.addStretch(3)

            title = QtWidgets.QLabel("AirDrums")
            title.setAlignment(QtCore.Qt.AlignHCenter)
            title.setStyleSheet(
                "font-size: 28px; font-weight: 800; "
                "letter-spacing: -1px; color: #313244;"
            )
            root.addWidget(title)
            root.addSpacing(36)

            self._spinner = SpinnerWidget()
            root.addWidget(self._spinner, 0, QtCore.Qt.AlignHCenter)
            root.addSpacing(28)

            self._step_lbl = QtWidgets.QLabel("Initialising…")
            self._step_lbl.setAlignment(QtCore.Qt.AlignHCenter)
            self._step_lbl.setStyleSheet("font-size: 13px; color: #6c7086;")
            root.addWidget(self._step_lbl)
            root.addSpacing(24)

            self._bar = ProgressBar()
            root.addWidget(self._bar)
            root.addSpacing(10)

            total = InitThread.TOTAL_STEPS
            self._counter_lbl = QtWidgets.QLabel(f"0 / {total}")
            self._counter_lbl.setAlignment(QtCore.Qt.AlignHCenter)
            self._counter_lbl.setStyleSheet("font-size: 11px; color: #45475a;")
            root.addWidget(self._counter_lbl)

            root.addStretch(2)

        def _start_thread(self):
            self._thread = InitThread(self._build_args, self._settings)
            self._thread.step_done.connect(self._on_step)
            self._thread.all_done.connect(self._on_done)
            self._thread.start()

        def _on_step(self, label: str, done: int):
            total = InitThread.TOTAL_STEPS
            self._step_lbl.setText(label)
            self._bar.set_progress(done / total)
            self._counter_lbl.setText(f"{done} / {total}")

        def _on_done(self, app):
            total = InitThread.TOTAL_STEPS
            self._bar.set_progress(1.0)
            self._step_lbl.setText("Ready")
            self._counter_lbl.setText(f"{total} / {total}")
            # Hand off on next event loop tick — no artificial delay
            QtCore.QTimer.singleShot(0, lambda: self._handoff(app))

        def _handoff(self, app):
            self.close()
            if on_start is not None:
                on_start(app, self._settings)

    # ------------------------------------------------------------------
    # Welcome window
    # ------------------------------------------------------------------
    class WelcomeWindow(QtWidgets.QWidget):
        def __init__(self):
            super().__init__()
            self.setWindowTitle("Not Quite My Tempo...")
            self.setMinimumSize(560, 420)
            self.resize(620, 480)
            self._settings_data: Dict = dict(initial)
            self._build()

        def paintEvent(self, _):  # noqa: N802
            p = QtGui.QPainter(self)
            _paint_bg(self, p)
            p.end()

        def _build(self):
            root = QtWidgets.QVBoxLayout(self)
            root.setContentsMargins(48, 36, 48, 32)
            root.setSpacing(0)

            top_bar = QtWidgets.QHBoxLayout()
            top_bar.addStretch()
            ver = QtWidgets.QLabel("v0.1.0")
            ver.setStyleSheet("color: #45475a; font-size: 11px;")
            top_bar.addWidget(ver)
            root.addLayout(top_bar)

            root.addStretch(1)

            self._logo = LogoWidget()
            root.addWidget(self._logo, 0, QtCore.Qt.AlignHCenter)
            root.addSpacing(24)

            title = QtWidgets.QLabel("Not Quite My Tempo...")
            title.setAlignment(QtCore.Qt.AlignHCenter)
            title.setStyleSheet(
                "font-size: 48px; font-weight: 800; "
                "letter-spacing: -2px; color: #cdd6f4;"
            )
            root.addWidget(title)
            root.addSpacing(10)

            sub = QtWidgets.QLabel("'Not quite my tempo' — Terrance Fletcher")
            sub.setAlignment(QtCore.Qt.AlignHCenter)
            sub.setStyleSheet("font-size: 14px; color: #6c7086;")
            root.addWidget(sub)
            root.addSpacing(32)

            root.addStretch(2)

            self._start_btn = QtWidgets.QPushButton("Start Drumming")
            self._start_btn.setFixedHeight(48)
            self._start_btn.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
            self._start_btn.setStyleSheet("""
                QPushButton {
                    background: #89b4fa;
                    color: #1e1e2e;
                    font-size: 15px;
                    font-weight: 700;
                    border: none;
                    border-radius: 10px;
                    padding: 0 20px;
                }
                QPushButton:hover  { background: #b4d0ff; }
                QPushButton:pressed { background: #7aa2f7; }
                QPushButton:disabled { background: #313244; color: #6c7086; }
            """)
            self._start_btn.clicked.connect(self._on_start)
            root.addWidget(self._start_btn)
            root.addSpacing(12)

            settings_btn = QtWidgets.QPushButton("⚙   Open Settings")
            settings_btn.setFlat(True)
            settings_btn.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
            settings_btn.setStyleSheet("""
                QPushButton {
                    background: transparent;
                    color: #6c7086;
                    font-size: 12px;
                    border: none;
                    padding: 4px;
                }
                QPushButton:hover { color: #89b4fa; }
            """)
            settings_btn.clicked.connect(self._open_settings)
            root.addWidget(settings_btn, 0, QtCore.Qt.AlignHCenter)

            root.addStretch(1)

            hw_text = f"MediaPipe Hands  ·  Camera {config.CAMERA_INDEX}"
            hw = QtWidgets.QLabel(hw_text)
            hw.setAlignment(QtCore.Qt.AlignHCenter)
            hw.setStyleSheet("font-size: 11px; color: #313244;")
            root.addWidget(hw)

        def _on_start(self):
            self._start_btn.setEnabled(False)
            self._start_btn.setText("Loading…")
            self.hide()
            self._loader = LoadingScreen(self._settings_data, args)
            self._loader.show()

        def _open_settings(self):
            from .settings import launch_settings_panel  # type: ignore

            def _on_apply(data: Dict):
                self._settings_data.update(data)

            launch_settings_panel(on_apply=_on_apply, initial=self._settings_data)

    app_qt = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    win = WelcomeWindow()
    win.show()
    app_qt.exec_()