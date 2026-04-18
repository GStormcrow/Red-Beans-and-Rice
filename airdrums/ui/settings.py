"""
airdrums.ui.settings
====================
PyQt5 settings panel. Controls:
  * Sample pack selector (with preview)
  * MIDI port selector
  * Camera index selector
  * Depth model size (Small / Base)
  * Drum zone layout editor (drag-to-reposition on 2D canvas)
  * Velocity curve editor (linear / logarithmic / exponential)
  * Per-drum cooldown sliders
  * Drumstick length slider (0.10 - 0.30)
  * HUD theme selector (dark / light / neon)
  * Hardware profile override
  * Save / load JSON profiles
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Callable, Dict, List, Optional

from .. import config

log = logging.getLogger(__name__)


def launch_settings_panel(on_apply: Optional[Callable[[Dict], None]] = None,
                          initial: Optional[Dict] = None) -> None:
    """Open the settings panel. Blocks until the window is closed."""
    try:
        from PyQt5 import QtCore, QtGui, QtWidgets   # type: ignore
    except Exception as exc:  # noqa: BLE001
        log.warning("PyQt5 unavailable (%s); settings panel disabled", exc)
        return

    initial = initial or {}

    class ZoneCanvas(QtWidgets.QWidget):
        """Click + drag rectangles to reposition zones in normalized coords."""

        def __init__(self, zones: List[config.DrumZone]):
            super().__init__()
            self.setMinimumSize(480, 270)
            self.zones = [config.DrumZone(**{
                "name": z.name, "x_range": z.x_range, "y_range": z.y_range,
                "z_range": z.z_range, "midi_note": z.midi_note,
                "color_bgr": z.color_bgr, "is_pedal": z.is_pedal,
                "stem_group": z.stem_group,
            }) for z in zones]
            self._drag_idx: Optional[int] = None
            self._drag_dx = 0.0
            self._drag_dy = 0.0

        def paintEvent(self, _event):   # noqa: N802
            p = QtGui.QPainter(self)
            p.fillRect(self.rect(), QtGui.QColor(20, 20, 26))
            w, h = self.width(), self.height()
            for z in self.zones:
                r = QtCore.QRect(int(z.x_range[0] * w), int(z.y_range[0] * h),
                                 int((z.x_range[1] - z.x_range[0]) * w),
                                 int((z.y_range[1] - z.y_range[0]) * h))
                b, g, rr = z.color_bgr
                p.setBrush(QtGui.QColor(rr, g, b, 90))
                p.setPen(QtGui.QColor(rr, g, b))
                p.drawRect(r)
                p.setPen(QtGui.QColor(230, 230, 230))
                p.drawText(r.x() + 4, r.y() + 14, z.name)

        def mousePressEvent(self, e):   # noqa: N802
            nx, ny = e.x() / self.width(), e.y() / self.height()
            for i, z in enumerate(self.zones):
                if z.x_range[0] <= nx <= z.x_range[1] and z.y_range[0] <= ny <= z.y_range[1]:
                    self._drag_idx = i
                    self._drag_dx = nx - z.x_range[0]
                    self._drag_dy = ny - z.y_range[0]
                    return

        def mouseMoveEvent(self, e):    # noqa: N802
            if self._drag_idx is None:
                return
            nx, ny = e.x() / self.width(), e.y() / self.height()
            z = self.zones[self._drag_idx]
            w = z.x_range[1] - z.x_range[0]
            h = z.y_range[1] - z.y_range[0]
            x0 = max(0.0, min(1.0 - w, nx - self._drag_dx))
            y0 = max(0.0, min(1.0 - h, ny - self._drag_dy))
            z.x_range = (x0, x0 + w)
            z.y_range = (y0, y0 + h)
            self.update()

        def mouseReleaseEvent(self, _event):   # noqa: N802
            self._drag_idx = None


    class SettingsWindow(QtWidgets.QWidget):
        def __init__(self):
            super().__init__()
            self.setWindowTitle("AirDrums - Settings")
            self.setMinimumSize(720, 620)
            layout = QtWidgets.QVBoxLayout(self)
            form = QtWidgets.QFormLayout()

            self.pack = QtWidgets.QComboBox()
            pack_root = config.SAMPLE_PACKS_DIR
            packs = [p.name for p in pack_root.iterdir()] if pack_root.exists() else []
            if not packs:
                packs = ["default"]
            self.pack.addItems(packs)
            self.pack.setCurrentText(initial.get("sample_pack", packs[0]))
            form.addRow("Sample pack", self.pack)

            self.midi_port = QtWidgets.QLineEdit(initial.get("midi_port", config.MIDI_PORT_NAME))
            form.addRow("MIDI port name", self.midi_port)

            self.camera = QtWidgets.QSpinBox()
            self.camera.setRange(0, 8)
            self.camera.setValue(initial.get("camera_index", config.CAMERA_INDEX))
            form.addRow("Camera index", self.camera)

            self.depth_model = QtWidgets.QComboBox()
            self.depth_model.addItems(["Small", "Base"])
            self.depth_model.setCurrentText(initial.get("depth_model", "Small"))
            form.addRow("Depth model", self.depth_model)

            self.profile = QtWidgets.QComboBox()
            self.profile.addItems(["auto", "low", "medium", "high"])
            self.profile.setCurrentText(initial.get("profile", "auto"))
            form.addRow("Hardware profile", self.profile)

            self.curve = QtWidgets.QComboBox()
            self.curve.addItems(["linear", "logarithmic", "exponential"])
            self.curve.setCurrentText(initial.get("velocity_curve", "linear"))
            form.addRow("Velocity curve", self.curve)

            self.theme = QtWidgets.QComboBox()
            self.theme.addItems(list(config.HUD_THEMES))
            self.theme.setCurrentText(initial.get("theme", config.DEFAULT_HUD_THEME))
            form.addRow("HUD theme", self.theme)

            # Drumstick length slider 0.10 - 0.30 (shown * 100)
            self.stick = QtWidgets.QSlider(QtCore.Qt.Horizontal)
            self.stick.setRange(int(config.DRUMSTICK_LENGTH_MIN * 100),
                                int(config.DRUMSTICK_LENGTH_MAX * 100))
            self.stick.setValue(int(initial.get("stick_length", config.DRUMSTICK_LENGTH_DEFAULT) * 100))
            self.stick_label = QtWidgets.QLabel(f"{self.stick.value() / 100:.2f}")
            self.stick.valueChanged.connect(
                lambda v: self.stick_label.setText(f"{v / 100:.2f}"))
            stick_row = QtWidgets.QHBoxLayout()
            stick_row.addWidget(self.stick); stick_row.addWidget(self.stick_label)
            stick_wrap = QtWidgets.QWidget(); stick_wrap.setLayout(stick_row)
            form.addRow("Drumstick length", stick_wrap)

            # Per-drum cooldowns
            self.cooldowns: Dict[str, QtWidgets.QSlider] = {}
            cdg = QtWidgets.QGroupBox("Per-drum cooldowns (ms)")
            cdl = QtWidgets.QFormLayout(cdg)
            defaults = initial.get("cooldowns", {})
            for z in config.DRUM_ZONES:
                s = QtWidgets.QSlider(QtCore.Qt.Horizontal)
                s.setRange(40, 400)
                s.setValue(int(defaults.get(z.name, config.STRIKE_COOLDOWN_MS)))
                self.cooldowns[z.name] = s
                cdl.addRow(z.name, s)
            layout.addLayout(form)
            layout.addWidget(cdg)

            # Zone editor
            self.zone_canvas = ZoneCanvas(config.DRUM_ZONES)
            layout.addWidget(QtWidgets.QLabel("Drag zones to reposition:"))
            layout.addWidget(self.zone_canvas, 1)

            # Save / load / apply
            btns = QtWidgets.QHBoxLayout()
            b_save = QtWidgets.QPushButton("Save profile...")
            b_load = QtWidgets.QPushButton("Load profile...")
            b_apply = QtWidgets.QPushButton("Apply")
            b_apply.setDefault(True)
            btns.addWidget(b_save); btns.addWidget(b_load); btns.addStretch(1); btns.addWidget(b_apply)
            layout.addLayout(btns)
            b_save.clicked.connect(self._save)
            b_load.clicked.connect(self._load)
            b_apply.clicked.connect(self._apply)

        # --------------------------------------------------------------
        def _collect(self) -> Dict:
            return {
                "sample_pack": self.pack.currentText(),
                "midi_port": self.midi_port.text(),
                "camera_index": int(self.camera.value()),
                "depth_model": self.depth_model.currentText(),
                "profile": self.profile.currentText(),
                "velocity_curve": self.curve.currentText(),
                "theme": self.theme.currentText(),
                "stick_length": self.stick.value() / 100,
                "cooldowns": {n: int(s.value()) for n, s in self.cooldowns.items()},
                "zones": [
                    {"name": z.name, "x_range": z.x_range, "y_range": z.y_range}
                    for z in self.zone_canvas.zones
                ],
            }

        def _save(self) -> None:
            path, _ = QtWidgets.QFileDialog.getSaveFileName(
                self, "Save profile", str(config.PROFILES_DIR / "profile.json"),
                "JSON (*.json)")
            if not path:
                return
            Path(path).write_text(json.dumps(self._collect(), indent=2))

        def _load(self) -> None:
            path, _ = QtWidgets.QFileDialog.getOpenFileName(
                self, "Load profile", str(config.PROFILES_DIR), "JSON (*.json)")
            if not path:
                return
            data = json.loads(Path(path).read_text())
            if "sample_pack" in data:
                self.pack.setCurrentText(data["sample_pack"])
            if "midi_port" in data:
                self.midi_port.setText(data["midi_port"])
            if "camera_index" in data:
                self.camera.setValue(int(data["camera_index"]))
            if "stick_length" in data:
                self.stick.setValue(int(float(data["stick_length"]) * 100))
            if "theme" in data:
                self.theme.setCurrentText(data["theme"])
            if "profile" in data:
                self.profile.setCurrentText(data["profile"])

        def _apply(self) -> None:
            if on_apply is not None:
                on_apply(self._collect())
            self.close()

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    win = SettingsWindow()
    win.show()
    app.exec_()
