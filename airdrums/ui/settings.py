"""
airdrums.ui.settings
====================
PyQt5 settings panel for AirDrums V2 with a full dark theme.

Entry point: :func:`launch_settings_panel`.

The window is a modal :class:`SettingsWindow` (``QDialog``) organised into
seven tabs:

1. Audio        — sample pack selector + per-drum volume trims
2. MIDI         — port name, per-drum notes, quantisation
3. Drum Lines   — interactive canvas + per-line detail fields
4. Drumstick    — stick length, handedness, velocity curve
5. Vel & Debounce — per-band cooldown sliders
6. Camera       — camera index, depth model, hardware profile
7. Profiles     — save / load JSON profiles
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

from .. import config

log = logging.getLogger(__name__)

# Held at module level so the QApplication is never garbage-collected between
# calls to launch_settings_panel, which would crash the Qt event loop.
_qt_app = None

# ---------------------------------------------------------------------------
# Dark-theme colour constants
# ---------------------------------------------------------------------------
BG = "#1A1A2E"
SURFACE = "#16213E"
ACCENT = "#E94560"
TEXT = "#EAEAEA"
BORDER = "#2A2A4E"


def launch_settings_panel(
    config_state: dict,
    drum_lines: List[config.DrumLine],
) -> Optional[dict]:
    """Open the modal settings window and return the result.

    Blocks until the user clicks *Save* or *Cancel* (or closes the window).

    Args:
        config_state: Current application configuration dictionary.  The
            window pre-populates all widgets from this mapping.
        drum_lines: List of :class:`config.DrumLine` objects representing the
            current drum layout.

    Returns:
        Updated configuration dictionary if the user pressed *Save*, or
        ``None`` if the user pressed *Cancel* or closed the window.
    """
    global _qt_app
    try:
        from PyQt5 import QtWidgets  # type: ignore
    except Exception as exc:  # noqa: BLE001
        log.warning("PyQt5 unavailable (%s); settings panel disabled", exc)
        return None

    # Keep the QApplication alive at module level — if the local reference were
    # the only one, Python would GC it when the function returns, which destroys
    # the Qt event loop and can crash the OpenCV window.
    if _qt_app is None:
        _qt_app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    win = SettingsWindow(config_state, drum_lines)
    accepted = win.exec_()   # blocks until Save or Cancel
    if accepted:
        return win.get_result()
    return None


# ---------------------------------------------------------------------------
# DrumLineCanvas
# ---------------------------------------------------------------------------

class DrumLineCanvas:
    """Qt widget that draws drum lines as horizontal bars on a dark canvas.

    Lines are clickable and draggable to reposition them.  After a move the
    ``line_moved`` signal is emitted with the line name and new normalised
    (x_center, y_position).

    This class is defined at module level so that ``launch_settings_panel``
    can be imported without a running Qt application — the ``QWidget`` base
    class is only resolved when an instance is created.
    """

    # We defer the actual class construction until first instantiation so that
    # importing this module never unconditionally triggers a PyQt5 import.
    _qt_class = None

    @classmethod
    def _build_qt_class(cls):
        """Build and cache the underlying QWidget subclass."""
        if cls._qt_class is not None:
            return cls._qt_class

        from PyQt5 import QtCore, QtGui, QtWidgets  # type: ignore

        class _Canvas(QtWidgets.QWidget):
            """Inner QWidget implementation for :class:`DrumLineCanvas`."""

            line_moved = QtCore.pyqtSignal(str, float, float)

            def __init__(self, drum_lines: List[config.DrumLine], parent=None):
                """Initialise the canvas with the given drum lines.

                Args:
                    drum_lines: Lines to display and allow editing.
                    parent: Optional Qt parent widget.
                """
                super().__init__(parent)
                self.setMinimumSize(500, 300)
                self.setCursor(QtCore.Qt.OpenHandCursor)

                # Work on shallow copies so we don't mutate the originals until
                # the user explicitly saves.
                import dataclasses
                self.drum_lines = [dataclasses.replace(dl) for dl in drum_lines]
                self._selected: Optional[str] = None
                self._drag_offset_x: float = 0.0
                self._drag_offset_y: float = 0.0

            def paintEvent(self, _event):  # noqa: N802
                """Render all drum lines with labels on a dark background."""
                p = QtGui.QPainter(self)
                p.setRenderHint(QtGui.QPainter.Antialiasing)
                p.fillRect(self.rect(), QtGui.QColor(BG))

                cw, ch = self.width(), self.height()
                for dl in self.drum_lines:
                    px = int(dl.x_center * cw)
                    py = int(dl.y_position * ch)
                    half_px = int(dl.half_width * cw)
                    b, g, r = dl.color_bgr
                    color = QtGui.QColor(r, g, b)
                    sel_color = QtGui.QColor(
                        min(255, r + 80), min(255, g + 80), min(255, b + 80)
                    )
                    is_selected = dl.name == self._selected

                    # Horizontal line
                    pen = QtGui.QPen(sel_color if is_selected else color)
                    pen.setWidth(3 if is_selected else 2)
                    p.setPen(pen)
                    p.drawLine(px - half_px, py, px + half_px, py)

                    # Endpoint ticks
                    tick_h = 6
                    p.drawLine(px - half_px, py - tick_h, px - half_px, py + tick_h)
                    p.drawLine(px + half_px, py - tick_h, px + half_px, py + tick_h)

                    # Center handle
                    p.setBrush(QtGui.QBrush(sel_color if is_selected else color))
                    p.setPen(QtCore.Qt.NoPen)
                    p.drawEllipse(QtCore.QPoint(px, py), 6, 6)

                    # Label
                    p.setPen(QtGui.QPen(QtGui.QColor(TEXT)))
                    p.setFont(QtGui.QFont("Monospace", 9))
                    p.drawText(px - half_px + 4, py - 6, dl.label)

            def _hit_test(self, x: int, y: int) -> Optional[str]:
                """Return the name of the drum line nearest the click, or None."""
                cw, ch = self.width(), self.height()
                for dl in self.drum_lines:
                    px = int(dl.x_center * cw)
                    py = int(dl.y_position * ch)
                    half_px = int(dl.half_width * cw)
                    if (px - half_px - 8 <= x <= px + half_px + 8
                            and abs(y - py) <= 10):
                        return dl.name
                return None

            def mousePressEvent(self, event):  # noqa: N802
                """Start a drag operation on the clicked drum line."""
                cw, ch = self.width(), self.height()
                name = self._hit_test(event.x(), event.y())
                self._selected = name
                if name is not None:
                    dl = next(d for d in self.drum_lines if d.name == name)
                    self._drag_offset_x = event.x() / cw - dl.x_center
                    self._drag_offset_y = event.y() / ch - dl.y_position
                    self.setCursor(QtCore.Qt.ClosedHandCursor)
                self.update()

            def mouseMoveEvent(self, event):  # noqa: N802
                """Update the selected line position during drag."""
                if self._selected is None:
                    return
                cw, ch = self.width(), self.height()
                dl = next(
                    (d for d in self.drum_lines if d.name == self._selected), None
                )
                if dl is None:
                    return
                new_x = max(dl.half_width, min(1.0 - dl.half_width,
                    event.x() / cw - self._drag_offset_x))
                new_y = max(0.0, min(1.0, event.y() / ch - self._drag_offset_y))
                dl.x_center = new_x
                dl.y_position = new_y
                self.line_moved.emit(dl.name, new_x, new_y)
                self.update()

            def mouseReleaseEvent(self, _event):  # noqa: N802
                """End the drag operation."""
                self.setCursor(QtCore.Qt.OpenHandCursor)

        cls._qt_class = _Canvas
        return _Canvas

    def __new__(cls, drum_lines: List[config.DrumLine], parent=None):
        """Construct and return the underlying QWidget instance directly."""
        qt_cls = cls._build_qt_class()
        return qt_cls(drum_lines, parent)


# ---------------------------------------------------------------------------
# SettingsWindow
# ---------------------------------------------------------------------------

class SettingsWindow:
    """Modal settings dialog for AirDrums V2.

    Tabs: Audio | MIDI | Drum Lines | Drumstick | Vel & Debounce | Camera | Profiles.

    This class wraps a PyQt5 ``QDialog`` via a deferred inner class so that the
    module can be imported without a running Qt application.
    """

    _qt_class = None

    @classmethod
    def _build_qt_class(cls):
        """Build and cache the QDialog subclass."""
        if cls._qt_class is not None:
            return cls._qt_class

        from PyQt5 import QtCore, QtGui, QtWidgets  # type: ignore

        class _Dialog(QtWidgets.QDialog):
            """Inner QDialog implementation of the settings window."""

            def __init__(
                self,
                config_state: dict,
                drum_lines: List[config.DrumLine],
                parent=None,
            ):
                """Initialise the settings dialog.

                Args:
                    config_state: Current application config dict for pre-population.
                    drum_lines: Current drum line layout.
                    parent: Optional Qt parent.
                """
                super().__init__(parent)
                self._config_state = dict(config_state)
                self._drum_lines = drum_lines
                self._result: Optional[dict] = None

                self.setWindowTitle("AirDrums V2 — Settings")
                self.setMinimumSize(780, 640)
                self.apply_dark_theme()

                root = QtWidgets.QVBoxLayout(self)
                root.setContentsMargins(8, 8, 8, 8)

                tabs = QtWidgets.QTabWidget()
                root.addWidget(tabs, 1)

                tabs.addTab(self._build_audio_tab(), "Audio")
                tabs.addTab(self._build_midi_tab(), "MIDI")
                tabs.addTab(self._build_drumstick_tab(), "Drumstick")
                tabs.addTab(self._build_velocity_tab(), "Vel & Debounce")
                tabs.addTab(self._build_camera_tab(), "Camera")
                tabs.addTab(self._build_profiles_tab(), "Profiles")

                # Bottom buttons
                btns = QtWidgets.QDialogButtonBox(
                    QtWidgets.QDialogButtonBox.Save | QtWidgets.QDialogButtonBox.Cancel
                )
                btns.accepted.connect(self._on_save)
                btns.rejected.connect(self.reject)
                root.addWidget(btns)

            # ------------------------------------------------------------------
            # Tab builders
            # ------------------------------------------------------------------

            def _build_audio_tab(self) -> QtWidgets.QWidget:
                """Build the Audio tab: sample pack selector and per-drum volume trims."""
                from PyQt5 import QtWidgets  # noqa: F811

                w = QtWidgets.QWidget()
                layout = QtWidgets.QVBoxLayout(w)
                form = QtWidgets.QFormLayout()

                # Sample pack selector + Preview button
                pack_row = QtWidgets.QHBoxLayout()
                self.w_sample_pack = QtWidgets.QComboBox()
                packs = []
                if config.PACKS_DIR.exists():
                    packs = [p.name for p in config.PACKS_DIR.iterdir() if p.is_dir()]
                if not packs:
                    packs = ["default"]
                self.w_sample_pack.addItems(packs)
                cur_pack = self._config_state.get("sample_pack", packs[0])
                self.w_sample_pack.setCurrentText(cur_pack)
                pack_row.addWidget(self.w_sample_pack, 1)
                preview_btn = QtWidgets.QPushButton("Preview")
                preview_btn.clicked.connect(self._preview_pack)
                pack_row.addWidget(preview_btn)
                pack_widget = QtWidgets.QWidget()
                pack_widget.setLayout(pack_row)
                form.addRow("Sample pack", pack_widget)

                layout.addLayout(form)

                # Per-drum volume trims
                layout.addWidget(QtWidgets.QLabel("Per-drum volume (0–200%, 100% = unity):"))
                scroll = QtWidgets.QScrollArea()
                scroll.setWidgetResizable(True)
                trim_widget = QtWidgets.QWidget()
                trim_form = QtWidgets.QFormLayout(trim_widget)
                self.w_volume_trims: Dict[str, QtWidgets.QSlider] = {}
                vol_trims = self._config_state.get("volume_trims", {})
                for dl in self._drum_lines:
                    slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
                    slider.setRange(0, 200)
                    slider.setValue(int(vol_trims.get(dl.name, 100)))
                    val_label = QtWidgets.QLabel(f"{slider.value()}%")
                    slider.valueChanged.connect(
                        lambda v, lbl=val_label: lbl.setText(f"{v}%")
                    )
                    row_widget = QtWidgets.QWidget()
                    row_layout = QtWidgets.QHBoxLayout(row_widget)
                    row_layout.addWidget(slider, 1)
                    row_layout.addWidget(val_label)
                    self.w_volume_trims[dl.name] = slider
                    trim_form.addRow(dl.label, row_widget)
                scroll.setWidget(trim_widget)
                layout.addWidget(scroll, 1)
                return w

            def _build_midi_tab(self) -> QtWidgets.QWidget:
                """Build the MIDI tab: port name, per-drum notes, quantisation."""
                w = QtWidgets.QWidget()
                layout = QtWidgets.QVBoxLayout(w)
                form = QtWidgets.QFormLayout()

                self.w_midi_port = QtWidgets.QLineEdit(
                    self._config_state.get("midi_port", config.MIDI_PORT_NAME)
                )
                form.addRow("Port name", self.w_midi_port)

                self.w_quantization = QtWidgets.QComboBox()
                quant_opts = ["none", "1/8", "1/16", "1/32"]
                self.w_quantization.addItems(quant_opts)
                self.w_quantization.setCurrentText(
                    self._config_state.get("quantization", config.DEFAULT_QUANTIZATION)
                )
                form.addRow("Quantization", self.w_quantization)
                layout.addLayout(form)

                layout.addWidget(QtWidgets.QLabel("Per-drum MIDI note (0–127):"))
                scroll = QtWidgets.QScrollArea()
                scroll.setWidgetResizable(True)
                note_widget = QtWidgets.QWidget()
                note_form = QtWidgets.QFormLayout(note_widget)
                self.w_midi_notes: Dict[str, QtWidgets.QSpinBox] = {}
                custom_notes = self._config_state.get("midi_notes", {})
                for dl in self._drum_lines:
                    spin = QtWidgets.QSpinBox()
                    spin.setRange(0, 127)
                    spin.setValue(int(custom_notes.get(dl.name, dl.midi_note)))
                    self.w_midi_notes[dl.name] = spin
                    note_form.addRow(dl.label, spin)
                scroll.setWidget(note_widget)
                layout.addWidget(scroll, 1)
                return w

            def _build_drum_lines_tab(self) -> QtWidgets.QWidget:
                """Build the Drum Lines tab: interactive canvas + detail editor."""
                w = QtWidgets.QWidget()
                layout = QtWidgets.QVBoxLayout(w)

                import dataclasses
                self.w_canvas = DrumLineCanvas(self._drum_lines)
                self.w_canvas.line_moved.connect(self._on_line_moved)
                layout.addWidget(self.w_canvas, 1)

                # Detail editor for selected line
                detail_group = QtWidgets.QGroupBox("Selected line properties")
                detail_form = QtWidgets.QFormLayout(detail_group)

                self.w_dl_x_center = QtWidgets.QDoubleSpinBox()
                self.w_dl_x_center.setRange(0.0, 1.0)
                self.w_dl_x_center.setSingleStep(0.01)
                self.w_dl_x_center.setDecimals(3)
                detail_form.addRow("x_center", self.w_dl_x_center)

                self.w_dl_y_position = QtWidgets.QDoubleSpinBox()
                self.w_dl_y_position.setRange(0.0, 1.0)
                self.w_dl_y_position.setSingleStep(0.01)
                self.w_dl_y_position.setDecimals(3)
                detail_form.addRow("y_position", self.w_dl_y_position)

                self.w_dl_half_width = QtWidgets.QDoubleSpinBox()
                self.w_dl_half_width.setRange(0.01, 0.49)
                self.w_dl_half_width.setSingleStep(0.01)
                self.w_dl_half_width.setDecimals(3)
                detail_form.addRow("half_width", self.w_dl_half_width)

                self.w_dl_label = QtWidgets.QLineEdit()
                detail_form.addRow("Label", self.w_dl_label)

                self.w_dl_midi_note = QtWidgets.QSpinBox()
                self.w_dl_midi_note.setRange(0, 127)
                detail_form.addRow("MIDI note", self.w_dl_midi_note)

                color_row = QtWidgets.QHBoxLayout()
                self.w_dl_color_btn = QtWidgets.QPushButton("Pick colour…")
                self._dl_color_bgr = (128, 128, 128)
                self.w_dl_color_btn.clicked.connect(self._pick_line_color)
                color_row.addWidget(self.w_dl_color_btn)
                color_wrap = QtWidgets.QWidget()
                color_wrap.setLayout(color_row)
                detail_form.addRow("Colour", color_wrap)

                self.w_dl_cymbal = QtWidgets.QCheckBox("Is cymbal")
                detail_form.addRow("", self.w_dl_cymbal)

                apply_btn = QtWidgets.QPushButton("Apply to selected")
                apply_btn.clicked.connect(self._apply_line_detail)
                detail_form.addRow("", apply_btn)

                layout.addWidget(detail_group)

                # Populate detail from first line
                if self._drum_lines:
                    self._populate_detail(self._drum_lines[0])

                return w

            def _build_drumstick_tab(self) -> QtWidgets.QWidget:
                """Build the Drumstick tab: length, handedness, velocity curve."""
                w = QtWidgets.QWidget()
                form = QtWidgets.QFormLayout(w)

                # Stick length slider: maps int 10-30 to float 0.10-0.30
                stick_row = QtWidgets.QHBoxLayout()
                self.w_stick_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
                self.w_stick_slider.setRange(10, 30)
                cur_len = self._config_state.get("stick_length", config.STICK_LENGTH_DEFAULT)
                self.w_stick_slider.setValue(int(cur_len * 100))
                self.w_stick_label = QtWidgets.QLabel(f"{cur_len:.2f}")
                self.w_stick_slider.valueChanged.connect(
                    lambda v: self.w_stick_label.setText(f"{v / 100:.2f}")
                )
                stick_row.addWidget(self.w_stick_slider, 1)
                stick_row.addWidget(self.w_stick_label)
                stick_widget = QtWidgets.QWidget()
                stick_widget.setLayout(stick_row)
                form.addRow("Stick length", stick_widget)

                # Handedness
                hand_row = QtWidgets.QHBoxLayout()
                self.w_hand_right = QtWidgets.QRadioButton("Right")
                self.w_hand_left = QtWidgets.QRadioButton("Left")
                mirrored = bool(self._config_state.get("mirror_mode", False))
                if mirrored:
                    self.w_hand_left.setChecked(True)
                else:
                    self.w_hand_right.setChecked(True)
                hand_row.addWidget(self.w_hand_right)
                hand_row.addWidget(self.w_hand_left)
                hand_widget = QtWidgets.QWidget()
                hand_widget.setLayout(hand_row)
                form.addRow("Handedness", hand_widget)

                # Velocity curve
                self.w_vel_curve = QtWidgets.QComboBox()
                self.w_vel_curve.addItems(["linear", "logarithmic", "exponential"])
                self.w_vel_curve.setCurrentText(
                    self._config_state.get("velocity_curve", "linear")
                )
                form.addRow("Velocity curve", self.w_vel_curve)

                return w

            def _build_velocity_tab(self) -> QtWidgets.QWidget:
                """Build the Velocity & Debounce tab: per-band cooldown sliders."""
                w = QtWidgets.QWidget()
                layout = QtWidgets.QVBoxLayout(w)
                layout.addWidget(
                    QtWidgets.QLabel("Adjust per-band cooldown (20–500 ms):")
                )

                scroll = QtWidgets.QScrollArea()
                scroll.setWidgetResizable(True)
                inner = QtWidgets.QWidget()
                inner_layout = QtWidgets.QVBoxLayout(inner)

                self.w_cooldowns: Dict[str, QtWidgets.QSlider] = {}
                saved_cooldowns = self._config_state.get("cooldowns", {})

                for band in config.VELOCITY_BANDS:
                    group = QtWidgets.QGroupBox(
                        f"{band.name.capitalize()}  "
                        f"(speed {band.speed_min:.2f}–{band.speed_max:.2f})"
                    )
                    group_layout = QtWidgets.QFormLayout(group)

                    speed_lbl = QtWidgets.QLabel(
                        f"{band.speed_min:.2f} – {band.speed_max:.2f}"
                    )
                    group_layout.addRow("Speed range", speed_lbl)

                    cd_row = QtWidgets.QHBoxLayout()
                    cd_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
                    cd_slider.setRange(20, 500)
                    cd_slider.setValue(
                        int(saved_cooldowns.get(band.name, band.cooldown_ms))
                    )
                    cd_val_lbl = QtWidgets.QLabel(f"{cd_slider.value()} ms")
                    cd_slider.valueChanged.connect(
                        lambda v, lbl=cd_val_lbl: lbl.setText(f"{v} ms")
                    )
                    cd_row.addWidget(cd_slider, 1)
                    cd_row.addWidget(cd_val_lbl)
                    cd_widget = QtWidgets.QWidget()
                    cd_widget.setLayout(cd_row)
                    group_layout.addRow("Cooldown", cd_widget)

                    self.w_cooldowns[band.name] = cd_slider
                    inner_layout.addWidget(group)

                inner_layout.addStretch(1)
                scroll.setWidget(inner)
                layout.addWidget(scroll, 1)
                return w

            def _build_camera_tab(self) -> QtWidgets.QWidget:
                """Build the Camera tab: index, depth model, hardware profile."""
                w = QtWidgets.QWidget()
                form = QtWidgets.QFormLayout(w)

                self.w_camera_index = QtWidgets.QSpinBox()
                self.w_camera_index.setRange(0, 9)
                self.w_camera_index.setValue(
                    int(self._config_state.get("camera_index", config.CAMERA_INDEX))
                )
                form.addRow("Camera index", self.w_camera_index)

                self.w_depth_model = QtWidgets.QComboBox()
                self.w_depth_model.addItems(["Small", "Base"])
                self.w_depth_model.setCurrentText(
                    self._config_state.get("depth_model", "Small")
                )
                form.addRow("Depth model", self.w_depth_model)

                self.w_hw_profile = QtWidgets.QComboBox()
                self.w_hw_profile.addItems(["auto", "low", "medium", "high"])
                self.w_hw_profile.setCurrentText(
                    self._config_state.get("hw_profile", "auto")
                )
                form.addRow("Hardware profile", self.w_hw_profile)

                return w

            def _build_profiles_tab(self) -> QtWidgets.QWidget:
                """Build the Profiles tab: save and load named JSON profiles."""
                w = QtWidgets.QWidget()
                layout = QtWidgets.QVBoxLayout(w)

                form = QtWidgets.QFormLayout()
                self.w_profile_name = QtWidgets.QLineEdit(
                    self._config_state.get("profile_name", "my_profile")
                )
                form.addRow("Profile name", self.w_profile_name)
                layout.addLayout(form)

                save_btn = QtWidgets.QPushButton("Save profile")
                save_btn.clicked.connect(self._save_profile)
                layout.addWidget(save_btn)

                layout.addWidget(QtWidgets.QLabel("Load profile:"))
                load_row = QtWidgets.QHBoxLayout()
                self.w_profile_combo = QtWidgets.QComboBox()
                self._refresh_profile_list()
                load_btn = QtWidgets.QPushButton("Load")
                load_btn.clicked.connect(self._load_profile)
                load_row.addWidget(self.w_profile_combo, 1)
                load_row.addWidget(load_btn)
                load_widget = QtWidgets.QWidget()
                load_widget.setLayout(load_row)
                layout.addWidget(load_widget)
                layout.addStretch(1)
                return w

            # ------------------------------------------------------------------
            # Helper callbacks
            # ------------------------------------------------------------------

            def _preview_pack(self) -> None:
                """Attempt to play the first sample from the selected pack."""
                pack_name = self.w_sample_pack.currentText()
                pack_path = config.PACKS_DIR / pack_name
                log.info("Preview requested for pack: %s", pack_path)
                wav_files = list(pack_path.glob("*.wav")) if pack_path.exists() else []
                if not wav_files:
                    log.warning("No WAV files found in pack: %s", pack_path)
                    return
                try:
                    import subprocess  # noqa: S404
                    subprocess.Popen(  # noqa: S603
                        ["python", "-c",
                         f"import playsound; playsound.playsound(r'{wav_files[0]}')"],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                except Exception as exc:  # noqa: BLE001
                    log.warning("Preview failed: %s", exc)

            def _on_line_moved(self, name: str, x: float, y: float) -> None:
                """Update detail fields when a line is repositioned on the canvas."""
                if hasattr(self, "w_canvas") and self.w_canvas._selected == name:
                    dl = next(
                        (d for d in self.w_canvas.drum_lines if d.name == name), None
                    )
                    if dl is not None:
                        self._populate_detail(dl)

            def _populate_detail(self, dl: config.DrumLine) -> None:
                """Fill the detail-editor widgets from the given drum line.

                Args:
                    dl: The :class:`config.DrumLine` to display.
                """
                self.w_dl_x_center.setValue(dl.x_center)
                self.w_dl_y_position.setValue(dl.y_position)
                self.w_dl_half_width.setValue(dl.half_width)
                self.w_dl_label.setText(dl.label)
                self.w_dl_midi_note.setValue(dl.midi_note)
                self.w_dl_cymbal.setChecked(dl.is_cymbal)
                self._dl_color_bgr = dl.color_bgr
                b, g, r = dl.color_bgr
                self.w_dl_color_btn.setStyleSheet(
                    f"background-color: rgb({r},{g},{b});"
                )

            def _apply_line_detail(self) -> None:
                """Write the detail-editor values back to the selected canvas line."""
                selected = getattr(self.w_canvas, "_selected", None)
                if selected is None:
                    return
                dl = next(
                    (d for d in self.w_canvas.drum_lines if d.name == selected), None
                )
                if dl is None:
                    return
                dl.x_center = self.w_dl_x_center.value()
                dl.y_position = self.w_dl_y_position.value()
                dl.half_width = self.w_dl_half_width.value()
                dl.label = self.w_dl_label.text()
                dl.midi_note = self.w_dl_midi_note.value()
                dl.is_cymbal = self.w_dl_cymbal.isChecked()
                dl.color_bgr = self._dl_color_bgr
                self.w_canvas.update()
                log.debug("Applied detail edits to line: %s", selected)

            def _pick_line_color(self) -> None:
                """Open a QColorDialog and store the chosen colour."""
                from PyQt5 import QtGui, QtWidgets  # noqa: F811

                b, g, r = self._dl_color_bgr
                initial = QtGui.QColor(r, g, b)
                color = QtWidgets.QColorDialog.getColor(initial, self, "Pick line colour")
                if color.isValid():
                    self._dl_color_bgr = (color.blue(), color.green(), color.red())
                    self.w_dl_color_btn.setStyleSheet(
                        f"background-color: rgb({color.red()},{color.green()},{color.blue()});"
                    )

            def _refresh_profile_list(self) -> None:
                """Populate the load-profile combo box from PROFILES_DIR."""
                self.w_profile_combo.clear()
                profiles = []
                if config.PROFILES_DIR.exists():
                    profiles = [
                        p.stem
                        for p in config.PROFILES_DIR.glob("*.json")
                    ]
                if not profiles:
                    profiles = ["(no profiles)"]
                self.w_profile_combo.addItems(profiles)

            def _save_profile(self) -> None:
                """Write current widget values to a JSON profile file."""
                name = self.w_profile_name.text().strip() or "profile"
                path = config.PROFILES_DIR / f"{name}.json"
                data = self.get_result()
                data["profile_name"] = name
                path.write_text(json.dumps(data, indent=2))
                log.info("Profile saved: %s", path)
                self._refresh_profile_list()

            def _load_profile(self) -> None:
                """Load a JSON profile and populate all widgets."""
                name = self.w_profile_combo.currentText()
                path = config.PROFILES_DIR / f"{name}.json"
                if not path.exists():
                    log.warning("Profile not found: %s", path)
                    return
                try:
                    data = json.loads(path.read_text())
                except Exception as exc:  # noqa: BLE001
                    log.error("Failed to load profile %s: %s", path, exc)
                    return
                self._populate_from_dict(data)
                log.info("Profile loaded: %s", path)

            def _populate_from_dict(self, data: dict) -> None:
                """Apply a config dictionary to all open widgets.

                Args:
                    data: Config dict (same format as :meth:`get_result`).
                """
                if "sample_pack" in data:
                    self.w_sample_pack.setCurrentText(data["sample_pack"])
                if "midi_port" in data:
                    self.w_midi_port.setText(data["midi_port"])
                if "quantization" in data:
                    self.w_quantization.setCurrentText(data["quantization"])
                if "camera_index" in data:
                    self.w_camera_index.setValue(int(data["camera_index"]))
                if "depth_model" in data:
                    self.w_depth_model.setCurrentText(data["depth_model"])
                if "hw_profile" in data:
                    self.w_hw_profile.setCurrentText(data["hw_profile"])
                if "stick_length" in data:
                    self.w_stick_slider.setValue(int(float(data["stick_length"]) * 100))
                if "mirror_mode" in data:
                    if data["mirror_mode"]:
                        self.w_hand_left.setChecked(True)
                    else:
                        self.w_hand_right.setChecked(True)
                if "velocity_curve" in data:
                    self.w_vel_curve.setCurrentText(data["velocity_curve"])
                vol_trims = data.get("volume_trims", {})
                for name, slider in self.w_volume_trims.items():
                    if name in vol_trims:
                        slider.setValue(int(vol_trims[name]))
                midi_notes = data.get("midi_notes", {})
                for name, spin in self.w_midi_notes.items():
                    if name in midi_notes:
                        spin.setValue(int(midi_notes[name]))
                cooldowns = data.get("cooldowns", {})
                for name, slider in self.w_cooldowns.items():
                    if name in cooldowns:
                        slider.setValue(int(cooldowns[name]))

            # ------------------------------------------------------------------
            # Save / result
            # ------------------------------------------------------------------

            def _on_save(self) -> None:
                """Collect widget values, store them and close the dialog."""
                self._result = self.get_result()
                self.accept()

            def get_result(self) -> dict:
                """Collect all widget values into a configuration dictionary.

                Returns:
                    Dictionary containing all settings as plain Python types.
                    Returns an empty dict if called before the dialog is shown.
                """
                return {
                    "sample_pack": self.w_sample_pack.currentText(),
                    "volume_trims": {
                        name: slider.value()
                        for name, slider in self.w_volume_trims.items()
                    },
                    "midi_port": self.w_midi_port.text(),
                    "quantization": self.w_quantization.currentText(),
                    "midi_notes": {
                        name: spin.value()
                        for name, spin in self.w_midi_notes.items()
                    },
                    "stick_length": self.w_stick_slider.value() / 100,
                    "mirror_mode": self.w_hand_left.isChecked(),
                    "velocity_curve": self.w_vel_curve.currentText(),
                    "cooldowns": {
                        name: slider.value()
                        for name, slider in self.w_cooldowns.items()
                    },
                    "camera_index": self.w_camera_index.value(),
                    "depth_model": self.w_depth_model.currentText(),
                    "hw_profile": self.w_hw_profile.currentText(),
                    "profile_name": self.w_profile_name.text().strip(),
                }

            # ------------------------------------------------------------------
            # Theming
            # ------------------------------------------------------------------

            def apply_dark_theme(self) -> None:
                """Apply the AirDrums dark QSS stylesheet to this dialog."""
                self.setStyleSheet(
                    f"""
                    QDialog, QWidget {{
                        background-color: {BG};
                        color: {TEXT};
                        font-family: "Segoe UI", "Helvetica Neue", sans-serif;
                        font-size: 13px;
                    }}
                    QTabWidget::pane {{
                        border: 1px solid {BORDER};
                        background-color: {SURFACE};
                    }}
                    QTabBar::tab {{
                        background-color: {BG};
                        color: {TEXT};
                        padding: 6px 16px;
                        border: 1px solid {BORDER};
                        border-bottom: none;
                    }}
                    QTabBar::tab:selected {{
                        background-color: {SURFACE};
                        color: {ACCENT};
                        border-bottom: 2px solid {ACCENT};
                    }}
                    QGroupBox {{
                        border: 1px solid {BORDER};
                        margin-top: 8px;
                        padding-top: 10px;
                        color: {TEXT};
                    }}
                    QGroupBox::title {{
                        subcontrol-origin: margin;
                        left: 8px;
                        padding: 0 4px;
                        color: {ACCENT};
                    }}
                    QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {{
                        background-color: {SURFACE};
                        color: {TEXT};
                        border: 1px solid {BORDER};
                        border-radius: 3px;
                        padding: 3px 6px;
                    }}
                    QComboBox::drop-down {{
                        border: none;
                    }}
                    QSlider::groove:horizontal {{
                        background: {BORDER};
                        height: 4px;
                        border-radius: 2px;
                    }}
                    QSlider::handle:horizontal {{
                        background: {ACCENT};
                        width: 14px;
                        height: 14px;
                        margin: -5px 0;
                        border-radius: 7px;
                    }}
                    QSlider::sub-page:horizontal {{
                        background: {ACCENT};
                        border-radius: 2px;
                    }}
                    QPushButton {{
                        background-color: {ACCENT};
                        color: {TEXT};
                        border: none;
                        padding: 5px 14px;
                        border-radius: 4px;
                    }}
                    QPushButton:hover {{
                        background-color: #ff6080;
                    }}
                    QPushButton:pressed {{
                        background-color: #c03050;
                    }}
                    QDialogButtonBox QPushButton {{
                        min-width: 80px;
                    }}
                    QScrollArea {{
                        border: none;
                    }}
                    QLabel {{
                        color: {TEXT};
                    }}
                    QCheckBox {{
                        color: {TEXT};
                    }}
                    QRadioButton {{
                        color: {TEXT};
                    }}
                    """
                )

        cls._qt_class = _Dialog
        return _Dialog

    def __new__(
        cls,
        config_state: dict,
        drum_lines: List[config.DrumLine],
        parent=None,
    ):
        """Construct and return the underlying QDialog instance directly.

        Args:
            config_state: Current application config dict.
            drum_lines: Current drum line layout.
            parent: Optional Qt parent widget.

        Returns:
            A fully initialised QDialog instance.
        """
        qt_cls = cls._build_qt_class()
        return qt_cls(config_state, drum_lines, parent)
