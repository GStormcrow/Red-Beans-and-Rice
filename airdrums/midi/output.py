"""
airdrums.midi.output
====================
Live MIDI output to a virtual port named "AirDrums" via python-rtmidi.
Any DAW that scans MIDI inputs (Ableton, FL, Logic, GarageBand, Reaper)
will see the port automatically.

Also emits MIDI clock (24 PPQN) synced to the detected BPM and Start/
Stop/Continue messages on session record and pause.
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Optional

from .. import config

log = logging.getLogger(__name__)

# Status bytes
NOTE_ON = 0x90
NOTE_OFF = 0x80
MIDI_CLOCK = 0xF8
MIDI_START = 0xFA
MIDI_CONTINUE = 0xFB
MIDI_STOP = 0xFC


class MidiOutput:
    """Virtual MIDI port + clock scheduler."""

    def __init__(self, port_name: str = config.MIDI_PORT_NAME):
        self.port_name = port_name
        self._port = None
        self._connected = False
        self._bpm = config.MIDI_DEFAULT_BPM
        self._clock_stop = threading.Event()
        self._clock_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._open()

    # ------------------------------------------------------------------
    def _open(self) -> None:
        try:
            import rtmidi  # type: ignore
            self._port = rtmidi.MidiOut()
            # Try to open a brand new virtual port (macOS/Linux support this
            # natively; on Windows this falls back to an existing port).
            try:
                self._port.open_virtual_port(self.port_name)
            except (NotImplementedError, AttributeError):
                ports = self._port.get_ports()
                if ports:
                    self._port.open_port(0)
                else:
                    raise RuntimeError("No MIDI output ports available")
            self._connected = True
            log.info("Opened MIDI output: %s", self.port_name)
        except Exception as exc:  # noqa: BLE001
            log.warning("MIDI output unavailable (%s); running silently", exc)
            self._connected = False

    # ------------------------------------------------------------------
    @property
    def connected(self) -> bool:
        return self._connected

    # ------------------------------------------------------------------
    def send_hit(self, midi_note: int, velocity: int) -> None:
        """Send Note On, then Note Off after MIDI_NOTE_OFF_MS."""
        if not self._connected:
            return
        ch = config.MIDI_CHANNEL & 0x0F
        try:
            with self._lock:
                self._port.send_message([NOTE_ON | ch, midi_note & 0x7F, velocity & 0x7F])
        except Exception as exc:  # noqa: BLE001
            log.debug("send_hit failed: %s", exc)
            return
        timer = threading.Timer(
            config.MIDI_NOTE_OFF_MS / 1000.0,
            self._send_note_off, args=(midi_note,)
        )
        timer.daemon = True
        timer.start()

    def _send_note_off(self, midi_note: int) -> None:
        if not self._connected:
            return
        ch = config.MIDI_CHANNEL & 0x0F
        try:
            with self._lock:
                self._port.send_message([NOTE_OFF | ch, midi_note & 0x7F, 0])
        except Exception:  # noqa: BLE001
            pass

    # ------------------------------------------------------------------
    # Transport + clock
    # ------------------------------------------------------------------
    def set_bpm(self, bpm: float) -> None:
        self._bpm = max(20.0, min(300.0, float(bpm)))

    def start_clock(self) -> None:
        if not self._connected or self._clock_thread is not None:
            return
        self._clock_stop.clear()
        self._send_raw(MIDI_START)
        self._clock_thread = threading.Thread(
            target=self._clock_loop, name="MIDIClock", daemon=True
        )
        self._clock_thread.start()

    def stop_clock(self) -> None:
        if self._clock_thread is None:
            return
        self._clock_stop.set()
        self._clock_thread.join(timeout=0.5)
        self._clock_thread = None
        self._send_raw(MIDI_STOP)

    def continue_clock(self) -> None:
        self._send_raw(MIDI_CONTINUE)

    def _clock_loop(self) -> None:
        """Emit 24 PPQN clock pulses synced to ``self._bpm``."""
        while not self._clock_stop.is_set():
            interval = 60.0 / (self._bpm * config.MIDI_CLOCK_PPQN)
            self._send_raw(MIDI_CLOCK)
            time.sleep(interval)

    def _send_raw(self, byte: int) -> None:
        if not self._connected:
            return
        try:
            with self._lock:
                self._port.send_message([byte])
        except Exception:  # noqa: BLE001
            pass

    # ------------------------------------------------------------------
    def close(self) -> None:
        self.stop_clock()
        if self._port is not None:
            try:
                self._port.close_port()
                del self._port
            except Exception:  # noqa: BLE001
                pass
        self._connected = False
