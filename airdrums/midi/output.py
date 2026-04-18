"""
airdrums.midi.output
====================
Live MIDI output to a virtual port via python-rtmidi.

Any DAW that scans MIDI inputs (Ableton Live, FL Studio, Logic Pro,
GarageBand, Reaper, …) will see the ``"AirDrums"`` port automatically on
macOS and Linux.  On Windows the code falls back to opening the first
available hardware MIDI output port.

MIDI clock (24 PPQN) is emitted in a background thread, synced to a
settable BPM.  Standard Start / Stop / Continue transport messages are
sent when the clock starts or stops.

Configuration constants used
----------------------------
- :data:`config.MIDI_PORT_NAME`       port name advertised to DAWs
- :data:`config.MIDI_CHANNEL`         0-indexed channel (9 → GM channel 10)
- :data:`config.MIDI_NOTE_OFF_DELAY_MS`  delay before Note Off message
- :data:`config.MIDI_PPQN`            clock pulses per quarter note
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Optional

from .. import config

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# MIDI status bytes
# ---------------------------------------------------------------------------
_NOTE_ON    = 0x90
_NOTE_OFF   = 0x80
_MIDI_CLOCK = 0xF8
_MIDI_START = 0xFA
_MIDI_CONT  = 0xFB
_MIDI_STOP  = 0xFC


class MidiOutput:
    """Virtual MIDI port with Note On/Off and MIDI clock support.

    Parameters
    ----------
    port_name:
        Name shown to DAWs when advertising the virtual port.  Defaults to
        :data:`config.MIDI_PORT_NAME`.
    """

    def __init__(self, port_name: str = config.MIDI_PORT_NAME) -> None:
        """Open the MIDI output port (virtual if supported by the OS)."""
        self.port_name = port_name
        self._port = None
        self._connected = False
        self._bpm: float = 120.0
        self._clock_stop = threading.Event()
        self._clock_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._open()

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    def _open(self) -> None:
        """Open a virtual MIDI output port, falling back to a hardware port."""
        try:
            import rtmidi  # type: ignore

            self._port = rtmidi.MidiOut()
            try:
                # Virtual ports work natively on macOS/Linux.
                self._port.open_virtual_port(self.port_name)
                log.info("Opened virtual MIDI output: '%s'", self.port_name)
            except (NotImplementedError, AttributeError):
                # Windows: open the first available hardware port.
                ports = self._port.get_ports()
                if ports:
                    self._port.open_port(0)
                    log.info("Opened hardware MIDI output: %s", ports[0])
                else:
                    raise RuntimeError("No MIDI output ports available")
            self._connected = True
        except Exception as exc:  # noqa: BLE001
            log.warning("MIDI output unavailable (%s); running silently", exc)
            self._port = None
            self._connected = False

    @property
    def connected(self) -> bool:
        """``True`` if a MIDI port is open and ready to send messages."""
        return self._connected

    # ------------------------------------------------------------------
    # Note messages
    # ------------------------------------------------------------------

    def send_hit(self, midi_note: int, velocity: int) -> None:
        """Send a Note On message, then a Note Off after the configured delay.

        The Note Off is dispatched asynchronously via a daemon timer so this
        method returns immediately and never blocks the caller.

        Parameters
        ----------
        midi_note:
            MIDI note number 0-127.
        velocity:
            MIDI velocity 1-127.  A velocity of 0 is silently clamped to 1.
        """
        if not self._connected:
            return

        ch = config.MIDI_CHANNEL & 0x0F
        note = midi_note & 0x7F
        vel  = max(1, velocity & 0x7F)

        try:
            with self._lock:
                self._port.send_message([_NOTE_ON | ch, note, vel])
        except Exception as exc:  # noqa: BLE001
            log.debug("send_hit failed (note %d): %s", midi_note, exc)
            return

        delay_s = config.MIDI_NOTE_OFF_DELAY_MS / 1000.0
        timer = threading.Timer(delay_s, self._send_note_off, args=(midi_note,))
        timer.daemon = True
        timer.start()

    def _send_note_off(self, midi_note: int) -> None:
        """Send a Note Off message for *midi_note* on the drum channel.

        Parameters
        ----------
        midi_note:
            MIDI note number 0-127.
        """
        if not self._connected:
            return
        ch = config.MIDI_CHANNEL & 0x0F
        try:
            with self._lock:
                self._port.send_message([_NOTE_OFF | ch, midi_note & 0x7F, 0])
        except Exception:  # noqa: BLE001
            pass

    # ------------------------------------------------------------------
    # MIDI clock / transport
    # ------------------------------------------------------------------

    def set_bpm(self, bpm: float) -> None:
        """Update the BPM used by the MIDI clock thread.

        The change takes effect on the next clock pulse interval without
        interrupting the clock thread.

        Parameters
        ----------
        bpm:
            Tempo in beats per minute; clamped to [20, 300].
        """
        self._bpm = max(20.0, min(300.0, float(bpm)))
        log.debug("MIDI clock BPM set to %.1f", self._bpm)

    def start_clock(self) -> None:
        """Send MIDI Start and begin emitting clock pulses at the current BPM.

        If the clock is already running this method is a no-op.
        """
        if not self._connected or self._clock_thread is not None:
            return
        self._clock_stop.clear()
        self._send_raw(_MIDI_START)
        self._clock_thread = threading.Thread(
            target=self._clock_loop,
            name="MIDIClock",
            daemon=True,
        )
        self._clock_thread.start()
        log.info("MIDI clock started at %.1f BPM", self._bpm)

    def stop_clock(self) -> None:
        """Stop the clock thread and send a MIDI Stop message."""
        if self._clock_thread is None:
            return
        self._clock_stop.set()
        self._clock_thread.join(timeout=0.5)
        self._clock_thread = None
        self._send_raw(_MIDI_STOP)
        log.info("MIDI clock stopped")

    def continue_clock(self) -> None:
        """Send a MIDI Continue message without resetting the clock position."""
        self._send_raw(_MIDI_CONT)

    def _clock_loop(self) -> None:
        """Background thread: emit ``MIDI_PPQN`` clock pulses per beat."""
        while not self._clock_stop.is_set():
            interval = 60.0 / (self._bpm * config.MIDI_PPQN)
            self._send_raw(_MIDI_CLOCK)
            time.sleep(interval)

    def _send_raw(self, byte: int) -> None:
        """Send a single-byte system real-time message.

        Parameters
        ----------
        byte:
            Raw MIDI status byte (e.g. ``0xF8`` for clock).
        """
        if not self._connected:
            return
        try:
            with self._lock:
                self._port.send_message([byte])
        except Exception:  # noqa: BLE001
            pass

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Stop the clock and release the MIDI port."""
        self.stop_clock()
        if self._port is not None:
            try:
                self._port.close_port()
                del self._port
            except Exception:  # noqa: BLE001
                pass
        self._connected = False
        log.info("MIDI output closed")
