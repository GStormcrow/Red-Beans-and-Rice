"""
airdrums.midi.export
====================
Write a standard Type-1 MIDI file from a list of recorded hit events.

Structure
---------
- **Track 0** – meta track: tempo map (from ``bpm_history``), time signature.
- **Track 1** – one per drum line that has at least one hit event.

Quantization options
--------------------
``"none"``, ``"1/8"``, ``"1/16"`` (default), ``"1/32"``

Hit event fields expected
-------------------------
Each element in *events* must expose:

- ``timestamp_ms``  – float, milliseconds from session start
- ``drum_name``     – str, matches :class:`~airdrums.config.DrumLine` ``name``
- ``midi_note``     – int, MIDI note number
- ``velocity``      – int, MIDI velocity 1-127
- ``velocity_band`` – int, velocity band index 0-4
- ``hand_side``     – str, ``"left"`` or ``"right"``

Configuration constants used
----------------------------
- :data:`config.MIDI_CHANNEL`           0-indexed channel (9 → GM channel 10)
- :data:`config.MIDI_NOTE_OFF_DELAY_MS` duration of each note in ms
- :data:`config.MIDI_PPQN`             pulses per quarter note (for clock ref)
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from .. import config
from ..config import DrumLine

log = logging.getLogger(__name__)

# Internal MIDI resolution for the exported file (independent of live PPQN)
_TICKS_PER_BEAT = 480

_QUANTIZE_DENOMINATORS: Dict[str, int] = {
    "1/8":  2,
    "1/16": 4,
    "1/32": 8,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _quantize_ms(ts_ms: float, bpm: float, grid: str) -> float:
    """Snap *ts_ms* to the nearest grid boundary.

    Parameters
    ----------
    ts_ms:
        Original event timestamp in milliseconds.
    bpm:
        Current tempo used to calculate grid step size.
    grid:
        One of ``"none"``, ``"1/8"``, ``"1/16"``, ``"1/32"``.

    Returns
    -------
    float
        Quantized timestamp in milliseconds.
    """
    if grid == "none" or grid not in _QUANTIZE_DENOMINATORS:
        return ts_ms
    beat_ms = 60_000.0 / bpm
    step_ms = beat_ms / _QUANTIZE_DENOMINATORS[grid]
    return round(ts_ms / step_ms) * step_ms


def _ms_to_ticks(ms: float, bpm: float) -> int:
    """Convert a millisecond offset to MIDI ticks at *bpm*.

    Parameters
    ----------
    ms:
        Time offset in milliseconds.
    bpm:
        Tempo in beats per minute.

    Returns
    -------
    int
        Absolute MIDI tick position.
    """
    beats = ms / 1000.0 * (bpm / 60.0)
    return int(beats * _TICKS_PER_BEAT)


def _note_off_ticks(bpm: float) -> int:
    """Return the tick duration of a note (Note On → Note Off gap).

    Parameters
    ----------
    bpm:
        Current tempo used to convert milliseconds to ticks.

    Returns
    -------
    int
        Tick count for :data:`config.MIDI_NOTE_OFF_DELAY_MS`.
    """
    return _ms_to_ticks(config.MIDI_NOTE_OFF_DELAY_MS, bpm)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def export_mid(
    events: Iterable,
    drum_lines: List[DrumLine],
    bpm_history: Iterable,
    out_path: "str | Path",
    quantization: str = "1/16",
) -> Path:
    """Serialise hit events to a Type-1 MIDI file.

    One MIDI track is created per drum line that has at least one matching
    event.  Track names are taken from :attr:`DrumLine.label`.  A meta
    track at index 0 contains the full tempo map built from *bpm_history*.

    Parameters
    ----------
    events:
        Iterable of hit event objects (see module docstring for fields).
    drum_lines:
        Drum line definitions; their ``label`` is used as MIDI track name.
    bpm_history:
        Iterable of ``(timestamp_ms, bpm)`` tuples representing tempo changes
        detected during the session.  Pass an empty list for a fixed tempo.
    out_path:
        Destination ``.mid`` file path (created or overwritten).
    quantization:
        Snap-to-grid setting: ``"none"``, ``"1/8"``, ``"1/16"``, ``"1/32"``.
        Defaults to ``"1/16"``.

    Returns
    -------
    Path
        Absolute path of the written MIDI file.

    Raises
    ------
    ImportError
        If the ``mido`` package is not installed.
    """
    import mido  # type: ignore

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    event_list = sorted(events, key=lambda e: e.timestamp_ms)

    # Determine a base BPM from bpm_history or a sane default
    bpm_list: List[Tuple[float, float]] = sorted(
        bpm_history, key=lambda p: p[0]
    )
    base_bpm: float = bpm_list[0][1] if bpm_list else 120.0

    # ------------------------------------------------------------------
    # MIDI file skeleton
    # ------------------------------------------------------------------
    mid = mido.MidiFile(type=1, ticks_per_beat=_TICKS_PER_BEAT)

    # ------------------------------------------------------------------
    # Track 0 – tempo map + time signature
    # ------------------------------------------------------------------
    meta_track = mido.MidiTrack()
    mid.tracks.append(meta_track)
    meta_track.append(mido.MetaMessage("track_name", name="AirDrums Tempo", time=0))
    meta_track.append(
        mido.MetaMessage("time_signature", numerator=4, denominator=4, time=0)
    )
    meta_track.append(
        mido.MetaMessage(
            "set_tempo", tempo=mido.bpm2tempo(base_bpm), time=0
        )
    )

    last_meta_tick = 0
    for ts_ms, b in bpm_list:
        tick = _ms_to_ticks(ts_ms, base_bpm)
        delta = max(0, tick - last_meta_tick)
        meta_track.append(
            mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(b), time=delta)
        )
        last_meta_tick = tick

    # ------------------------------------------------------------------
    # Per-drum tracks
    # ------------------------------------------------------------------
    ch = config.MIDI_CHANNEL & 0x0F

    # Build a map from drum_name -> list of events
    name_to_events: Dict[str, list] = {}
    for ev in event_list:
        name_to_events.setdefault(ev.drum_name, []).append(ev)

    # Build a map from drum_name -> DrumLine for label lookup
    name_to_line: Dict[str, DrumLine] = {dl.name: dl for dl in drum_lines}

    for dl in drum_lines:
        drum_events = name_to_events.get(dl.name)
        if not drum_events:
            continue  # skip drums with no hits

        drum_track = mido.MidiTrack()
        mid.tracks.append(drum_track)
        drum_track.append(
            mido.MetaMessage("track_name", name=dl.label, time=0)
        )

        # Build a flat list of (absolute_tick, type, note, velocity)
        abs_events: List[Tuple[int, str, int, int]] = []
        note_off_dur = _note_off_ticks(base_bpm)

        for ev in drum_events:
            bpm_at_event = _bpm_at(bpm_list, ev.timestamp_ms, base_bpm)
            ts_q = _quantize_ms(ev.timestamp_ms, bpm_at_event, quantization)
            on_tick  = _ms_to_ticks(ts_q, base_bpm)
            off_tick = on_tick + note_off_dur
            vel = max(1, min(127, ev.velocity))
            abs_events.append((on_tick,  "on",  ev.midi_note, vel))
            abs_events.append((off_tick, "off", ev.midi_note, 0))

        # Sort: ascending tick; Note Off before Note On at same tick
        abs_events.sort(key=lambda e: (e[0], 0 if e[1] == "off" else 1))

        last_tick = 0
        for tick, kind, note, vel in abs_events:
            delta = max(0, tick - last_tick)
            if kind == "on":
                drum_track.append(
                    mido.Message(
                        "note_on", channel=ch,
                        note=note & 0x7F, velocity=vel, time=delta,
                    )
                )
            else:
                drum_track.append(
                    mido.Message(
                        "note_off", channel=ch,
                        note=note & 0x7F, velocity=0, time=delta,
                    )
                )
            last_tick = tick

    mid.save(str(out_path))
    log.info(
        "Wrote MIDI file: %s  (%d drum tracks, quantization=%s)",
        out_path,
        len(mid.tracks) - 1,
        quantization,
    )
    return out_path


# ---------------------------------------------------------------------------
# Internal utilities
# ---------------------------------------------------------------------------

def _bpm_at(
    bpm_history: List[Tuple[float, float]],
    timestamp_ms: float,
    default_bpm: float,
) -> float:
    """Return the BPM in effect at *timestamp_ms* according to *bpm_history*.

    Parameters
    ----------
    bpm_history:
        Sorted list of ``(timestamp_ms, bpm)`` change points.
    timestamp_ms:
        Query time in milliseconds.
    default_bpm:
        Fallback value when *bpm_history* is empty.

    Returns
    -------
    float
        BPM value active at the requested time.
    """
    current = default_bpm
    for ts, b in bpm_history:
        if ts <= timestamp_ms:
            current = b
        else:
            break
    return current
