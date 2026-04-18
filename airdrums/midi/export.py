"""
airdrums.midi.export
====================
Write a standard Type-1 MIDI file from a recorded session.

* All drums on track 1, channel 10 (0-indexed 9).
* Tempo map embedded from BPM history.
* Optional quantization to 1/8, 1/16, or 1/32 note grids.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

from .. import config

log = logging.getLogger(__name__)

TICKS_PER_BEAT = 480


def _quantize_ms(ts_ms: float, bpm: float, grid: str) -> float:
    if grid == "none":
        return ts_ms
    denominator = {"1/8": 2, "1/16": 4, "1/32": 8}.get(grid)
    if not denominator:
        return ts_ms
    beat_ms = 60000.0 / bpm
    step = beat_ms / denominator
    return round(ts_ms / step) * step


def export_mid(session, bpm: float, path: str | Path, quantize: str = "none") -> Path:
    """Serialise a Session to a Type-1 MIDI file at ``path``."""
    import mido  # type: ignore
    path = Path(path)

    mid = mido.MidiFile(type=1, ticks_per_beat=TICKS_PER_BEAT)
    meta_track = mido.MidiTrack()
    mid.tracks.append(meta_track)

    # Tempo map: initial tempo from requested bpm, then any bpm_history entries
    tempo_us_per_beat = mido.bpm2tempo(bpm)
    meta_track.append(mido.MetaMessage("set_tempo", tempo=tempo_us_per_beat, time=0))
    meta_track.append(mido.MetaMessage("track_name", name="AirDrums Tempo", time=0))
    meta_track.append(mido.MetaMessage("time_signature",
                                       numerator=4, denominator=4, time=0))

    last_tick = 0
    for ts_ms, b in sorted(getattr(session, "bpm_history", []), key=lambda p: p[0]):
        tick = int((ts_ms / 1000.0) * (bpm / 60.0) * TICKS_PER_BEAT)
        delta = max(0, tick - last_tick)
        meta_track.append(mido.MetaMessage("set_tempo",
                                           tempo=mido.bpm2tempo(b), time=delta))
        last_tick = tick

    # Drum track
    drum_track = mido.MidiTrack()
    mid.tracks.append(drum_track)
    drum_track.append(mido.MetaMessage("track_name", name="AirDrums", time=0))

    # Quantize + sort events
    events = list(session.events) if hasattr(session, "events") else list(session)
    events = sorted(events, key=lambda e: e.timestamp_ms)
    ch = config.MIDI_CHANNEL & 0x0F
    last_tick = 0

    # Build interleaved on/off list then sort by absolute tick
    abs_events: list[tuple[int, str, int, int]] = []
    for ev in events:
        ts = _quantize_ms(ev.timestamp_ms, bpm, quantize)
        on_tick = int((ts / 1000.0) * (bpm / 60.0) * TICKS_PER_BEAT)
        off_tick = on_tick + int((config.MIDI_NOTE_OFF_MS / 1000.0)
                                 * (bpm / 60.0) * TICKS_PER_BEAT)
        abs_events.append((on_tick, "on", ev.midi_note, ev.velocity))
        abs_events.append((off_tick, "off", ev.midi_note, 0))

    abs_events.sort(key=lambda e: (e[0], 0 if e[1] == "off" else 1))
    for tick, kind, note, vel in abs_events:
        delta = max(0, tick - last_tick)
        if kind == "on":
            drum_track.append(mido.Message("note_on", channel=ch,
                                           note=note, velocity=vel, time=delta))
        else:
            drum_track.append(mido.Message("note_off", channel=ch,
                                           note=note, velocity=0, time=delta))
        last_tick = tick

    mid.save(str(path))
    log.info("Wrote MIDI file: %s", path)
    return path
