"""
airdrums.audio.export
=====================
Offline render of recorded hit events into audio files.

Public functions
----------------
- :func:`export_wav`   – full stereo mixdown, normalised to −1 dBFS
- :func:`export_mp3`   – 320 kbps CBR MP3 transcoded from a WAV via pydub
- :func:`export_stems` – one WAV per drum group (cymbals, snare, hihat, toms)

Sample naming convention (both primary and legacy suffixes are recognised)::

    packs/<pack>/<drum>_ghost.wav   or  packs/<pack>/<drum>_pp.wav
    packs/<pack>/<drum>_soft.wav    or  packs/<pack>/<drum>_mp.wav
    packs/<pack>/<drum>_medium.wav  or  packs/<pack>/<drum>_mf.wav
    packs/<pack>/<drum>_hard.wav    or  packs/<pack>/<drum>_f.wav
    packs/<pack>/<drum>_accent.wav  or  packs/<pack>/<drum>_ff.wav
    packs/<pack>/<drum>.wav                        (no-suffix fallback)

All audio is rendered as 44 100 Hz 16-bit stereo PCM.

Parameters for ``events``
-------------------------
Each element must expose:

- ``timestamp_ms``  – float, milliseconds from session start
- ``drum_name``     – str, matches DrumLine.name
- ``midi_note``     – int
- ``velocity``      – int, MIDI velocity 1-127
- ``velocity_band`` – int, band index 0-4
- ``hand_side``     – str, ``"left"`` or ``"right"``
"""
from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np

from .. import config
from ..config import DrumLine

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Internal helpers – suffix tables
# ---------------------------------------------------------------------------

_PRIMARY_SUFFIXES: List[str] = [band.suffix.lstrip("_") for band in config.VELOCITY_BANDS]
_ALT_SUFFIXES: List[str]     = [band.alt_suffix.lstrip("_") for band in config.VELOCITY_BANDS]

# Map suffix token -> band index
_SUFFIX_TO_BAND: Dict[str, int] = {}
for _i, _band in enumerate(config.VELOCITY_BANDS):
    _SUFFIX_TO_BAND[_band.suffix.lstrip("_")] = _i
    _SUFFIX_TO_BAND[_band.alt_suffix.lstrip("_")] = _i

_FALLBACK_BAND = 2  # "medium" / "mf"


def _parse_stem(stem: str) -> tuple:
    """Return ``(drum_name, band_idx)`` for a WAV file stem.

    Tries to split off the last ``_<token>`` and match it against both primary
    and legacy suffix tables.  Falls back to the whole stem as drum name and
    band 2 (medium) if nothing matches.

    Parameters
    ----------
    stem:
        Filename without extension (e.g. ``"snare_medium"`` or ``"kick_mf"``).
    """
    if "_" in stem:
        parts = stem.rsplit("_", 1)
        token = parts[1]
        if token in _SUFFIX_TO_BAND:
            return parts[0], _SUFFIX_TO_BAND[token]
    return stem, _FALLBACK_BAND


# ---------------------------------------------------------------------------
# Sample loading (lightweight – avoids pygame for offline render)
# ---------------------------------------------------------------------------

def _load_pack_samples(
    pack_name: str,
) -> Dict[str, Dict[int, List[np.ndarray]]]:
    """Load all WAV files from a pack directory into float32 numpy arrays.

    Returns a nested dict: ``samples[drum_name][band_idx] = [array, ...]``
    where each array is ``(frames, 2)`` float32 stereo.

    Parameters
    ----------
    pack_name:
        Sub-directory of :data:`config.PACKS_DIR` to scan.
    """
    import soundfile as sf  # type: ignore

    pack_path: Path = config.PACKS_DIR / pack_name
    samples: Dict[str, Dict[int, List[np.ndarray]]] = {}

    if not pack_path.exists():
        log.warning("Sample pack '%s' not found at %s", pack_name, pack_path)
        return samples

    for wav in sorted(pack_path.glob("*.wav")):
        try:
            data, sr = sf.read(str(wav), always_2d=True)
            # Resample if needed (rare – most packs are already at 44100 Hz)
            if sr != config.AUDIO_SAMPLE_RATE:
                try:
                    import librosa  # type: ignore
                    data = librosa.resample(
                        data.T,
                        orig_sr=sr,
                        target_sr=config.AUDIO_SAMPLE_RATE,
                    ).T
                except Exception as exc:  # noqa: BLE001
                    log.debug("Resample failed for %s (%s); skipping", wav, exc)
                    continue

            # Convert to stereo if mono
            if data.shape[1] == 1:
                data = np.concatenate([data, data], axis=1)
            elif data.shape[1] > 2:
                data = data[:, :2]

            drum_name, band_idx = _parse_stem(wav.stem)
            samples.setdefault(drum_name, {}).setdefault(band_idx, []).append(
                data.astype(np.float32)
            )
        except Exception as exc:  # noqa: BLE001
            log.debug("Could not load %s: %s", wav, exc)

    total = sum(
        len(lst) for d in samples.values() for lst in d.values()
    )
    log.info("Export loader: %d samples from pack '%s'", total, pack_name)
    return samples


def _pick_sample(
    samples: Dict[str, Dict[int, List[np.ndarray]]],
    drum_name: str,
    band_idx: int,
    velocity: int,
    counter: Dict[str, int],
) -> Optional[np.ndarray]:
    """Return a velocity-scaled sample array for one hit, or ``None``.

    Round-robins through available samples for the drum+band pair.
    Falls back to any available band if the exact one has no samples.

    Parameters
    ----------
    samples:
        Pack sample dict returned by :func:`_load_pack_samples`.
    drum_name:
        Name of the drum (matches :class:`~airdrums.config.DrumLine` ``name``).
    band_idx:
        Velocity band index 0-4.
    velocity:
        MIDI velocity 1-127 for amplitude scaling.
    counter:
        Mutable round-robin counter dict (keyed by ``drum_name``).
    """
    drum_data = samples.get(drum_name)
    if not drum_data:
        return None

    options = drum_data.get(band_idx)
    if not options:
        for fallback_idx in sorted(drum_data.keys()):
            options = drum_data[fallback_idx]
            if options:
                break

    if not options:
        return None

    rr_key = f"{drum_name}:{band_idx}"
    idx = counter.get(rr_key, 0) % len(options)
    counter[rr_key] = idx + 1
    return options[idx] * (velocity / 127.0)


# ---------------------------------------------------------------------------
# Drum group classification helpers
# ---------------------------------------------------------------------------

_GROUP_KEYWORDS: Dict[str, List[str]] = {
    "cymbals": ["crash", "ride", "splash", "china"],
    "hihat":   ["hihat", "hi_hat", "hh"],
    "snare":   ["snare", "rim", "clap"],
    "toms":    ["tom", "floor"],
}

_GROUP_ORDER = ["cymbals", "hihat", "snare", "toms", "kick"]


def _group_for_drum_line(dl: DrumLine) -> str:
    """Return the stem-group name for a :class:`DrumLine`.

    Cymbals (``is_cymbal=True``) are always placed in the ``"cymbals"`` group.
    Non-cymbals are matched by keyword against ``DrumLine.name``.

    Parameters
    ----------
    dl:
        A drum line definition.
    """
    if dl.is_cymbal:
        return "cymbals"
    name_lower = dl.name.lower()
    for group, keywords in _GROUP_KEYWORDS.items():
        if group == "cymbals":
            continue
        if any(kw in name_lower for kw in keywords):
            return group
    return "kick"


# ---------------------------------------------------------------------------
# Core renderer
# ---------------------------------------------------------------------------

def _render(
    events: Iterable,
    drum_lines: List[DrumLine],
    pack_name: str,
    stem_filter: Optional[str] = None,
) -> np.ndarray:
    """Render hit events to an interleaved stereo float32 buffer.

    Parameters
    ----------
    events:
        Iterable of hit event objects (see module docstring for required fields).
    drum_lines:
        Drum line definitions used to look up ``band_idx`` from ``drum_name``.
    pack_name:
        Sample pack to use for rendering.
    stem_filter:
        If given, only events whose drum group matches this string are rendered.

    Returns
    -------
    np.ndarray
        Shape ``(N_frames, 2)``, dtype ``float32``, normalised to −1 dBFS.
    """
    samples = _load_pack_samples(pack_name)
    event_list = list(events)

    if not event_list:
        return np.zeros((config.AUDIO_SAMPLE_RATE, 2), dtype=np.float32)

    # Build a name -> group map for filtering
    name_to_group: Dict[str, str] = {dl.name: _group_for_drum_line(dl) for dl in drum_lines}
    # Build a name -> band_idx map (not needed here; band comes from event)
    # but keep for reference; the event already carries velocity / band info.

    duration_ms = max(ev.timestamp_ms for ev in event_list) + 4000.0
    total_frames = int(duration_ms / 1000.0 * config.AUDIO_SAMPLE_RATE)
    buf = np.zeros((total_frames, 2), dtype=np.float32)
    counter: Dict[str, int] = {}

    for ev in event_list:
        if stem_filter is not None:
            group = name_to_group.get(ev.drum_name, "kick")
            if group != stem_filter:
                continue

        band_idx = getattr(ev, "velocity_band", _FALLBACK_BAND)
        sample = _pick_sample(samples, ev.drum_name, band_idx, ev.velocity, counter)
        if sample is None:
            continue

        start = int(ev.timestamp_ms / 1000.0 * config.AUDIO_SAMPLE_RATE)
        end = min(total_frames, start + sample.shape[0])
        if end > start:
            clip = sample[: end - start]
            if clip.ndim == 1:
                clip = np.stack([clip, clip], axis=1)
            buf[start:end] += clip

    # Normalise to −1 dBFS
    peak = float(np.max(np.abs(buf)))
    if peak > 0.0:
        target = 10 ** (-1.0 / 20.0)   # ≈ 0.891
        buf *= target / peak

    return buf


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def export_wav(
    events: Iterable,
    drum_lines: List[DrumLine],
    pack_name: str,
    out_path: "str | Path",
) -> Path:
    """Render hit events to a 16-bit stereo WAV file normalised to −1 dBFS.

    Parameters
    ----------
    events:
        Iterable of hit event objects.
    drum_lines:
        Drum line definitions (list of :class:`~airdrums.config.DrumLine`).
    pack_name:
        Sample pack sub-directory name under :data:`config.PACKS_DIR`.
    out_path:
        Destination file path (will be created or overwritten).

    Returns
    -------
    Path
        Absolute path of the written WAV file.
    """
    import soundfile as sf  # type: ignore

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    buf = _render(events, drum_lines, pack_name)
    sf.write(str(out_path), buf, config.AUDIO_SAMPLE_RATE, subtype="PCM_16")
    log.info("Wrote mixdown WAV: %s", out_path)
    return out_path


def export_mp3(
    wav_path: "str | Path",
    out_path: "str | Path",
) -> Path:
    """Transcode a WAV file to 320 kbps CBR MP3 via pydub + ffmpeg.

    Parameters
    ----------
    wav_path:
        Path to a WAV file previously written by :func:`export_wav`.
    out_path:
        Destination MP3 file path.

    Returns
    -------
    Path
        Absolute path of the written MP3 file.
    """
    from pydub import AudioSegment  # type: ignore

    wav_path = Path(wav_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    seg = AudioSegment.from_wav(str(wav_path))
    seg.export(str(out_path), format="mp3", bitrate="320k")
    log.info("Wrote MP3: %s", out_path)
    return out_path


def export_stems(
    events: Iterable,
    drum_lines: List[DrumLine],
    pack_name: str,
    out_dir: "str | Path",
) -> List[Path]:
    """Render one WAV stem per drum group in parallel.

    Groups are: ``cymbals``, ``hihat``, ``snare``, ``toms``, ``kick``.
    Each stem is normalised independently to −1 dBFS.

    Parameters
    ----------
    events:
        Iterable of hit event objects.  Consumed once; buffered internally.
    drum_lines:
        Drum line definitions for group classification.
    pack_name:
        Sample pack sub-directory name.
    out_dir:
        Directory in which ``stem_<group>.wav`` files will be written.

    Returns
    -------
    List[Path]
        Paths of all written stem WAV files.
    """
    import soundfile as sf  # type: ignore

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Buffer the events so each thread can iterate them independently
    event_list = list(events)

    paths: List[Path] = []
    paths_lock = threading.Lock()

    def _render_one(group: str) -> None:
        buf = _render(event_list, drum_lines, pack_name, stem_filter=group)
        p = out_dir / f"stem_{group}.wav"
        sf.write(str(p), buf, config.AUDIO_SAMPLE_RATE, subtype="PCM_16")
        log.info("Wrote stem: %s", p)
        with paths_lock:
            paths.append(p)

    threads = [
        threading.Thread(target=_render_one, args=(group,), daemon=True)
        for group in _GROUP_ORDER
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    return sorted(paths)
