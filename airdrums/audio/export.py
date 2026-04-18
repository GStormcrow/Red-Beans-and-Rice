"""
airdrums.audio.export
=====================
Offline render of a recorded :class:`recording.session.Session` into audio
files: a full stereo mixdown WAV, a 320kbps MP3, and per-drum stems.
"""
from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Dict, List

import numpy as np

from .. import config

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Sample loading (lightweight; avoids pygame for offline render)
# ---------------------------------------------------------------------------
def _load_pack_samples(pack_name: str) -> Dict[str, Dict[str, List[np.ndarray]]]:
    import soundfile as sf  # type: ignore
    pack_path = config.SAMPLE_PACKS_DIR / pack_name
    samples: Dict[str, Dict[str, List[np.ndarray]]] = {}
    if not pack_path.exists():
        return samples
    for wav in sorted(pack_path.glob("*.wav")):
        try:
            data, sr = sf.read(str(wav), always_2d=True)
            if sr != config.AUDIO_SAMPLE_RATE:
                # Simple resample via librosa if needed
                import librosa  # type: ignore
                data = librosa.resample(data.T, orig_sr=sr, target_sr=config.AUDIO_SAMPLE_RATE).T
            stem = wav.stem
            parts = stem.rsplit("_", 1)
            layer = parts[-1] if parts[-1] in config.VELOCITY_LAYERS else "mf"
            drum = parts[0] if parts[-1] in config.VELOCITY_LAYERS else stem
            samples.setdefault(drum, {}).setdefault(layer, []).append(data.astype(np.float32))
        except Exception as exc:  # noqa: BLE001
            log.debug("Could not load %s: %s", wav, exc)
    return samples


def _pick_sample(samples: Dict[str, Dict[str, List[np.ndarray]]], drum: str,
                 velocity: int, counter: Dict[str, int]) -> np.ndarray | None:
    drum_key = drum.lower().replace(" ", "_").replace("-", "_")
    drum_key = {"hi_hat_closed": "hihat_closed",
                "hi_hat_pedal": "hihat_closed",
                "bass_drum": "kick"}.get(drum_key, drum_key)
    if drum_key not in samples:
        return None
    layer = "pp" if velocity < 40 else "mp" if velocity < 70 else \
            "mf" if velocity < 95 else "f" if velocity < 115 else "ff"
    options = samples[drum_key].get(layer)
    if not options:
        for l in config.VELOCITY_LAYERS:
            if l in samples[drum_key]:
                options = samples[drum_key][l]
                break
    if not options:
        return None
    idx = counter.get(drum_key, 0) % len(options)
    counter[drum_key] = counter.get(drum_key, 0) + 1
    return options[idx] * (velocity / 127.0)


# ---------------------------------------------------------------------------
# Mixdown
# ---------------------------------------------------------------------------
def _render(session, sample_pack: str,
            stem_filter: str | None = None) -> np.ndarray:
    """Render the session to an interleaved stereo float32 buffer."""
    samples = _load_pack_samples(sample_pack)
    events = session.events  # list of HitEvent
    if not events:
        return np.zeros((config.AUDIO_SAMPLE_RATE, 2), dtype=np.float32)

    duration_ms = max(ev.timestamp_ms for ev in events) + 4000
    total_frames = int(duration_ms / 1000.0 * config.AUDIO_SAMPLE_RATE)
    buf = np.zeros((total_frames, 2), dtype=np.float32)
    counter: Dict[str, int] = {}

    for ev in events:
        if stem_filter is not None:
            zone = config.DRUM_ZONES_BY_NAME.get(ev.drum_name)
            if zone is None or zone.stem_group != stem_filter:
                continue
        sample = _pick_sample(samples, ev.drum_name, ev.velocity, counter)
        if sample is None:
            continue
        start = int(ev.timestamp_ms / 1000.0 * config.AUDIO_SAMPLE_RATE)
        end = min(total_frames, start + sample.shape[0])
        if end > start:
            slice_in = sample[: end - start]
            if slice_in.ndim == 1:
                slice_in = np.stack([slice_in, slice_in], axis=1)
            buf[start:end] += slice_in

    # Normalize to -1 dBFS
    peak = float(np.max(np.abs(buf))) or 1.0
    buf *= (10 ** (-1.0 / 20.0)) / peak
    return buf


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def export_wav(session, sample_pack: str, path: str | Path) -> Path:
    """Render the full mixdown to a WAV file. Non-blocking via a thread."""
    import soundfile as sf  # type: ignore
    path = Path(path)
    buf = _render(session, sample_pack)
    sf.write(str(path), buf, config.AUDIO_SAMPLE_RATE, subtype="PCM_16")
    log.info("Wrote mixdown WAV: %s", path)
    return path


def export_mp3(wav_path: str | Path, output_path: str | Path) -> Path:
    """Transcode a WAV render to 320kbps CBR MP3 via pydub + ffmpeg."""
    from pydub import AudioSegment  # type: ignore
    wav_path = Path(wav_path)
    output_path = Path(output_path)
    seg = AudioSegment.from_wav(str(wav_path))
    seg.export(str(output_path), format="mp3", bitrate="320k")
    log.info("Wrote MP3: %s", output_path)
    return output_path


def export_stems(session, sample_pack: str, output_dir: str | Path) -> List[Path]:
    """Render one WAV per stem group: kick, snare, hihat, toms, cymbals."""
    import soundfile as sf  # type: ignore
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    groups = {"kick", "snare", "hihat", "toms", "cymbals"}
    paths: List[Path] = []

    def _one(group: str) -> None:
        buf = _render(session, sample_pack, stem_filter=group)
        p = output_dir / f"stem_{group}.wav"
        sf.write(str(p), buf, config.AUDIO_SAMPLE_RATE, subtype="PCM_16")
        paths.append(p)
        log.info("Wrote stem: %s", p)

    threads = [threading.Thread(target=_one, args=(g,)) for g in groups]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return paths
