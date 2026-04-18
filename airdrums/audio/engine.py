"""
airdrums.audio.engine
=====================
Low-latency sample playback via pygame.mixer.

Sample pack layout::

    packs/<pack_name>/<drum_name>_<layer>.wav

where ``<layer>`` is one of ``pp, mp, mf, f, ff``. Hi-hat samples get three
state buckets: ``hihat_closed_*.wav``, ``hihat_open_*.wav``, ``hihat_half_*.wav``
plus a pedal-chick file ``hihat_chick.wav``. Multiple files per drum/layer are
cycled round-robin to avoid the "machine gun" effect on rolls.
"""
from __future__ import annotations

import logging
import random
import threading
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

from .. import config

log = logging.getLogger(__name__)


class AudioEngine:
    """pygame.mixer wrapper that plays drum samples with velocity layers."""

    def __init__(self, sample_pack: str = "default"):
        self.sample_pack = sample_pack
        self.pack_path = config.SAMPLE_PACKS_DIR / sample_pack
        self._samples: Dict[str, Dict[str, List["_Sound"]]] = defaultdict(lambda: defaultdict(list))
        self._round_robin: Dict[str, int] = defaultdict(int)
        self._lock = threading.Lock()
        self._hihat_state = "closed"
        self._initialised = False
        self._init_mixer()
        if self._initialised:
            self._load_pack()

    # ------------------------------------------------------------------
    def _init_mixer(self) -> None:
        try:
            import pygame  # type: ignore
            pygame.mixer.pre_init(
                frequency=config.AUDIO_SAMPLE_RATE,
                size=-config.AUDIO_BIT_DEPTH,
                channels=config.AUDIO_CHANNELS,
                buffer=config.AUDIO_BUFFER_SIZE,
            )
            pygame.mixer.init()
            pygame.mixer.set_num_channels(32)
            self._pygame = pygame
            self._initialised = True
        except Exception as exc:  # noqa: BLE001
            log.warning("pygame mixer unavailable (%s); audio disabled", exc)
            self._pygame = None
            self._initialised = False

    # ------------------------------------------------------------------
    def _load_pack(self) -> None:
        if not self.pack_path.exists():
            log.warning("Sample pack %s not found; creating empty placeholder", self.pack_path)
            self.pack_path.mkdir(parents=True, exist_ok=True)
            return
        for wav in sorted(self.pack_path.glob("*.wav")):
            stem = wav.stem
            # Parse "<drum>_<layer>" or "<drum>_<layer>_<idx>"
            parts = stem.rsplit("_", 1)
            layer = parts[-1] if parts[-1] in config.VELOCITY_LAYERS else "mf"
            drum = parts[0] if parts[-1] in config.VELOCITY_LAYERS else stem
            try:
                snd = self._pygame.mixer.Sound(str(wav))
                self._samples[drum][layer].append(_Sound(snd))
            except Exception as exc:  # noqa: BLE001
                log.debug("Failed to load %s: %s", wav, exc)
        total = sum(len(l) for d in self._samples.values() for l in d.values())
        log.info("Loaded %d samples from %s", total, self.pack_path)

    # ------------------------------------------------------------------
    def set_hihat_state(self, state: str) -> None:
        self._hihat_state = state

    # ------------------------------------------------------------------
    def play(self, drum_name: str, midi_velocity: int) -> None:
        """Play the correct velocity-layered sample for a drum hit."""
        if not self._initialised:
            return
        drum_key = self._resolve_drum_key(drum_name)
        layer = self._pick_layer(midi_velocity)
        with self._lock:
            options = self._samples.get(drum_key, {}).get(layer)
            if not options:
                # Fall back to any available layer
                for l in config.VELOCITY_LAYERS:
                    options = self._samples.get(drum_key, {}).get(l)
                    if options:
                        break
            if not options:
                return
            idx = self._round_robin[drum_key] % len(options)
            self._round_robin[drum_key] += 1
            sound = options[idx]
        try:
            channel = sound.sound.play()
            if channel is not None:
                channel.set_volume(self._velocity_to_volume(midi_velocity))
        except Exception as exc:  # noqa: BLE001
            log.debug("play failed (%s): %s", drum_name, exc)

    # ------------------------------------------------------------------
    def play_pedal_chick(self) -> None:
        self.play("hihat_chick", 80)

    # ------------------------------------------------------------------
    def _resolve_drum_key(self, drum_name: str) -> str:
        # Map HH to state-appropriate sample bank
        if drum_name in ("Hi-Hat Closed", "Hi-Hat Pedal"):
            if self._hihat_state == "open":
                return "hihat_open"
            if self._hihat_state in ("closing", "opening"):
                return "hihat_half"
            return "hihat_closed"
        return drum_name.lower().replace(" ", "_").replace("-", "_")

    @staticmethod
    def _pick_layer(velocity: int) -> str:
        if velocity < 40:
            return "pp"
        if velocity < 70:
            return "mp"
        if velocity < 95:
            return "mf"
        if velocity < 115:
            return "f"
        return "ff"

    @staticmethod
    def _velocity_to_volume(velocity: int) -> float:
        return max(0.05, min(1.0, velocity / 127.0))

    # ------------------------------------------------------------------
    def shutdown(self) -> None:
        if self._initialised and self._pygame is not None:
            try:
                self._pygame.mixer.quit()
            except Exception:  # noqa: BLE001
                pass


class _Sound:
    """Thin wrapper so we can extend later (envelopes, etc.)."""
    __slots__ = ("sound",)

    def __init__(self, sound):
        self.sound = sound
