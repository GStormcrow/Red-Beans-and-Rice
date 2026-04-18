"""
airdrums.audio.engine
=====================
Low-latency sample playback via pygame.mixer.

Sample pack layout::

    packs/<pack_name>/<drum_name><band_suffix>.wav

where ``<band_suffix>`` is one of the primary suffixes
(``_ghost``, ``_soft``, ``_medium``, ``_hard``, ``_accent``) or the
legacy alt-suffixes (``_pp``, ``_mp``, ``_mf``, ``_f``, ``_ff``).
Multiple files per drum/band are cycled round-robin to avoid the
"machine-gun" effect on fast rolls.

Cymbal drums each get a **dedicated** :class:`pygame.mixer.Channel` so
that a new hit chokes (stops) the previous ring.  Non-cymbal drums share
the mixer's default multi-channel pool.
"""
from __future__ import annotations

import logging
import threading
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .. import config

log = logging.getLogger(__name__)


class AudioEngine:
    """pygame.mixer wrapper that plays drum samples with velocity layers.

    Parameters
    ----------
    pack_name:
        Sub-directory name inside :data:`config.PACKS_DIR` to load on startup.
    """

    def __init__(self, pack_name: str = "default") -> None:
        """Initialise pygame mixer and load the requested sample pack."""
        # _samples[drum_name][band_idx] -> list[pygame.mixer.Sound]
        self._samples: Dict[str, Dict[int, List]] = defaultdict(lambda: defaultdict(list))
        # round-robin counters keyed by (drum_name, band_idx)
        self._rr: Dict[Tuple[str, int], int] = defaultdict(int)
        # dedicated pygame.Channel objects for cymbal drums
        self._cymbal_channels: Dict[str, object] = {}
        # per-drum volume trim multipliers
        self._trims: Dict[str, float] = defaultdict(lambda: 1.0)
        # set of drum names that are cymbals (populated by set_cymbal_lines)
        self._cymbal_names: set = set()
        self._lock = threading.Lock()
        self._pygame = None
        self._initialised = False

        self._init_mixer()
        if self._initialised:
            self.load_pack(pack_name)

    # ------------------------------------------------------------------
    # Internal setup
    # ------------------------------------------------------------------

    def _init_mixer(self) -> None:
        """Initialise pygame.mixer with project-wide audio settings."""
        try:
            import pygame  # type: ignore
            pygame.mixer.init(
                frequency=config.AUDIO_SAMPLE_RATE,
                size=-config.AUDIO_BIT_DEPTH,
                channels=config.AUDIO_CHANNELS,
                buffer=config.AUDIO_BUFFER,
            )
            pygame.mixer.set_num_channels(32)
            self._pygame = pygame
            self._initialised = True
            log.info(
                "pygame.mixer initialised: %d Hz, %d-bit, %d ch, buffer %d",
                config.AUDIO_SAMPLE_RATE,
                config.AUDIO_BIT_DEPTH,
                config.AUDIO_CHANNELS,
                config.AUDIO_BUFFER,
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("pygame mixer unavailable (%s); audio disabled", exc)
            self._pygame = None
            self._initialised = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_pack(self, pack_name: str) -> None:
        """Scan *packs/<pack_name>/* and load all matching WAV files.

        For each drum name the loader tries, in order:

        1. Primary band suffix  (``_ghost``, ``_soft``, ``_medium``,
           ``_hard``, ``_accent``)
        2. Legacy alt-suffix    (``_pp``, ``_mp``, ``_mf``, ``_f``, ``_ff``)
        3. No suffix at all     (bare ``<drum>.wav``)

        The result is stored in ``self._samples[drum_name][band_idx]`` as a
        list of :class:`pygame.mixer.Sound` objects ready for round-robin
        playback.

        Parameters
        ----------
        pack_name:
            Sub-directory of :data:`config.PACKS_DIR` to load.
        """
        if not self._initialised:
            return

        pack_path: Path = config.PACKS_DIR / pack_name
        if not pack_path.exists():
            log.warning("Sample pack '%s' not found at %s", pack_name, pack_path)
            return

        # Build lookup tables for suffix -> band index
        primary_to_idx: Dict[str, int] = {
            band.suffix.lstrip("_"): i
            for i, band in enumerate(config.VELOCITY_BANDS)
        }
        alt_to_idx: Dict[str, int] = {
            band.alt_suffix.lstrip("_"): i
            for i, band in enumerate(config.VELOCITY_BANDS)
        }

        # Clear previous pack data
        with self._lock:
            self._samples.clear()
            self._rr.clear()

        loaded = 0
        for wav in sorted(pack_path.glob("*.wav")):
            stem = wav.stem  # e.g. "snare_medium" or "snare_mf" or "snare"

            # Determine drum name and band index from file stem
            drum_name, band_idx = self._parse_stem(stem, primary_to_idx, alt_to_idx)

            try:
                sound = self._pygame.mixer.Sound(str(wav))
                with self._lock:
                    self._samples[drum_name][band_idx].append(sound)
                loaded += 1
            except Exception as exc:  # noqa: BLE001
                log.debug("Failed to load %s: %s", wav, exc)

        log.info("Loaded %d samples from pack '%s'", loaded, pack_name)

    @staticmethod
    def _parse_stem(
        stem: str,
        primary_to_idx: Dict[str, int],
        alt_to_idx: Dict[str, int],
    ) -> Tuple[str, int]:
        """Return ``(drum_name, band_idx)`` for a WAV file stem.

        The stem is expected to be ``<drum>[_<suffix>]``.  The suffix is
        matched first against primary band names, then legacy names.  If no
        recognised suffix is found the file is assigned to the "medium" band
        (index 2) as a generic fallback.
        """
        # Default: whole stem is the drum name, band 2 (medium)
        fallback_band = 2  # "medium" / "mf"

        # Try splitting off the last "_<token>" segment
        if "_" in stem:
            parts = stem.rsplit("_", 1)
            suffix_candidate = parts[1]
            if suffix_candidate in primary_to_idx:
                return parts[0], primary_to_idx[suffix_candidate]
            if suffix_candidate in alt_to_idx:
                return parts[0], alt_to_idx[suffix_candidate]

        # No recognised suffix — treat whole stem as drum name, medium band
        return stem, fallback_band

    def set_cymbal_lines(self, drum_lines: list) -> None:
        """Register which drum names are cymbals and pre-allocate channels.

        Each cymbal drum gets its own dedicated :class:`pygame.mixer.Channel`
        so that a new hit chokes (silences) the previous ring immediately.

        Parameters
        ----------
        drum_lines:
            List of :class:`~airdrums.config.DrumLine` objects.  All entries
            with ``is_cymbal=True`` receive a dedicated channel.
        """
        if not self._initialised:
            return

        with self._lock:
            self._cymbal_names.clear()
            self._cymbal_channels.clear()

        for dl in drum_lines:
            is_cymbal = dl.is_cymbal if isinstance(dl, config.DrumLine) else dl.get("is_cymbal", False)
            dl_name = dl.name if isinstance(dl, config.DrumLine) else dl.get("name", "")
            if is_cymbal:
                try:
                    channel = self._pygame.mixer.Channel(
                        # Use a high channel index to avoid colliding with
                        # the normal pool (channels 0-15).
                        16 + len(self._cymbal_channels)
                    )
                    with self._lock:
                        self._cymbal_names.add(dl_name)
                        self._cymbal_channels[dl_name] = channel
                    log.debug(
                        "Dedicated pygame.Channel allocated for cymbal '%s'", dl_name
                    )
                except Exception as exc:  # noqa: BLE001
                    log.warning(
                        "Could not allocate channel for cymbal '%s': %s", dl_name, exc
                    )

    def set_volume_trim(self, drum_name: str, trim: float) -> None:
        """Set a per-drum volume multiplier applied on top of MIDI velocity.

        Parameters
        ----------
        drum_name:
            Name of the drum (must match the key in :attr:`_samples`).
        trim:
            Multiplier in [0.0, 2.0].  Values outside this range are clamped.
        """
        self._trims[drum_name] = max(0.0, min(2.0, float(trim)))
        log.debug("Volume trim for '%s' set to %.2f", drum_name, self._trims[drum_name])

    def play_hit(self, drum_name: str, band_idx: int, midi_velocity: int) -> None:
        """Play the correct velocity-layered sample for one drum hit.

        Cymbals use their dedicated channel (choke effect on re-trigger).
        Non-cymbals use ``pygame.mixer.find_channel(force=True)`` from the
        shared pool.

        If the exact ``band_idx`` has no samples the method falls back to any
        available band for that drum, in ascending band-index order.

        Parameters
        ----------
        drum_name:
            Key used to look up samples (must match filenames minus suffix).
        band_idx:
            Velocity band index 0-4 (ghost … accent).
        midi_velocity:
            Raw MIDI velocity 1-127 used for final volume scaling.
        """
        if not self._initialised:
            return

        with self._lock:
            drum_samples = self._samples.get(drum_name)
            if not drum_samples:
                log.debug("No samples for drum '%s'", drum_name)
                return

            # Find the best available band
            options = drum_samples.get(band_idx)
            if not options:
                for fallback_idx in sorted(drum_samples.keys()):
                    options = drum_samples[fallback_idx]
                    if options:
                        break

            if not options:
                log.debug("No samples for drum '%s' band %d", drum_name, band_idx)
                return

            # Round-robin selection
            rr_key = (drum_name, band_idx)
            idx = self._rr[rr_key] % len(options)
            self._rr[rr_key] += 1
            sound = options[idx]
            is_cymbal = drum_name in self._cymbal_names
            cymbal_channel = self._cymbal_channels.get(drum_name)
            trim = self._trims[drum_name]

        volume = max(0.0, min(1.0, (midi_velocity / 127.0) * trim))

        try:
            sound.set_volume(volume)
            if is_cymbal and cymbal_channel is not None:
                cymbal_channel.stop()
                cymbal_channel.play(sound)
            else:
                channel = self._pygame.mixer.find_channel(True)  # force=True
                if channel is not None:
                    channel.play(sound)
        except Exception as exc:  # noqa: BLE001
            log.debug("play_hit failed for '%s': %s", drum_name, exc)

    def close(self) -> None:
        """Stop all playback and release pygame.mixer resources."""
        if self._initialised and self._pygame is not None:
            try:
                self._pygame.mixer.quit()
                log.info("pygame.mixer closed")
            except Exception:  # noqa: BLE001
                pass
        self._initialised = False
