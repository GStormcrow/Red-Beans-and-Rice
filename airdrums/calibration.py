"""
airdrums.calibration
====================
Four-stage calibration sequence run at first launch (skippable if a saved
profile exists).

Stage 1 - Depth scale: player stands 150cm from camera with arms out; we
          measure wrist depth and compute ``depth_scale = 150 / wrist_depth``.
Stage 2 - Strike threshold: player strikes each zone 3 times with both
          drumsticks. 30th-percentile tip speed sets ``spike_threshold``.
Stage 3 - Pedal threshold: player stomps right foot 5 times; heel velocity
          threshold is auto-set for FootPedalDetector.
Stage 4 - Zone fit: player holds stick tips at zone boundaries to confirm
          the layout fits their reach.

Calibration data is persisted to ``~/.airdrums/profiles/default.json`` and
used to seed runtime thresholds.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np

from . import config

log = logging.getLogger(__name__)


class Calibrator:
    """Runs the calibration sequence.

    Callers supply a ``poll_fn`` that returns the current skeleton joints +
    drumstick tips each frame and a ``draw_fn`` that renders status on the
    OpenCV window. The calibrator returns a dict with the computed values.
    """

    def __init__(self, poll_fn: Callable[[], Dict],
                 prompt_fn: Callable[[str], None]):
        self.poll = poll_fn
        self.prompt = prompt_fn
        self.result: Dict[str, float] = {}

    # ------------------------------------------------------------------
    def run(self) -> Dict[str, float]:
        log.info("Starting calibration")
        self._stage1_depth_scale()
        self._stage2_strike_threshold()
        self._stage3_pedal_threshold()
        self._stage4_zone_fit()
        self._save()
        return self.result

    # ------------------------------------------------------------------
    def _stage1_depth_scale(self) -> None:
        self.prompt("Stage 1/4: Stand 150cm from camera with arms out. (5s)")
        samples: List[float] = []
        t_end = time.time() + 5
        while time.time() < t_end:
            s = self.poll()
            lw = s.get("joints", {}).get("left_wrist")
            rw = s.get("joints", {}).get("right_wrist")
            for j in (lw, rw):
                if j is not None and getattr(j, "visible", False) and j.z_depth > 0:
                    samples.append(j.z_depth)
            time.sleep(0.03)
        if samples:
            measured = float(np.median(samples))
            depth_scale = 150.0 / max(measured, 1e-3)
            self.result["depth_scale"] = depth_scale
            log.info("Depth scale = %.3f (measured=%.3f)", depth_scale, measured)
        else:
            self.result["depth_scale"] = config.DEPTH_DEFAULT_SCALE

    # ------------------------------------------------------------------
    def _stage2_strike_threshold(self) -> None:
        self.prompt("Stage 2/4: Strike each zone 3 times with both sticks (15s)")
        speeds: List[float] = []
        t_end = time.time() + 15
        while time.time() < t_end:
            s = self.poll()
            for tip in s.get("tips", []):
                if tip.visibility > 0.5 and tip.speed > 0.3:
                    speeds.append(float(tip.speed))
            time.sleep(0.02)
        if speeds:
            thresh = float(np.percentile(speeds, 30))
            self.result["spike_threshold"] = max(0.3, thresh)
            log.info("spike_threshold = %.3f (n=%d)", thresh, len(speeds))
        else:
            self.result["spike_threshold"] = config.STRIKE_SPIKE_THRESHOLD

    # ------------------------------------------------------------------
    def _stage3_pedal_threshold(self) -> None:
        self.prompt("Stage 3/4: Stomp right foot 5 times (10s)")
        speeds: List[float] = []
        t_end = time.time() + 10
        while time.time() < t_end:
            s = self.poll()
            heel = s.get("joints", {}).get("right_heel")
            if heel is not None and getattr(heel, "visible", False) and heel.vy > 0.2:
                speeds.append(float(heel.vy))
            time.sleep(0.02)
        if speeds:
            thresh = float(np.percentile(speeds, 30))
            self.result["pedal_threshold"] = max(0.4, thresh)
            log.info("pedal_threshold = %.3f", thresh)
        else:
            self.result["pedal_threshold"] = config.PEDAL_HEEL_THRESHOLD

    # ------------------------------------------------------------------
    def _stage4_zone_fit(self) -> None:
        self.prompt("Stage 4/4: Hold stick tips at zone boundaries (5s)")
        reach: List[float] = []
        t_end = time.time() + 5
        while time.time() < t_end:
            s = self.poll()
            for tip in s.get("tips", []):
                if tip.visibility > 0.5:
                    reach.append(abs(tip.x - 0.5))
            time.sleep(0.05)
        if reach:
            self.result["reach"] = float(np.percentile(reach, 95))
            log.info("reach radius = %.3f", self.result["reach"])
        else:
            self.result["reach"] = 0.45

    # ------------------------------------------------------------------
    def _save(self) -> None:
        config.DEFAULT_PROFILE_PATH.write_text(json.dumps(self.result, indent=2))
        log.info("Saved calibration to %s", config.DEFAULT_PROFILE_PATH)


def load_profile(path: Optional[Path] = None) -> Dict[str, float]:
    path = path or config.DEFAULT_PROFILE_PATH
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception as exc:  # noqa: BLE001
        log.warning("Could not load profile %s: %s", path, exc)
        return {}
