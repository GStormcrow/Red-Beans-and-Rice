"""
airdrums.analytics.stats
========================
Live + post-session analytics.

Live:
  * BPM via inter-hit interval rolling median (8 hits).
  * BPM stability (stdev per bar), drift indicator (speeding up / dragging).

Post-session:
  * Per-drum hit heatmap (2D density of tip positions) as PNG.
  * Velocity histogram per drum as PNG.
  * Timing accuracy vs nearest grid position (ms offset).
  * Hand balance (left/right count + velocity ratio + limb-independence score).
  * Session summary (hits, HPM, duration, most/least used drum).
  * Combined PDF session report via reportlab.
"""
from __future__ import annotations

import logging
import statistics
from collections import defaultdict, deque
from pathlib import Path
from typing import Deque, Dict, List, Optional, Tuple

from .. import config

log = logging.getLogger(__name__)


class SessionAnalytics:
    """Computes live + post-session analytics over a :class:`Session`."""

    KICK_SNARE_NOTES = {36, 38, 37}    # used for live BPM inference

    def __init__(self, session):
        self.session = session
        self._ibi_window: Deque[float] = deque(maxlen=config.BPM_ROLLING_WINDOW)
        self._last_kick_snare_ms: Optional[float] = None
        self._bpm_history: List[float] = []
        self._bpm_per_bar: List[List[float]] = [[]]
        self._strike_positions: Dict[str, List[Tuple[float, float]]] = defaultdict(list)

    # ------------------------------------------------------------------
    # Live
    # ------------------------------------------------------------------
    def on_hit(self, event, tip_xy: Optional[Tuple[float, float]] = None) -> None:
        """Call this immediately after each HitEvent is recorded."""
        if tip_xy is not None:
            self._strike_positions[event.drum_name].append(tip_xy)
        if event.midi_note in self.KICK_SNARE_NOTES:
            if self._last_kick_snare_ms is not None:
                dt_ms = event.timestamp_ms - self._last_kick_snare_ms
                if 80 <= dt_ms <= 2000:
                    bpm = 60000.0 / dt_ms
                    # Assume downbeats every 1-2 hits
                    bpm = min(bpm * 2, 240) if bpm < 60 else bpm
                    self._ibi_window.append(bpm)
                    self._bpm_per_bar[-1].append(bpm)
            self._last_kick_snare_ms = event.timestamp_ms

    @property
    def bpm(self) -> float:
        if not self._ibi_window:
            return config.MIDI_DEFAULT_BPM
        return float(statistics.median(self._ibi_window))

    @property
    def bpm_stability(self) -> float:
        """Standard deviation of BPM within the current bar (lower = tighter)."""
        cur = self._bpm_per_bar[-1]
        if len(cur) < 2:
            return 0.0
        return float(statistics.pstdev(cur))

    @property
    def drift(self) -> str:
        """'speeding' | 'dragging' | 'steady'."""
        if len(self._ibi_window) < 4:
            return "steady"
        first_half = list(self._ibi_window)[: len(self._ibi_window) // 2]
        second_half = list(self._ibi_window)[len(self._ibi_window) // 2:]
        d = statistics.mean(second_half) - statistics.mean(first_half)
        if d > 1.5:
            return "speeding"
        if d < -1.5:
            return "dragging"
        return "steady"

    def tick_bar(self) -> None:
        """Advance the per-bar accumulator. Call on each downbeat."""
        self._bpm_per_bar.append([])

    # ------------------------------------------------------------------
    # Post-session - heatmap + histogram
    # ------------------------------------------------------------------
    def export_heatmaps(self, output_dir: str | Path) -> List[Path]:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        paths: List[Path] = []

        for drum, pts in self._strike_positions.items():
            if not pts:
                continue
            xs = np.array([p[0] for p in pts])
            ys = np.array([p[1] for p in pts])
            fig, ax = plt.subplots(figsize=(4, 4))
            ax.hist2d(xs, ys, bins=24, range=[[0, 1], [0, 1]], cmap="inferno")
            ax.invert_yaxis()
            ax.set_title(f"Hit heatmap — {drum}")
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            fn = output_dir / f"heatmap_{drum.replace(' ', '_').lower()}.png"
            fig.tight_layout()
            fig.savefig(fn, dpi=120)
            plt.close(fig)
            paths.append(fn)
        log.info("Wrote %d heatmap PNGs to %s", len(paths), output_dir)
        return paths

    def export_velocity_histograms(self, output_dir: str | Path) -> List[Path]:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        per_drum: Dict[str, List[int]] = defaultdict(list)
        for ev in self.session.events:
            per_drum[ev.drum_name].append(ev.velocity)

        paths: List[Path] = []
        for drum, vels in per_drum.items():
            if not vels:
                continue
            fig, ax = plt.subplots(figsize=(4, 3))
            ax.hist(vels, bins=16, range=(0, 127), color="#ff7a00", edgecolor="black")
            ax.set_xlim(0, 127)
            ax.set_title(f"Velocity distribution — {drum}")
            ax.set_xlabel("MIDI velocity")
            ax.set_ylabel("hits")
            fn = output_dir / f"velocity_{drum.replace(' ', '_').lower()}.png"
            fig.tight_layout()
            fig.savefig(fn, dpi=120)
            plt.close(fig)
            paths.append(fn)
        log.info("Wrote %d velocity histograms to %s", len(paths), output_dir)
        return paths

    # ------------------------------------------------------------------
    # Post-session - timing
    # ------------------------------------------------------------------
    def timing_report(self, bpm: Optional[float] = None, grid: str = "1/16"
                      ) -> Dict[str, Dict[str, float]]:
        bpm = bpm or self.bpm
        denom = {"1/8": 2, "1/16": 4, "1/32": 8}.get(grid, 4)
        step_ms = (60000.0 / bpm) / denom
        per_drum: Dict[str, List[float]] = defaultdict(list)
        for ev in self.session.events:
            offset = ev.timestamp_ms - round(ev.timestamp_ms / step_ms) * step_ms
            per_drum[ev.drum_name].append(offset)

        report: Dict[str, Dict[str, float]] = {}
        for drum, offs in per_drum.items():
            if not offs:
                continue
            report[drum] = {
                "mean_offset_ms": float(statistics.mean(offs)),
                "abs_mean_offset_ms": float(statistics.mean(abs(o) for o in offs)),
                "stdev_ms": float(statistics.pstdev(offs)) if len(offs) > 1 else 0.0,
                "tendency": ("rushing" if statistics.mean(offs) < -5
                             else "dragging" if statistics.mean(offs) > 5
                             else "locked"),
            }
        return report

    # ------------------------------------------------------------------
    # Post-session - hand balance
    # ------------------------------------------------------------------
    def hand_balance(self) -> Dict[str, float]:
        left, right = [], []
        for ev in self.session.events:
            if ev.hand_side == "left":
                left.append(ev.velocity)
            elif ev.hand_side == "right":
                right.append(ev.velocity)
        total = len(left) + len(right)
        if total == 0:
            return {"left_count": 0, "right_count": 0, "independence": 0.0}
        # Limb independence = fraction of near-simultaneous L/R hits
        events = sorted(self.session.events, key=lambda e: e.timestamp_ms)
        simultaneous = 0
        for i in range(len(events) - 1):
            if (events[i].hand_side in ("left", "right") and
                    events[i + 1].hand_side in ("left", "right") and
                    events[i].hand_side != events[i + 1].hand_side and
                    abs(events[i].timestamp_ms - events[i + 1].timestamp_ms) < 40):
                simultaneous += 1
        return {
            "left_count": len(left),
            "right_count": len(right),
            "left_avg_velocity": float(statistics.mean(left)) if left else 0.0,
            "right_avg_velocity": float(statistics.mean(right)) if right else 0.0,
            "independence": simultaneous / max(1, total),
        }

    # ------------------------------------------------------------------
    # Post-session - summary
    # ------------------------------------------------------------------
    def summary(self) -> Dict[str, object]:
        events = self.session.events
        if not events:
            return {"total_hits": 0, "duration_s": 0, "hits_per_minute": 0.0,
                    "most_used_drum": None, "least_used_drum": None}
        duration_s = (events[-1].timestamp_ms - events[0].timestamp_ms) / 1000.0
        counts: Dict[str, int] = defaultdict(int)
        for ev in events:
            counts[ev.drum_name] += 1
        most = max(counts, key=counts.get)
        least = min(counts, key=counts.get)
        return {
            "total_hits": len(events),
            "duration_s": round(duration_s, 2),
            "hits_per_minute": round(len(events) / max(duration_s, 1) * 60, 1),
            "most_used_drum": most,
            "least_used_drum": least,
            "bpm": round(self.bpm, 2),
            "bpm_stability": round(self.bpm_stability, 2),
        }

    # ------------------------------------------------------------------
    # PDF report
    # ------------------------------------------------------------------
    def export_pdf(self, path: str | Path, include_png_dir: Optional[Path] = None
                   ) -> Path:
        from reportlab.lib.pagesizes import letter  # type: ignore
        from reportlab.pdfgen import canvas        # type: ignore
        from reportlab.lib.units import inch       # type: ignore

        path = Path(path)
        c = canvas.Canvas(str(path), pagesize=letter)
        w, h = letter

        c.setFillColorRGB(0.08, 0.08, 0.12)
        c.rect(0, 0, w, h, fill=True, stroke=False)
        c.setFillColorRGB(1, 0.5, 0.1)
        c.setFont("Helvetica-Bold", 22)
        c.drawString(0.5 * inch, h - 0.7 * inch, "AirDrums - Session Report")

        c.setFillColorRGB(0.9, 0.9, 0.9)
        c.setFont("Helvetica", 12)
        y = h - 1.1 * inch
        for k, v in self.summary().items():
            c.drawString(0.6 * inch, y, f"{k.replace('_', ' ').title()}: {v}")
            y -= 16

        y -= 10
        c.setFont("Helvetica-Bold", 14)
        c.drawString(0.5 * inch, y, "Hand balance")
        c.setFont("Helvetica", 11)
        y -= 16
        for k, v in self.hand_balance().items():
            c.drawString(0.6 * inch, y, f"{k}: {v}")
            y -= 14

        y -= 10
        c.setFont("Helvetica-Bold", 14)
        c.drawString(0.5 * inch, y, "Timing accuracy (1/16 grid)")
        c.setFont("Helvetica", 11)
        y -= 16
        for drum, rep in self.timing_report().items():
            c.drawString(0.6 * inch, y,
                         f"{drum}: {rep['tendency']}  "
                         f"|avg|={rep['abs_mean_offset_ms']:.1f} ms  "
                         f"stdev={rep['stdev_ms']:.1f} ms")
            y -= 14
            if y < 1.0 * inch:
                c.showPage()
                y = h - 0.7 * inch

        if include_png_dir is not None and include_png_dir.exists():
            for png in sorted(include_png_dir.glob("*.png")):
                c.showPage()
                c.setFillColorRGB(0.08, 0.08, 0.12)
                c.rect(0, 0, w, h, fill=True, stroke=False)
                c.setFillColorRGB(0.9, 0.9, 0.9)
                c.setFont("Helvetica-Bold", 14)
                c.drawString(0.5 * inch, h - 0.7 * inch, png.stem)
                try:
                    c.drawImage(str(png), 0.5 * inch, 1.0 * inch,
                                width=w - 1.0 * inch, height=h - 2.0 * inch,
                                preserveAspectRatio=True)
                except Exception as exc:  # noqa: BLE001
                    log.debug("Failed to embed %s: %s", png, exc)

        c.save()
        log.info("Wrote PDF report: %s", path)
        return path
