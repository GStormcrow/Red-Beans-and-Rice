"""
airdrums.analytics.stats
========================
Live and post-session analytics for AirDrums V2.

Live analytics
--------------
* BPM via rolling median of inter-hit intervals on snare hits.
* BPM stability as rolling standard deviation over the last
  ``BPM_STABILITY_BARS`` bars.
* Drift indicator: ``"rushing"`` / ``"dragging"`` / ``"steady"`` derived from
  the linear trend of recent inter-hit intervals.

Post-session analytics
----------------------
* Per-drum hit heatmap — 2D scatter of approximate x positions, grouped by
  drum name and mapped to DrumLine.x_center.
* Velocity histogram per drum — count of events per velocity_band.
* Timing accuracy — per-hit offset from the nearest BPM grid position.
* Hand balance — left vs right hit count and per-hand average velocity.
* Session summary — total hits, duration, hits per minute, most/least used.
* PDF report via reportlab with all charts.
"""
from __future__ import annotations

import logging
import statistics
from collections import defaultdict, deque
from pathlib import Path
from typing import Deque, Dict, List, Optional, Tuple

from airdrums import config
from airdrums.config import DrumLine
from airdrums.recording.session import HitEvent

log = logging.getLogger(__name__)


class SessionAnalytics:
    """Computes live and post-session analytics.

    Parameters
    ----------
    drum_lines:
        The kit layout, used to map drum names to approximate x positions for
        heatmap generation and for grouping in charts.
    """

    def __init__(self, drum_lines: List[DrumLine]) -> None:
        """Initialise the analytics engine."""
        self.drum_lines: List[DrumLine] = drum_lines

        # Map drum name -> x_center for heatmap fallback
        self._x_center_map: Dict[str, float] = {
            dl.name: dl.x_center for dl in drum_lines
        }

        # Live BPM — rolling window of inter-hit BPM estimates from snare
        self._snare_ibi_window: Deque[float] = deque(
            maxlen=config.BPM_SMOOTHING_WINDOW
        )
        self._last_snare_ms: Optional[float] = None

        # BPM stability — accumulate estimates per bar
        self._bpm_per_bar: List[List[float]] = [[]]

        # Linear-trend drift — recent inter-hit intervals in ms
        self._recent_ihi_ms: Deque[float] = deque(maxlen=16)
        self._last_hit_ms: Optional[float] = None

    # ------------------------------------------------------------------
    # Live — call on every hit event
    # ------------------------------------------------------------------

    def on_hit(self, event: HitEvent) -> None:
        """Update live analytics on each new HitEvent.

        Should be called immediately after the event is recorded.  Updates
        the BPM estimate, stability window, and drift indicator.

        Parameters
        ----------
        event:
            The newly recorded drum hit.
        """
        # Drift: track inter-hit intervals regardless of drum
        if self._last_hit_ms is not None:
            dt = event.timestamp_ms - self._last_hit_ms
            if 20 <= dt <= 5000:
                self._recent_ihi_ms.append(dt)
        self._last_hit_ms = event.timestamp_ms

        # BPM from snare hits only
        if event.drum_name == "snare":
            if self._last_snare_ms is not None:
                dt_ms = event.timestamp_ms - self._last_snare_ms
                if 80 <= dt_ms <= 2000:
                    bpm = 60000.0 / dt_ms
                    # A single snare hit per beat implies the interval is one
                    # beat; double if too slow (half-time feel heuristic).
                    if bpm < 60:
                        bpm = min(bpm * 2, 240)
                    self._snare_ibi_window.append(bpm)
                    self._bpm_per_bar[-1].append(bpm)
            self._last_snare_ms = event.timestamp_ms

    def tick_bar(self) -> None:
        """Advance the per-bar BPM accumulator.

        Call this on each detected downbeat.  The stability window retains
        only the last :data:`~airdrums.config.BPM_STABILITY_BARS` bars.
        """
        self._bpm_per_bar.append([])
        # Trim to BPM_STABILITY_BARS + 1 (current bar being filled)
        if len(self._bpm_per_bar) > config.BPM_STABILITY_BARS + 1:
            self._bpm_per_bar.pop(0)

    # ------------------------------------------------------------------
    # Live properties
    # ------------------------------------------------------------------

    @property
    def bpm(self) -> float:
        """Current BPM estimate as a rolling median over snare inter-hit intervals.

        Returns 120.0 if no snare hits have been recorded yet.
        """
        if not self._snare_ibi_window:
            return 120.0
        return float(statistics.median(self._snare_ibi_window))

    @property
    def bpm_stability(self) -> float:
        """Rolling std dev of BPM estimates over the last stability window.

        A lower value indicates tighter tempo.  Returns 0.0 when there is
        insufficient data.
        """
        # Flatten all bars in the stability window (excluding current)
        all_vals: List[float] = []
        for bar in self._bpm_per_bar[:-1]:
            all_vals.extend(bar)
        # Include current bar too for a richer estimate
        all_vals.extend(self._bpm_per_bar[-1])

        if len(all_vals) < 2:
            return 0.0
        return float(statistics.pstdev(all_vals))

    @property
    def drift(self) -> str:
        """Tempo drift indicator based on linear trend of recent inter-hit intervals.

        Returns
        -------
        str
            ``"rushing"`` if the player is speeding up,
            ``"dragging"`` if slowing down, or
            ``"steady"`` if within ±5 % of the mean interval.
        """
        ihi = list(self._recent_ihi_ms)
        if len(ihi) < 4:
            return "steady"

        # Simple linear regression slope (ms/hit)
        n = len(ihi)
        xs = list(range(n))
        mean_x = (n - 1) / 2.0
        mean_y = statistics.mean(ihi)
        num = sum((xs[i] - mean_x) * (ihi[i] - mean_y) for i in range(n))
        den = sum((xs[i] - mean_x) ** 2 for i in range(n))
        slope = num / den if den != 0 else 0.0

        # Positive slope → intervals growing → player dragging
        # Negative slope → intervals shrinking → player rushing
        threshold = mean_y * 0.05  # 5 % of mean interval
        if slope > threshold:
            return "dragging"
        if slope < -threshold:
            return "rushing"
        return "steady"

    # ------------------------------------------------------------------
    # Post-session — heatmap
    # ------------------------------------------------------------------

    def hit_heatmap(
        self,
        events: List[HitEvent],
        output_dir: "str | Path",
    ) -> List[Path]:
        """Generate a 2-D scatter heatmap per drum line and save as PNG files.

        Because :class:`~airdrums.recording.session.HitEvent` does not store
        the raw tip position, the drum's ``x_center`` from the kit layout is
        used as the approximate horizontal position.  A small random jitter is
        applied so that the density plot is readable.  The y axis represents
        velocity (0–127).

        Parameters
        ----------
        events:
            Full list of :class:`~airdrums.recording.session.HitEvent` from
            :meth:`~airdrums.recording.session.Session.get_active_events`.
        output_dir:
            Directory where PNGs will be written.

        Returns
        -------
        list[Path]
            Paths to the generated PNG files.
        """
        import random

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Group events by drum name
        per_drum: Dict[str, List[HitEvent]] = defaultdict(list)
        for ev in events:
            per_drum[ev.drum_name].append(ev)

        paths: List[Path] = []
        rng = random.Random(0)

        for drum_name, drum_events in per_drum.items():
            x_center = self._x_center_map.get(drum_name, 0.5)
            xs = [x_center + rng.uniform(-0.04, 0.04) for _ in drum_events]
            ys = [ev.velocity for ev in drum_events]

            label = next(
                (dl.label for dl in self.drum_lines if dl.name == drum_name),
                drum_name,
            )
            fig, ax = plt.subplots(figsize=(4, 4))
            ax.scatter(xs, ys, alpha=0.5, s=20, c="#ff7a00")
            ax.set_xlim(0.0, 1.0)
            ax.set_ylim(0, 127)
            ax.set_title(f"Hit scatter — {label}")
            ax.set_xlabel("x position (approx.)")
            ax.set_ylabel("velocity (MIDI)")

            fn = output_dir / f"heatmap_{drum_name.replace(' ', '_').lower()}.png"
            fig.tight_layout()
            fig.savefig(fn, dpi=120)
            plt.close(fig)
            paths.append(fn)

        log.info("Wrote %d heatmap PNGs to %s.", len(paths), output_dir)
        return paths

    # ------------------------------------------------------------------
    # Post-session — velocity histogram per velocity_band
    # ------------------------------------------------------------------

    def velocity_histogram(
        self,
        events: List[HitEvent],
        output_dir: "str | Path",
    ) -> List[Path]:
        """Generate a velocity-band bar chart per drum and save as PNG files.

        Bars represent hit counts per velocity band name
        (``"ghost"``, ``"soft"``, ``"medium"``, ``"hard"``, ``"accent"``).

        Parameters
        ----------
        events:
            Full hit-event list.
        output_dir:
            Directory where PNGs will be written.

        Returns
        -------
        list[Path]
            Paths to the generated PNG files.
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        band_order = ["ghost", "soft", "medium", "hard", "accent"]

        per_drum: Dict[str, Dict[str, int]] = defaultdict(
            lambda: {b: 0 for b in band_order}
        )
        for ev in events:
            per_drum[ev.drum_name][ev.velocity_band] = (
                per_drum[ev.drum_name].get(ev.velocity_band, 0) + 1
            )

        paths: List[Path] = []
        for drum_name, band_counts in per_drum.items():
            counts = [band_counts.get(b, 0) for b in band_order]
            label = next(
                (dl.label for dl in self.drum_lines if dl.name == drum_name),
                drum_name,
            )
            fig, ax = plt.subplots(figsize=(5, 3))
            ax.bar(band_order, counts, color="#4a90d9", edgecolor="black")
            ax.set_title(f"Velocity bands — {label}")
            ax.set_xlabel("velocity band")
            ax.set_ylabel("hits")

            fn = output_dir / f"velocity_{drum_name.replace(' ', '_').lower()}.png"
            fig.tight_layout()
            fig.savefig(fn, dpi=120)
            plt.close(fig)
            paths.append(fn)

        log.info(
            "Wrote %d velocity-band histograms to %s.", len(paths), output_dir
        )
        return paths

    # ------------------------------------------------------------------
    # Post-session — timing accuracy
    # ------------------------------------------------------------------

    def timing_accuracy(
        self,
        events: List[HitEvent],
        bpm: Optional[float] = None,
        grid: str = config.DEFAULT_QUANTIZATION,
    ) -> Dict[str, Dict[str, float]]:
        """Compute per-drum timing offset from the nearest BPM grid position.

        Parameters
        ----------
        events:
            Full hit-event list.
        bpm:
            Tempo to use for grid calculation.  Falls back to
            :attr:`self.bpm` if not provided.
        grid:
            Quantization grid string, e.g. ``"1/16"``.

        Returns
        -------
        dict
            Mapping of drum name -> stats dict with keys
            ``mean_offset_ms``, ``abs_mean_offset_ms``, ``stdev_ms``,
            and ``tendency`` (``"rushing"`` | ``"dragging"`` | ``"locked"``).
        """
        effective_bpm = bpm if bpm is not None else self.bpm
        denom = {"1/8": 2, "1/16": 4, "1/32": 8}.get(grid, 4)
        step_ms = (60000.0 / effective_bpm) / denom

        per_drum: Dict[str, List[float]] = defaultdict(list)
        for ev in events:
            offset = ev.timestamp_ms - round(ev.timestamp_ms / step_ms) * step_ms
            per_drum[ev.drum_name].append(offset)

        report: Dict[str, Dict[str, float]] = {}
        for drum, offsets in per_drum.items():
            if not offsets:
                continue
            mean_off = statistics.mean(offsets)
            abs_mean = statistics.mean(abs(o) for o in offsets)
            stdev = statistics.pstdev(offsets) if len(offsets) > 1 else 0.0
            tendency = (
                "rushing" if mean_off < -5
                else "dragging" if mean_off > 5
                else "locked"
            )
            report[drum] = {
                "mean_offset_ms": round(mean_off, 3),
                "abs_mean_offset_ms": round(abs_mean, 3),
                "stdev_ms": round(stdev, 3),
                "tendency": tendency,
            }
        return report

    # ------------------------------------------------------------------
    # Post-session — hand balance
    # ------------------------------------------------------------------

    def hand_balance(self, events: List[HitEvent]) -> Dict[str, object]:
        """Compute left/right hit count and per-hand average velocity.

        Parameters
        ----------
        events:
            Full hit-event list.

        Returns
        -------
        dict
            Keys: ``left_count``, ``right_count``, ``left_avg_velocity``,
            ``right_avg_velocity``.
        """
        left_vels: List[int] = []
        right_vels: List[int] = []
        for ev in events:
            if ev.hand_side == "left":
                left_vels.append(ev.velocity)
            elif ev.hand_side == "right":
                right_vels.append(ev.velocity)

        return {
            "left_count": len(left_vels),
            "right_count": len(right_vels),
            "left_avg_velocity": (
                round(statistics.mean(left_vels), 2) if left_vels else 0.0
            ),
            "right_avg_velocity": (
                round(statistics.mean(right_vels), 2) if right_vels else 0.0
            ),
        }

    # ------------------------------------------------------------------
    # Post-session — session summary
    # ------------------------------------------------------------------

    def session_summary(self, events: List[HitEvent]) -> Dict[str, object]:
        """Return a high-level summary of the session.

        Parameters
        ----------
        events:
            Full merged event list (from
            :meth:`~airdrums.recording.session.Session.get_active_events`).

        Returns
        -------
        dict
            Keys: ``total_hits``, ``duration_s``, ``hits_per_minute``,
            ``most_used_drum``, ``least_used_drum``.
        """
        if not events:
            return {
                "total_hits": 0,
                "duration_s": 0.0,
                "hits_per_minute": 0.0,
                "most_used_drum": None,
                "least_used_drum": None,
            }

        duration_s = (
            (events[-1].timestamp_ms - events[0].timestamp_ms) / 1000.0
        )
        counts: Dict[str, int] = defaultdict(int)
        for ev in events:
            counts[ev.drum_name] += 1

        most_used = max(counts, key=lambda k: counts[k])
        least_used = min(counts, key=lambda k: counts[k])
        hpm = len(events) / max(duration_s, 1.0) * 60.0

        return {
            "total_hits": len(events),
            "duration_s": round(duration_s, 2),
            "hits_per_minute": round(hpm, 1),
            "most_used_drum": most_used,
            "least_used_drum": least_used,
        }

    # ------------------------------------------------------------------
    # PDF report
    # ------------------------------------------------------------------

    def export_pdf(
        self,
        events: List[HitEvent],
        path: "str | Path",
        include_png_dir: Optional[Path] = None,
        bpm: Optional[float] = None,
    ) -> Path:
        """Generate a PDF session report using reportlab.

        The report includes the session summary, hand balance, and timing
        accuracy table.  If *include_png_dir* is provided, all ``.png`` files
        in that directory are appended as additional pages.

        Parameters
        ----------
        events:
            Full hit-event list.
        path:
            Destination PDF file path.
        include_png_dir:
            Optional directory of PNG charts to embed.
        bpm:
            Override BPM for timing calculations; defaults to :attr:`self.bpm`.

        Returns
        -------
        Path
            The written PDF file.
        """
        from reportlab.lib.pagesizes import letter  # type: ignore
        from reportlab.lib.units import inch         # type: ignore
        from reportlab.pdfgen import canvas          # type: ignore

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        c = canvas.Canvas(str(path), pagesize=letter)
        w, h = letter

        def _dark_page() -> None:
            c.setFillColorRGB(0.08, 0.08, 0.12)
            c.rect(0, 0, w, h, fill=True, stroke=False)

        # --- Page 1: Summary + hand balance + timing ---
        _dark_page()
        c.setFillColorRGB(1, 0.5, 0.1)
        c.setFont("Helvetica-Bold", 22)
        c.drawString(0.5 * inch, h - 0.7 * inch, "AirDrums V2 — Session Report")

        c.setFillColorRGB(0.9, 0.9, 0.9)
        c.setFont("Helvetica", 12)
        y = h - 1.2 * inch

        summary = self.session_summary(events)
        for k, v in summary.items():
            c.drawString(0.6 * inch, y, f"{k.replace('_', ' ').title()}: {v}")
            y -= 16

        y -= 8
        c.setFillColorRGB(1, 0.5, 0.1)
        c.setFont("Helvetica-Bold", 14)
        c.drawString(0.5 * inch, y, "Hand Balance")
        y -= 18
        c.setFillColorRGB(0.9, 0.9, 0.9)
        c.setFont("Helvetica", 11)
        for k, v in self.hand_balance(events).items():
            c.drawString(0.6 * inch, y, f"{k.replace('_', ' ').title()}: {v}")
            y -= 14

        y -= 8
        c.setFillColorRGB(1, 0.5, 0.1)
        c.setFont("Helvetica-Bold", 14)
        c.drawString(0.5 * inch, y, f"Timing Accuracy ({config.DEFAULT_QUANTIZATION} grid)")
        y -= 18
        c.setFillColorRGB(0.9, 0.9, 0.9)
        c.setFont("Helvetica", 11)
        for drum, rep in self.timing_accuracy(events, bpm=bpm).items():
            label = next(
                (dl.label for dl in self.drum_lines if dl.name == drum),
                drum,
            )
            line = (
                f"{label}: {rep['tendency']}  "
                f"|avg|={rep['abs_mean_offset_ms']:.1f} ms  "
                f"stdev={rep['stdev_ms']:.1f} ms"
            )
            c.drawString(0.6 * inch, y, line)
            y -= 14
            if y < 1.0 * inch:
                c.showPage()
                _dark_page()
                c.setFillColorRGB(0.9, 0.9, 0.9)
                c.setFont("Helvetica", 11)
                y = h - 0.7 * inch

        # --- Additional pages: PNG charts ---
        if include_png_dir is not None:
            png_dir = Path(include_png_dir)
            if png_dir.exists():
                for png in sorted(png_dir.glob("*.png")):
                    c.showPage()
                    _dark_page()
                    c.setFillColorRGB(0.9, 0.9, 0.9)
                    c.setFont("Helvetica-Bold", 14)
                    c.drawString(0.5 * inch, h - 0.7 * inch, png.stem)
                    try:
                        c.drawImage(
                            str(png),
                            0.5 * inch,
                            1.0 * inch,
                            width=w - 1.0 * inch,
                            height=h - 2.0 * inch,
                            preserveAspectRatio=True,
                        )
                    except Exception:  # noqa: BLE001
                        log.debug("Failed to embed %s in PDF.", png, exc_info=True)

        c.save()
        log.info("Wrote PDF report: %s.", path)
        return path
