"""
airdrums.tracking.depth
=======================
Threaded wrapper around Depth Anything V2.  Runs inference on a daemon thread
reading from a bounded queue so the main OpenCV loop is never blocked.  If the
model cannot be loaded (no network, no torch, etc.) the engine silently falls
back to a no-op and exposes ``latest_depth == None``.

The D-key toggle (enable/disable) is handled externally; this module simply
produces depth frames whenever the engine is running.
"""
from __future__ import annotations

import logging
import queue
import threading
from typing import Optional

import numpy as np

from .. import config

log = logging.getLogger(__name__)


class DepthEngine:
    """Background Depth Anything V2 inference engine.

    Producer (main thread) pushes BGR frames via :meth:`submit_frame`.
    Consumer (worker thread) runs the model and writes the result to
    :attr:`latest_depth`.

    The queue holds at most ``DEPTH_QUEUE_SIZE`` frames so the worker always
    processes the freshest frame instead of falling behind.

    Args:
        profile: :class:`config.HardwareProfile` that selects the model.
        enabled: Set to ``False`` to disable depth estimation entirely.
    """

    def __init__(self, profile: config.HardwareProfile, enabled: bool = True) -> None:
        self.profile = profile
        self.enabled = enabled and getattr(profile, "depth_enabled", True)

        self._queue: "queue.Queue[np.ndarray]" = queue.Queue(
            maxsize=config.DEPTH_QUEUE_SIZE
        )
        self._latest: Optional[np.ndarray] = None
        self._latest_lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._pipe = None
        self._device = "cpu"

        if self.enabled:
            self._load_model()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def start(self) -> None:
        """Start the background inference thread.

        Safe to call multiple times; subsequent calls are no-ops if the
        thread is already running.
        """
        if not self.enabled or self._pipe is None:
            log.info("DepthEngine disabled or model unavailable — skipping start()")
            return
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._worker, name="DepthEngine", daemon=True
        )
        self._thread.start()
        log.info("DepthEngine started (model=%s, device=%s)", self.profile.depth_model, self._device)

    def stop(self) -> None:
        """Signal the worker thread to stop and wait for it to finish."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        log.info("DepthEngine stopped")

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------
    def _load_model(self) -> None:
        """Load the Depth Anything V2 model via the HuggingFace pipeline API.

        Selects the best available device (CUDA > MPS > CPU).  On failure the
        engine disables itself and logs a warning.
        """
        try:
            import torch  # type: ignore
            from transformers import pipeline  # type: ignore

            if torch.cuda.is_available():
                self._device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self._device = "mps"
            else:
                self._device = "cpu"

            log.info(
                "Loading depth model %s on %s",
                self.profile.depth_model, self._device,
            )
            self._pipe = pipeline(
                task="depth-estimation",
                model=self.profile.depth_model,
                device=self._device,
            )
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "Depth model load failed (%s); running without depth.", exc
            )
            self._pipe = None
            self.enabled = False

    # ------------------------------------------------------------------
    # Producer API
    # ------------------------------------------------------------------
    def submit_frame(self, frame_bgr: np.ndarray) -> None:
        """Push a BGR frame to the inference queue.

        If the queue is full the oldest stale frame is discarded before
        inserting the new one, so the worker always sees the most recent frame.

        Args:
            frame_bgr: ``uint8`` BGR numpy array from the capture device.
        """
        if not self.enabled or self._pipe is None:
            return
        try:
            self._queue.put_nowait(frame_bgr)
        except queue.Full:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._queue.put_nowait(frame_bgr)
            except queue.Full:
                pass

    # Alias kept for callers that used the old name.
    def submit(self, frame_bgr: np.ndarray) -> None:
        """Alias for :meth:`submit_frame` (backwards compatibility)."""
        self.submit_frame(frame_bgr)

    @property
    def latest_depth(self) -> Optional[np.ndarray]:
        """Most recent depth map produced by the inference thread.

        Returns a normalised ``float32`` array in [0, 1] with shape
        ``(DEPTH_INPUT_HEIGHT, DEPTH_INPUT_WIDTH)``, or ``None`` when no
        inference has completed yet.
        """
        with self._latest_lock:
            return self._latest

    # ------------------------------------------------------------------
    # Worker
    # ------------------------------------------------------------------
    def _worker(self) -> None:
        """Background thread: pull frames from the queue and run the model."""
        import cv2  # local import
        from PIL import Image  # type: ignore

        target_size = (config.DEPTH_INPUT_WIDTH, config.DEPTH_INPUT_HEIGHT)
        log.info("Depth worker started (target_size=%s)", target_size)

        while not self._stop.is_set():
            try:
                frame = self._queue.get(timeout=0.25)
            except queue.Empty:
                continue
            try:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil = Image.fromarray(rgb).resize(target_size)
                result = self._pipe(pil)  # type: ignore[misc]
                depth = np.asarray(result["depth"], dtype=np.float32)
                # Normalise to [0, 1] for stable downstream thresholds
                dmin, dmax = float(depth.min()), float(depth.max())
                if dmax > dmin:
                    depth = (depth - dmin) / (dmax - dmin)
                with self._latest_lock:
                    self._latest = depth
            except Exception as exc:  # noqa: BLE001
                log.debug("Depth inference failed: %s", exc)
