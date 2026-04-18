"""
airdrums.tracking.depth
=======================
Threaded wrapper around Depth Anything V2. Runs inference on a daemon thread
reading from a bounded queue so the main OpenCV loop is never blocked. If the
model cannot be loaded (no network, no torch, etc.) the engine silently falls
back to a no-op and exposes `latest_depth == None`.
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

    Producer (main thread) pushes BGR frames via ``submit(frame)``. Consumer
    (worker thread) runs the model and writes the result to ``latest_depth``.
    The queue is size-2 so we overwrite-on-full and always give the worker
    the freshest frame instead of letting it fall behind.
    """

    def __init__(self, profile: config.HardwareProfile, enabled: bool = True):
        self.profile = profile
        self.enabled = enabled and profile.depth_enabled
        self._queue: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=config.DEPTH_QUEUE_SIZE)
        self._latest: Optional[np.ndarray] = None
        self._latest_lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.depth_scale: float = config.DEPTH_DEFAULT_SCALE
        self._pipe = None
        self._device = "cpu"

        if self.enabled:
            self._load_model()
            self._thread = threading.Thread(target=self._worker, name="DepthEngine", daemon=True)
            self._thread.start()

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------
    def _load_model(self) -> None:
        try:
            import torch  # type: ignore
            from transformers import pipeline  # type: ignore

            if torch.cuda.is_available():
                self._device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self._device = "mps"
            else:
                self._device = "cpu"

            log.info("Loading depth model %s on %s", self.profile.depth_model, self._device)
            self._pipe = pipeline(
                task="depth-estimation",
                model=self.profile.depth_model,
                device=self._device,
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("Depth model load failed (%s); running without depth.", exc)
            self._pipe = None
            self.enabled = False

    # ------------------------------------------------------------------
    # Producer API
    # ------------------------------------------------------------------
    def submit(self, frame_bgr: np.ndarray) -> None:
        """Push a frame to the worker, dropping stale frames if full."""
        if not self.enabled or self._pipe is None:
            return
        try:
            self._queue.put_nowait(frame_bgr)
        except queue.Full:
            # Drain one stale entry and put the fresh one.
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._queue.put_nowait(frame_bgr)
            except queue.Full:
                pass

    @property
    def latest_depth(self) -> Optional[np.ndarray]:
        with self._latest_lock:
            return self._latest

    # ------------------------------------------------------------------
    # Worker
    # ------------------------------------------------------------------
    def _worker(self) -> None:
        import cv2  # local import
        from PIL import Image  # type: ignore

        log.info("Depth worker started")
        while not self._stop.is_set():
            try:
                frame = self._queue.get(timeout=0.25)
            except queue.Empty:
                continue
            try:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil = Image.fromarray(rgb).resize(config.DEPTH_INPUT_SIZE)
                result = self._pipe(pil)  # type: ignore[misc]
                depth = np.asarray(result["depth"], dtype=np.float32)
                # Normalize to 0..1 for stable downstream thresholds
                dmin, dmax = float(depth.min()), float(depth.max())
                if dmax > dmin:
                    depth = (depth - dmin) / (dmax - dmin)
                with self._latest_lock:
                    self._latest = depth
            except Exception as exc:  # noqa: BLE001
                log.debug("Depth inference failed: %s", exc)

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    # ------------------------------------------------------------------
    # Calibration hook
    # ------------------------------------------------------------------
    def set_depth_scale(self, scale: float) -> None:
        self.depth_scale = float(scale)
        log.info("Depth scale set to %.3f", self.depth_scale)
