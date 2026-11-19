"""Cameras on the robot computer. Each runs a background thread and hands out the latest RGB frame.

RealSense (e.g. the G1 head D435i) needs ``pyrealsense2``; any V4L2 / USB camera works with OpenCV.
Spec strings for the CLI: ``name=realsense:<serial>[:WxH@FPS]`` or ``name=opencv:<device>[:WxH@FPS]``.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)


class Camera:
    def __init__(self, name: str) -> None:
        self.name = name
        self._frame: Optional[np.ndarray] = None
        self._t = 0.0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        self._open()
        self._thread = threading.Thread(target=self._loop, name=f"cam-{self.name}", daemon=True)
        self._thread.start()
        t = time.time()
        while self._frame is None:
            if time.time() - t > 5.0:
                raise RuntimeError(f"camera '{self.name}' gave no frame within 5 s")
            time.sleep(0.01)

    def latest(self, max_age_s: float = 0.5) -> np.ndarray:
        with self._lock:
            frame, t = self._frame, self._t
        if frame is None or time.time() - t > max_age_s:
            raise RuntimeError(f"camera '{self.name}' is stale ({time.time() - t:.2f} s)")
        return frame

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                f = self._read()
            except Exception as e:  # noqa: BLE001
                log.warning("camera %s: %s", self.name, e)
                time.sleep(0.05)
                continue
            if f is not None:
                with self._lock:
                    self._frame, self._t = f, time.time()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1.0)
        self._close()

    def _open(self) -> None: ...
    def _read(self) -> Optional[np.ndarray]: ...
    def _close(self) -> None: ...


class RealSenseCamera(Camera):
    def __init__(self, name: str, serial: str, w: int = 640, h: int = 480, fps: int = 30) -> None:
        super().__init__(name)
        self.serial, self.w, self.h, self.fps = serial, w, h, fps

    def _open(self) -> None:
        import pyrealsense2 as rs
        self.pipe, cfg = rs.pipeline(), rs.config()
        if self.serial:
            cfg.enable_device(self.serial)
        cfg.enable_stream(rs.stream.color, self.w, self.h, rs.format.rgb8, self.fps)
        self.pipe.start(cfg)

    def _read(self) -> Optional[np.ndarray]:
        frames = self.pipe.wait_for_frames(1000)
        c = frames.get_color_frame()
        return None if not c else np.asanyarray(c.get_data()).copy()

    def _close(self) -> None:
        try:
            self.pipe.stop()
        except Exception:  # noqa: BLE001
            pass


class OpenCVCamera(Camera):
    def __init__(self, name: str, device: str, w: int = 640, h: int = 480, fps: int = 30) -> None:
        super().__init__(name)
        self.device, self.w, self.h, self.fps = device, w, h, fps

    def _open(self) -> None:
        import cv2
        self.cv2 = cv2
        dev = int(self.device) if self.device.isdigit() else self.device
        self.cap = cv2.VideoCapture(dev)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.w)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.h)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        if not self.cap.isOpened():
            raise RuntimeError(f"cannot open camera {self.device}")

    def _read(self) -> Optional[np.ndarray]:
        ok, bgr = self.cap.read()
        return self.cv2.cvtColor(bgr, self.cv2.COLOR_BGR2RGB) if ok else None

    def _close(self) -> None:
        self.cap.release()


def parse_camera(spec: str) -> Camera:
    """'head=realsense:123456:640x480@30' or 'wrist=opencv:/dev/video4'."""
    name, _, rest = spec.partition("=")
    parts = rest.split(":")
    kind, dev = parts[0], (parts[1] if len(parts) > 1 else "")
    w, h, fps = 640, 480, 30
    if len(parts) > 2:
        size, _, f = parts[2].partition("@")
        w, h = (int(x) for x in size.split("x"))
        fps = int(f or fps)
    if kind == "realsense":
        return RealSenseCamera(name, dev, w, h, fps)
    if kind == "opencv":
        return OpenCVCamera(name, dev or "0", w, h, fps)
    raise ValueError(f"camera '{spec}': kind must be realsense or opencv")
