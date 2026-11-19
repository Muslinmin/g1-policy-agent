"""A G1-shaped robot with no hardware: first-order joint tracking and synthetic camera frames.

Used by the tests and by ``g1-agent run --robot mock`` to check a server, the chunking choice and the
timing before anything moves.
"""

from __future__ import annotations

import time

import numpy as np

from . import g1_layout
from .base import Observation, Robot


class MockG1(Robot):
    def __init__(self, with_waist: bool = False, with_hands: bool = True,
                 image_names: tuple[str, ...] = ("head",), image_hw: tuple[int, int] = (480, 640),
                 tau_s: float = 0.05, hz: float = 30.0) -> None:
        self.joint_names = g1_layout.joint_names(with_waist, with_hands)
        self.lo, self.hi = g1_layout.limits_for(self.joint_names)
        self.q = np.clip(np.zeros(len(self.joint_names), np.float32), self.lo, self.hi)
        self.target = self.q.copy()
        self.image_names, self.image_hw = image_names, image_hw
        self.alpha = 1.0 - np.exp(-1.0 / (hz * tau_s))
        self.commands = 0
        self.holds = 0

    def connect(self) -> None:
        pass

    def observe(self) -> Observation:
        self.q = self.q + self.alpha * (self.target - self.q)
        h, w = self.image_hw
        imgs = {}
        for k, name in enumerate(self.image_names):
            img = np.zeros((h, w, 3), np.uint8)
            x = int((0.5 + 0.4 * np.tanh(self.q[0] + k)) * (w - 1))
            img[:, max(0, x - 4):x + 4, :] = 255
            imgs[name] = img
        return Observation(self.q.copy(), imgs, time.time())

    def command(self, q_target: np.ndarray) -> None:
        self.target = np.clip(np.asarray(q_target, np.float32), self.lo, self.hi)
        self.commands += 1

    def hold(self) -> None:
        self.holds += 1

    def close(self) -> None:
        pass
