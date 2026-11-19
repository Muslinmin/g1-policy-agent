"""Checks between the policy and the robot. Every target goes through ``Safety.filter``."""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)


class SafetyError(RuntimeError):
    pass


class Safety:
    """Joint limits, a per-tick step limit, and a first-action guard.

    * limits: targets are clipped to [lo + margin, hi - margin].
    * step: a target more than ``max_step`` rad from the previous target is moved only ``max_step`` toward
      it (the robot driver additionally rate-limits at its own publish rate).
    * first action: if the policy's first target is more than ``first_jump`` rad from the measured posture on
      any joint, the run is refused (``SafetyError``) unless ``allow_far_start``; then the step limit walks
      the arm there slowly.
    * non-finite targets raise ``SafetyError``.
    """

    def __init__(self, lo: np.ndarray, hi: np.ndarray, margin: float = 0.02, max_step: float = 0.05,
                 first_jump: float = 0.35, allow_far_start: bool = False) -> None:
        self.lo, self.hi = lo + margin, hi - margin
        self.max_step, self.first_jump, self.allow_far_start = max_step, first_jump, allow_far_start
        self.prev: Optional[np.ndarray] = None
        self.clipped = 0

    def reset(self, q_measured: np.ndarray) -> None:
        self.prev = np.asarray(q_measured, np.float32).copy()
        self._first = True

    def filter(self, target: np.ndarray, names: list[str]) -> np.ndarray:
        t = np.asarray(target, np.float32)
        if t.shape != self.lo.shape:
            raise SafetyError(f"policy action has {t.shape[0]} values, robot has {self.lo.shape[0]} joints")
        if not np.all(np.isfinite(t)):
            raise SafetyError("policy returned a non-finite action")
        if getattr(self, "_first", False):
            self._first = False
            jump = np.abs(t - self.prev)
            worst = int(np.argmax(jump))
            if jump[worst] > self.first_jump and not self.allow_far_start:
                raise SafetyError(
                    f"first action is {jump[worst]:.2f} rad from the measured posture on {names[worst]} "
                    f"(limit {self.first_jump}); start the robot nearer the training start pose or pass "
                    "--allow-far-start (it then moves there at the step limit)")
        c = np.clip(t, self.lo, self.hi)
        if np.any(c != t):
            self.clipped += 1
        step = np.clip(c - self.prev, -self.max_step, self.max_step)
        self.prev = self.prev + step
        return self.prev.copy()
