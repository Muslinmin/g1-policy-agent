"""A backend turns a list of observation frames into an action chunk, and describes its model."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from g1agent.spec import PolicySpec


class Backend(ABC):
    @abstractmethod
    def spec(self) -> PolicySpec:
        ...

    @abstractmethod
    def infer(self, frames: list[dict], task: Optional[str]) -> np.ndarray:
        """frames: oldest first, each {"state", "images", "t"[, "extra"]}. Returns (T, D) float32 absolute
        joint targets in the robot's joint order (``spec().action_names``), un-normalised."""

    def reset(self) -> None:
        pass

    def warm_up(self) -> None:
        s = self.spec()
        frame = {"state": np.zeros(s.state_dim or 1, np.float32),
                 "images": {k: np.zeros((*s.image_hw.get(k, [224, 224]), 3), np.uint8) for k in s.image_keys},
                 "t": 0.0}
        for _ in range(2):
            self.infer([frame] * max(1, s.n_obs_steps), "warm up" if s.needs_task else None)
