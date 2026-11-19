"""Robot interface the agent drives. Implementations: MockG1 (no hardware) and UnitreeG1 (unitree_sdk2)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Observation:
    state: np.ndarray                                     # (S,) float32, in Robot.joint_names order
    images: dict[str, np.ndarray] = field(default_factory=dict)  # name -> (H, W, 3) uint8 RGB
    t: float = 0.0                                        # time.time() when the state was read

    def to_frame(self) -> dict:
        return {"state": self.state.astype(np.float32), "images": self.images, "t": self.t}


class Robot(ABC):
    joint_names: list[str]

    @abstractmethod
    def connect(self) -> None:
        """Open communication; must leave the robot holding its current posture."""

    @abstractmethod
    def observe(self) -> Observation:
        ...

    @abstractmethod
    def command(self, q_target: np.ndarray) -> None:
        """Absolute joint target (rad), Robot.joint_names order. Implementations rate-limit it."""

    @abstractmethod
    def hold(self) -> None:
        """Stop moving: keep the last commanded posture."""

    @abstractmethod
    def close(self) -> None:
        """Hand control back safely (e.g. ramp the arm_sdk weight to 0)."""
