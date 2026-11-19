"""Chunk execution strategies.

Time is counted in control ticks (the agent's fixed-rate loop). A chunk returned for the observation taken
at tick ``t0`` holds ``actions[i]`` = the joint target for tick ``t0 + i``. If the reply arrives at tick
``t > t0`` (network + inference latency), the first ``t - t0`` actions are already in the past and are
skipped: every strategy indexes chunks by absolute tick, never by "next unused action".
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class Chunk:
    actions: np.ndarray  # (T, D)
    t0: int              # tick of the newest observation frame used for this chunk

    @property
    def end(self) -> int:
        """First tick NOT covered by this chunk."""
        return self.t0 + len(self.actions)

    def at(self, tick: int) -> Optional[np.ndarray]:
        i = tick - self.t0
        return self.actions[i] if 0 <= i < len(self.actions) else None


class Strategy(ABC):
    name = "base"
    blocking = True  # True: the agent waits for the reply in the same tick; False: request runs in background

    def __init__(self) -> None:
        self.lead = 0  # expected latency in ticks (set by the agent from the measured round trip)

    def reset(self) -> None:
        pass

    def set_latency_ticks(self, ticks: int) -> None:
        self.lead = max(0, int(ticks))

    @abstractmethod
    def wants_query(self, tick: int, pending: bool) -> bool:
        """Start a new inference request at this tick? ``pending``: one is already in flight."""

    @abstractmethod
    def add(self, chunk: Chunk, tick: int) -> None:
        """A reply arrived at ``tick``."""

    @abstractmethod
    def action(self, tick: int) -> Optional[np.ndarray]:
        """Target for this tick, or None if nothing covers it (the agent then holds the robot)."""

    def describe(self) -> str:
        return self.name
