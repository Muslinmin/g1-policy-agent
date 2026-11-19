"""A model-free backend for tests and dry runs: a slow sinusoid around the measured posture.

``--chunk 0`` hides the chunk length from ``describe`` (the agent must probe it); ``--latency-ms`` adds a fixed
inference delay so the async path can be exercised; ``--family`` makes it report any family.
"""

from __future__ import annotations

import time
from typing import Optional

import numpy as np

from g1agent.robot import g1_layout
from g1agent.spec import PolicySpec

from .base import Backend


class DummyBackend(Backend):
    def __init__(self, chunk: int = 16, hz: float = 30.0, family: str = "unknown", latency_ms: float = 0.0,
                 amplitude: float = 0.1, with_hands: bool = True, with_waist: bool = False,
                 images: tuple[str, ...] = ("head",), report_chunk: bool = True) -> None:
        self.chunk, self.hz, self.family = int(chunk), float(hz), family
        self.latency_s, self.amp = latency_ms / 1e3, float(amplitude)
        self.names = g1_layout.joint_names(with_waist, with_hands)
        self.images, self.report_chunk = images, report_chunk

    def spec(self) -> PolicySpec:
        return PolicySpec(model=f"dummy-sine-{self.chunk}", family=self.family, control_hz=self.hz,
                          chunk_size=self.chunk if self.report_chunk else None, n_obs_steps=1,
                          state_dim=len(self.names), action_dim=len(self.names), state_names=self.names,
                          action_names=self.names, image_keys=list(self.images),
                          image_hw={k: [480, 640] for k in self.images}, needs_task=False)

    def infer(self, frames: list[dict], task: Optional[str]) -> np.ndarray:
        if self.latency_s:
            time.sleep(self.latency_s)
        f = frames[-1]
        q = np.asarray(f["state"], np.float32)
        t = f.get("t", 0.0) + np.arange(self.chunk) / self.hz
        wave = self.amp * np.sin(2 * np.pi * 0.25 * t)[:, None]
        out = np.repeat(q[None], self.chunk, 0)
        out[:, :2] += wave  # move the first two joints only
        return out.astype(np.float32)
