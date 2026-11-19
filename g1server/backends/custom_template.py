"""Template for a model that is not a LeRobot checkpoint (openpi server, Isaac-GR00T's own inference, a
research model). Copy this file, fill in the three methods, and run ``python -m g1server --backend
mypkg.mymodule:MyBackend --arg key=value``.

Rules the agent relies on:
  * ``infer`` returns (T, D) float32 ABSOLUTE joint targets in the robot's joint order, un-normalised.
    If your model predicts deltas, add them to frames[-1]["state"] here. If it predicts end-effector poses,
    solve IK here. The agent only ever executes absolute joint targets.
  * actions[0] is meant for the tick at which frames[-1] was taken.
  * Fill in ``chunk_size`` if you know it. Leave it None if you do not: the agent probes it.
  * ``n_obs_steps`` = how many consecutive frames ``infer`` wants (oldest first).
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from g1agent.spec import PolicySpec

from .base import Backend


class MyBackend(Backend):
    def __init__(self, ckpt: str = "", device: str = "cuda") -> None:
        self.ckpt = ckpt
        # self.model = load_my_model(ckpt).to(device).eval()

    def spec(self) -> PolicySpec:
        return PolicySpec(model=self.ckpt, family="unknown", control_hz=None, chunk_size=None,
                          n_obs_steps=1, state_dim=None, action_dim=None, image_keys=["head"],
                          image_hw={"head": [224, 224]}, needs_task=False)

    def infer(self, frames: list[dict], task: Optional[str]) -> np.ndarray:
        raise NotImplementedError("load your model in __init__ and return (T, D) absolute joint targets")
