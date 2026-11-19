"""What a policy server says about its model (``describe``), and what the agent fills in when it does not.

Every field except ``model`` may be unknown (None). The agent then uses the family profile
(``g1agent.profiles``) and, for the chunk length, a probe inference (``Agent.probe``).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from typing import Optional


@dataclass
class PolicySpec:
    model: str                                  # checkpoint name / path, for logs
    family: str = "unknown"                     # act | diffusion | pi0 | pi05 | smolvla | groot | unknown
    control_hz: Optional[float] = None          # rate the model was trained at (dataset fps)
    chunk_size: Optional[int] = None            # actions returned per inference (1 = single step)
    n_action_steps: Optional[int] = None        # how many of them the training recipe executes per call
    n_obs_steps: int = 1                        # frames of history the model takes
    state_dim: Optional[int] = None
    action_dim: Optional[int] = None
    state_names: list[str] = field(default_factory=list)   # robot joint names, in vector order
    action_names: list[str] = field(default_factory=list)
    image_keys: list[str] = field(default_factory=list)    # camera names the model expects
    image_hw: dict[str, list[int]] = field(default_factory=dict)  # name -> [H, W] (server resizes)
    action_space: str = "joint_absolute"        # the agent only executes absolute joint targets
    needs_task: bool = False                    # language-conditioned (pi0 / pi05 / smolvla / groot)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "PolicySpec":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})
