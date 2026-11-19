"""Chunking defaults per model family.

These are the execution recipes the reference implementations use. They are starting points, not
measurements on your robot: tune ``execute`` with the dry run (docs/CHUNKING.md).

    act        chunk 100, temporal ensembling every tick (Zhao et al. 2023, ACT), k = 0.01
    diffusion  horizon 16, executes 8, 2 observation frames (Chi et al. 2023, Diffusion Policy)
    pi0/pi05   chunk 50, executes part of it and replans; open-loop chunks with async prefetch
               (Physical Intelligence openpi defaults; real-time chunking, Black et al. 2025)
    smolvla    chunk 50, async (LeRobot async inference)
    groot      chunk 16, executes 8 (NVIDIA Isaac-GR00T N1.x action horizon 16)
    unknown    decided at start-up from the probed chunk length and the measured round trip
"""

from __future__ import annotations

FAMILY_DEFAULTS: dict[str, dict] = {
    "act":       {"strategy": "ensemble", "chunk_size": 100, "execute": 50, "ensemble_k": 0.01},
    "diffusion": {"strategy": "receding", "chunk_size": 16, "execute": 8, "n_obs_steps": 2},
    "pi0":       {"strategy": "async", "chunk_size": 50, "execute": 25},
    "pi05":      {"strategy": "async", "chunk_size": 50, "execute": 25},
    "smolvla":   {"strategy": "async", "chunk_size": 50, "execute": 25},
    "groot":     {"strategy": "async", "chunk_size": 16, "execute": 8},
    "unknown":   {"strategy": "auto"},
}

# lerobot `config.type` -> family
LEROBOT_TYPE_TO_FAMILY = {
    "act": "act", "diffusion": "diffusion", "pi0": "pi0", "pi05": "pi05",
    "smolvla": "smolvla", "groot": "groot",
}


def family_defaults(family: str) -> dict:
    return dict(FAMILY_DEFAULTS.get(family, FAMILY_DEFAULTS["unknown"]))
