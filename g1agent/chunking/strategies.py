"""The four execution strategies. See docs/CHUNKING.md for when each one is used."""

from __future__ import annotations

from collections import deque
from typing import Optional

import numpy as np

from .base import Chunk, Strategy


class SingleStep(Strategy):
    """One action per inference, queried every tick (blocking). For policies whose chunk length is 1."""

    name = "single"
    blocking = True

    def __init__(self) -> None:
        super().__init__()
        self.chunk: Optional[Chunk] = None

    def reset(self) -> None:
        self.chunk = None

    def wants_query(self, tick: int, pending: bool) -> bool:
        return not pending

    def add(self, chunk: Chunk, tick: int) -> None:
        self.chunk = chunk

    def action(self, tick: int) -> Optional[np.ndarray]:
        # A single-step reply is used for the tick it arrives in, even if it is a tick late.
        return None if self.chunk is None else self.chunk.actions[0]


class Receding(Strategy):
    """Execute the first ``execute`` actions of a chunk, then stop and query again (blocking).

    The classic Diffusion Policy / GR00T recipe. Simple and predictable; the robot pauses for one round trip
    at every replan, so use it when the round trip is short compared with the control period.
    """

    name = "receding"
    blocking = True

    def __init__(self, execute: int) -> None:
        super().__init__()
        if execute < 1:
            raise ValueError("execute must be >= 1")
        self.execute = int(execute)
        self.chunk: Optional[Chunk] = None

    def reset(self) -> None:
        self.chunk = None

    def wants_query(self, tick: int, pending: bool) -> bool:
        if pending:
            return False
        return self.chunk is None or tick - self.chunk.t0 >= min(self.execute, len(self.chunk.actions))

    def add(self, chunk: Chunk, tick: int) -> None:
        self.chunk = chunk

    def action(self, tick: int) -> Optional[np.ndarray]:
        if self.chunk is None or tick - self.chunk.t0 >= self.execute:
            return None
        return self.chunk.at(tick)

    def describe(self) -> str:
        return f"receding(execute={self.execute})"


class AsyncChunking(Strategy):
    """Keep executing the current chunk while the next one is computed (non-blocking).

    A new request starts ``lead`` ticks (the measured latency, plus a margin) before the current chunk's
    ``execute`` budget runs out, so the next chunk normally arrives before the robot has nothing to do.
    When it arrives, its stale prefix is skipped and the switch is cross-faded over ``blend`` ticks to avoid a
    step at the seam. This is the open-loop core of "real-time chunking" (Black et al. 2025) without the
    inpainting of the overlap, which needs support inside the model.
    """

    name = "async"
    blocking = False

    def __init__(self, execute: int, blend: int = 4, margin: int = 2) -> None:
        super().__init__()
        if execute < 1:
            raise ValueError("execute must be >= 1")
        self.execute, self.blend, self.margin = int(execute), int(blend), int(margin)
        self.cur: Optional[Chunk] = None
        self.prev: Optional[Chunk] = None
        self.switch_tick = 0

    def reset(self) -> None:
        self.cur = self.prev = None

    def wants_query(self, tick: int, pending: bool) -> bool:
        if pending:
            return False
        if self.cur is None:
            return True
        return tick >= self.cur.t0 + self.execute - self.lead - self.margin

    def add(self, chunk: Chunk, tick: int) -> None:
        if chunk.end <= tick:          # arrived entirely in the past: useless, keep the current one
            return
        self.prev, self.cur, self.switch_tick = self.cur, chunk, tick

    def action(self, tick: int) -> Optional[np.ndarray]:
        new = None if self.cur is None else self.cur.at(tick)
        old = None if self.prev is None else self.prev.at(tick)
        if new is None:
            return old  # current chunk ran out (starvation) or not started: fall back to the old tail
        if old is None or self.blend <= 0:
            return new
        w = min(1.0, (tick - self.switch_tick + 1) / float(self.blend))
        return (1.0 - w) * old + w * new

    def describe(self) -> str:
        return f"async(execute={self.execute}, blend={self.blend}, lead={self.lead}+{self.margin})"


class TemporalEnsemble(Strategy):
    """ACT temporal ensembling: query every tick and average all chunks that cover the current tick.

    Weights follow ACT: w_i = exp(-k * i), i = 0 for the OLDEST prediction (so older predictions weigh more,
    which smooths the motion). Works blocking (round trip < one period) or non-blocking (one request in
    flight at a time; fewer, latency-aligned chunks).
    """

    name = "ensemble"

    def __init__(self, k: float = 0.01, blocking: bool = True, max_chunks: int = 200) -> None:
        super().__init__()
        self.k, self.blocking = float(k), bool(blocking)
        self.chunks: deque[Chunk] = deque(maxlen=max_chunks)

    def reset(self) -> None:
        self.chunks.clear()

    def wants_query(self, tick: int, pending: bool) -> bool:
        return not pending

    def add(self, chunk: Chunk, tick: int) -> None:
        self.chunks.append(chunk)
        while self.chunks and self.chunks[0].end <= tick:
            self.chunks.popleft()

    def action(self, tick: int) -> Optional[np.ndarray]:
        preds = [a for a in (c.at(tick) for c in self.chunks) if a is not None]  # oldest first
        if not preds:
            return None
        w = np.exp(-self.k * np.arange(len(preds)))
        return (w[:, None] * np.stack(preds)).sum(0) / w.sum()

    def describe(self) -> str:
        return f"ensemble(k={self.k}, {'blocking' if self.blocking else 'non-blocking'})"
