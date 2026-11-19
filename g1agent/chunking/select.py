"""Pick an execution strategy from what the server says, the family defaults, and the measured latency.

Order of precedence:
  1. an explicit ``--strategy`` from the operator;
  2. the chunk length: reported by the server, else probed (one inference); 1 -> single step;
  3. the family default (act -> ensemble, diffusion -> receding, pi0/pi05/smolvla/groot -> async);
  4. latency checks that downgrade a choice that cannot work on this network:
       ensemble, blocking, needs round trip < 0.8 period, else non-blocking ensemble;
       receding pauses one round trip per replan: above 2 periods it becomes async;
       async needs execute > lead + margin, else execute is raised (capped at the chunk length).
Unknown family with a chunk: async with execute = half the chunk (or receding if the round trip is short).
"""

from __future__ import annotations

import logging
import math
from typing import Optional

from ..profiles import family_defaults
from ..spec import PolicySpec
from .base import Strategy
from .strategies import AsyncChunking, Receding, SingleStep, TemporalEnsemble

log = logging.getLogger(__name__)


def parse_override(text: str) -> tuple[str, dict]:
    """'async:20' -> ('async', {'execute': 20}); 'ensemble:0.02' -> ('ensemble', {'k': 0.02})."""
    name, _, arg = text.partition(":")
    name = name.strip().lower()
    if name not in ("auto", "single", "receding", "async", "ensemble"):
        raise ValueError(f"unknown strategy '{text}' (auto | single | receding:K | async:K | ensemble[:k])")
    opts: dict = {}
    if arg:
        opts = {"k": float(arg)} if name == "ensemble" else {"execute": int(arg)}
    return name, opts


def select_strategy(spec: PolicySpec, rtt_s: float, control_hz: float, override: str = "auto") -> Strategy:
    period = 1.0 / control_hz
    lead = int(math.ceil(rtt_s / period))
    chunk = int(spec.chunk_size or 1)
    prof = family_defaults(spec.family)
    name, opts = parse_override(override)
    reason = "operator override"

    if name == "auto":
        if chunk <= 1:
            name, reason = "single", "chunk length 1"
        elif prof["strategy"] != "auto":
            name, reason = prof["strategy"], f"{spec.family} default"
        else:
            name = "receding" if rtt_s < 0.5 * period else "async"
            reason = f"unknown family, chunk {chunk}, round trip {rtt_s * 1e3:.0f} ms"

    if chunk <= 1 and name != "single":
        log.warning("chunk length is 1: '%s' makes no sense, using single step", name)
        name = "single"

    execute = int(opts.get("execute") or spec.n_action_steps or prof.get("execute") or max(1, chunk // 2))
    execute = max(1, min(execute, chunk))

    if name == "single":
        s: Strategy = SingleStep()
        if rtt_s > period:
            log.warning("single step with a %.0f ms round trip at %.0f Hz: the loop will run slower than "
                        "%.0f Hz", rtt_s * 1e3, control_hz, control_hz)
    elif name == "ensemble":
        blocking = rtt_s < 0.8 * period
        s = TemporalEnsemble(k=float(opts.get("k", prof.get("ensemble_k", 0.01))), blocking=blocking)
        if not blocking:
            reason += f"; round trip {rtt_s * 1e3:.0f} ms > 0.8 period -> non-blocking"
    elif name == "receding" and rtt_s <= 2 * period:
        s = Receding(execute)
    else:
        if name == "receding":
            reason += f"; round trip {rtt_s * 1e3:.0f} ms > 2 periods -> async (no pause per replan)"
        margin = 2
        if execute <= lead + margin:
            new = min(chunk, lead + margin + 1)
            log.warning("execute %d <= latency %d + margin %d ticks: raising execute to %d", execute, lead,
                        margin, new)
            execute = new
        if lead + margin >= chunk:
            log.warning("round trip (%d ticks) is about as long as the whole chunk (%d): expect stalls",
                        lead, chunk)
        s = AsyncChunking(execute=execute, blend=min(4, max(1, execute // 4)), margin=margin)
    s.set_latency_ticks(lead)
    log.info("strategy: %s (%s; chunk %d, latency %d ticks at %.1f Hz)", s.describe(), reason, chunk, lead,
             control_hz)
    return s
