import numpy as np
import pytest

from g1agent.chunking import (AsyncChunking, Chunk, Receding, SingleStep, TemporalEnsemble, parse_override,
                              select_strategy)
from g1agent.spec import PolicySpec


def ramp(t0, T, D=2):
    """chunk whose action for tick t equals t (so alignment errors show up as wrong values)."""
    return Chunk(np.repeat(np.arange(t0, t0 + T, dtype=np.float32)[:, None], D, 1), t0)


def simulate(strategy, ticks, latency, T=16):
    """Drive a strategy with replies that arrive `latency` ticks after the request."""
    inflight = None
    out = []
    for t in range(ticks):
        if inflight and inflight[1] <= t:
            strategy.add(ramp(inflight[0], T), t)
            inflight = None
        if strategy.wants_query(t, inflight is not None):
            if strategy.blocking:
                strategy.add(ramp(t, T), t)
            else:
                inflight = (t, t + latency)
        out.append(strategy.action(t))
    return out


def test_receding_executes_k_then_replans():
    s = Receding(execute=8)
    acts = simulate(s, 40, latency=0)
    assert all(a is not None and a[0] == t for t, a in enumerate(acts))


def test_async_aligns_stale_prefix_and_never_starves_when_lead_is_right():
    s = AsyncChunking(execute=8, blend=0, margin=1)
    s.set_latency_ticks(3)
    acts = simulate(s, 120, latency=3)
    first = next(i for i, a in enumerate(acts) if a is not None)
    assert first == 3                                   # first chunk only exists once it arrived
    assert all(a is not None for a in acts[first:])     # no starvation afterwards
    assert all(a[0] == t for t, a in enumerate(acts) if a is not None)  # action for tick t is t


def test_async_starves_when_latency_underestimated():
    s = AsyncChunking(execute=4, blend=0, margin=0)
    s.set_latency_ticks(0)
    acts = simulate(s, 60, latency=10, T=8)
    assert any(a is None for a in acts[12:])


def test_async_blend_is_continuous():
    s = AsyncChunking(execute=8, blend=4, margin=1)
    s.set_latency_ticks(2)
    acts = [a for a in simulate(s, 80, latency=2) if a is not None]
    assert max(abs(b[0] - a[0]) for a, b in zip(acts, acts[1:])) <= 1.0 + 1e-6


def test_single_step_and_ensemble():
    assert all(a[0] == t for t, a in enumerate(simulate(SingleStep(), 10, 0, T=1)))
    e = TemporalEnsemble(k=0.01)
    acts = simulate(e, 30, latency=0, T=20)
    assert all(abs(a[0] - t) < 1e-5 for t, a in enumerate(acts))  # all chunks agree -> average equals t


def test_ensemble_weights_oldest_most():
    e = TemporalEnsemble(k=1.0)
    e.add(Chunk(np.zeros((5, 1), np.float32), 0), 0)
    e.add(Chunk(np.ones((5, 1), np.float32), 1), 1)
    a = e.action(1)[0]
    assert 0 < a < 0.5   # older (0) weighs exp(0), newer (1) exp(-1)


def spec(family, chunk, n_action_steps=None):
    return PolicySpec(model="m", family=family, chunk_size=chunk, n_action_steps=n_action_steps)


@pytest.mark.parametrize("family,chunk,rtt,expect", [
    ("act", 100, 0.010, "ensemble"),
    ("diffusion", 16, 0.010, "receding"),
    ("diffusion", 16, 0.200, "async"),      # too slow for blocking replans -> async
    ("pi05", 50, 0.120, "async"),
    ("groot", 16, 0.050, "async"),
    ("unknown", 1, 0.010, "single"),
    ("unknown", 32, 0.005, "receding"),
    ("unknown", 32, 0.100, "async"),
])
def test_selection(family, chunk, rtt, expect):
    assert select_strategy(spec(family, chunk), rtt, 30.0).name == expect


def test_async_execute_raised_above_latency():
    s = select_strategy(spec("groot", 16), rtt_s=0.25, control_hz=30.0)   # 8 ticks of latency
    assert s.name == "async" and s.execute > s.lead + s.margin


def test_ensemble_goes_non_blocking_on_slow_link():
    s = select_strategy(spec("act", 100), rtt_s=0.05, control_hz=30.0)
    assert s.name == "ensemble" and not s.blocking


def test_override_parsing():
    assert parse_override("async:20") == ("async", {"execute": 20})
    assert parse_override("ensemble:0.02") == ("ensemble", {"k": 0.02})
    assert select_strategy(spec("act", 100), 0.01, 30.0, "receding:10").execute == 10
    with pytest.raises(ValueError):
        parse_override("magic")
