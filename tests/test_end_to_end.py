"""Dummy policy server over real ZeroMQ + mock G1: the whole agent path, per model family."""

import socket
import threading

import numpy as np
import pytest

from g1agent.agent import Agent, AgentConfig
from g1agent.robot.mock import MockG1
from g1server.backends.dummy import DummyBackend
from g1server.server import serve


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def start_server(backend):
    port = free_port()
    threading.Thread(target=serve, args=(backend, "127.0.0.1", port), daemon=True).start()
    return f"127.0.0.1:{port}"


@pytest.mark.parametrize("family,chunk,latency_ms,report,expect", [
    ("groot", 16, 40.0, True, "async"),
    ("act", 100, 0.0, True, "ensemble"),
    ("diffusion", 16, 0.0, True, "receding"),
    ("unknown", 24, 60.0, False, "async"),   # chunk hidden: probed
    ("unknown", 1, 0.0, True, "single"),
])
def test_agent_runs(family, chunk, latency_ms, report, expect, tmp_path):
    addr = start_server(DummyBackend(chunk=chunk, hz=30.0, family=family, latency_ms=latency_ms,
                                     report_chunk=report))
    robot = MockG1()
    agent = Agent(robot, AgentConfig(server=addr, duration_s=2.0, log_dir=str(tmp_path), probe_infers=2))
    agent.start()
    assert agent.spec.chunk_size == chunk
    assert agent.strategy.name == expect
    s = agent.run()
    assert s["errors"] == 0 and "safety_stop" not in s
    assert s["achieved_hz"] > 25.0
    assert s["commanded"] >= 0.85 * s["ticks"], s
    assert robot.commands > 0
    assert len(list(tmp_path.glob("run_*.jsonl"))) == 1


def test_far_first_action_is_refused():
    class Far(DummyBackend):
        def infer(self, frames, task):
            return super().infer(frames, task) + 1.0
    addr = start_server(Far(chunk=16))
    agent = Agent(MockG1(), AgentConfig(server=addr, duration_s=1.0, log_dir=None, probe_infers=1))
    agent.start()
    s = agent.run()
    assert "first action" in s.get("safety_stop", "")


def test_dimension_mismatch_detected():
    addr = start_server(DummyBackend(chunk=16, with_hands=False))
    agent = Agent(MockG1(with_hands=True), AgentConfig(server=addr, duration_s=1.0, log_dir=None))
    with pytest.raises(Exception, match="state|joints"):
        agent.start()


def test_dry_run_commands_nothing():
    addr = start_server(DummyBackend(chunk=16))
    robot = MockG1()
    agent = Agent(robot, AgentConfig(server=addr, duration_s=1.0, log_dir=None, dry_run=True))
    agent.start()
    s = agent.run()
    assert s["commanded"] > 0 and robot.commands == 0
    np.testing.assert_allclose(robot.q, 0.0)
