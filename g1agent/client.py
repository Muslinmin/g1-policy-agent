"""ZeroMQ client for the policy server, plus a one-request-in-flight background requester.

A REQ socket that misses a reply is stuck in the wrong state, so on any timeout the socket is closed
(linger 0) and recreated. Each socket is used by one thread only: ``Requester`` owns its own client.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass
from typing import Optional

import zmq

from . import protocol
from .spec import PolicySpec

log = logging.getLogger(__name__)


class PolicyError(RuntimeError):
    pass


class PolicyClient:
    def __init__(self, address: str, timeout_s: float = 2.0) -> None:
        if not address.startswith("tcp://"):
            address = "tcp://" + address
        self.address, self.timeout_ms = address, int(timeout_s * 1000)
        self.ctx = zmq.Context.instance()
        self.sock: Optional[zmq.Socket] = None
        self._connect()

    def _connect(self) -> None:
        if self.sock is not None:
            self.sock.close(linger=0)
        self.sock = self.ctx.socket(zmq.REQ)
        self.sock.setsockopt(zmq.LINGER, 0)
        self.sock.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
        self.sock.setsockopt(zmq.SNDTIMEO, self.timeout_ms)
        self.sock.connect(self.address)

    def call(self, msg: dict, timeout_s: Optional[float] = None) -> dict:
        if timeout_s is not None:
            self.sock.setsockopt(zmq.RCVTIMEO, int(timeout_s * 1000))
        try:
            self.sock.send(protocol.pack(msg))
            reply = protocol.unpack(self.sock.recv())
        except zmq.Again:
            self._connect()
            raise PolicyError(f"no reply from {self.address} within the timeout ({msg.get('method')})")
        finally:
            if timeout_s is not None and self.sock is not None:
                self.sock.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
        if not reply.get("ok", False):
            raise PolicyError(f"server error: {reply.get('error')}")
        return reply

    def ping(self) -> float:
        t = time.perf_counter()
        self.call({"method": "ping"})
        return time.perf_counter() - t

    def describe(self) -> PolicySpec:
        return PolicySpec.from_dict(self.call({"method": "describe"}, timeout_s=10.0)["spec"])

    def reset(self) -> None:
        self.call({"method": "reset"})

    def infer(self, frames: list[dict], task: Optional[str], request_id: int = 0,
              timeout_s: Optional[float] = None) -> dict:
        return self.call({"method": "infer", "obs": frames, "task": task, "request_id": request_id},
                         timeout_s=timeout_s)

    def close(self) -> None:
        if self.sock is not None:
            self.sock.close(linger=0)
            self.sock = None


@dataclass
class Reply:
    request_id: int
    t0: int                      # tick of the newest frame in the request
    actions: Optional[object]    # (T, D) np.ndarray, None on failure
    rtt_s: float
    infer_ms: float = 0.0
    error: str = ""


class Requester:
    """Runs inference requests in one background thread, one at a time."""

    def __init__(self, address: str, timeout_s: float, task: Optional[str]) -> None:
        self.client = PolicyClient(address, timeout_s)
        self.task = task
        self._in: "queue.Queue[Optional[tuple]]" = queue.Queue(maxsize=1)
        self._out: "queue.Queue[Reply]" = queue.Queue()
        self.pending = False
        self._next_id = 0
        self._thread = threading.Thread(target=self._run, name="policy-requester", daemon=True)
        self._thread.start()

    def submit(self, frames: list[dict], t0: int) -> int:
        if self.pending:
            raise RuntimeError("a request is already in flight")
        self._next_id += 1
        self.pending = True
        self._in.put((self._next_id, t0, frames))
        return self._next_id

    def poll(self, wait_s: float = 0.0) -> Optional[Reply]:
        try:
            r = self._out.get(timeout=wait_s) if wait_s > 0 else self._out.get_nowait()
        except queue.Empty:
            return None
        self.pending = False
        return r

    def _run(self) -> None:
        while True:
            item = self._in.get()
            if item is None:
                return
            rid, t0, frames = item
            t = time.perf_counter()
            try:
                rep = self.client.infer(frames, self.task, rid)
                self._out.put(Reply(rid, t0, rep["actions"], time.perf_counter() - t, rep.get("infer_ms", 0.0)))
            except Exception as e:  # noqa: BLE001 - any failure is a missing chunk, the agent holds
                self._out.put(Reply(rid, t0, None, time.perf_counter() - t, error=str(e)))

    def close(self) -> None:
        self._in.put(None)
        self._thread.join(timeout=2.0)
        self.client.close()
