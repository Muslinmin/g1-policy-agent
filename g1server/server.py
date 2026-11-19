"""Policy server: runs on the GPU machine, answers the agent on one TCP port.

    python -m g1server --backend lerobot --ckpt outputs/train/<run>/checkpoints/last/pretrained_model \\
                       --host 0.0.0.0 --port 5555
    python -m g1server --backend dummy --chunk 16 --latency-ms 40          # no model, for dry runs
    python -m g1server --backend mypkg.mymodule:MyBackend --arg ckpt=/path

Protocol: g1agent/protocol.py. One request at a time (ZeroMQ REP). Stateless across requests.
"""

from __future__ import annotations

import argparse
import importlib
import logging
import sys
import time

import numpy as np
import zmq

from g1agent import protocol

from .backends.base import Backend

log = logging.getLogger("g1server")


def make_backend(args) -> Backend:
    if args.backend == "dummy":
        from .backends.dummy import DummyBackend
        return DummyBackend(chunk=args.chunk or 16, hz=args.hz or 30.0, family=args.family,
                            latency_ms=args.latency_ms, report_chunk=args.chunk != 0,
                            with_hands=not args.no_hands, with_waist=args.waist)
    if args.backend == "lerobot":
        if not args.ckpt:
            raise SystemExit("--ckpt is required for --backend lerobot")
        from .backends.lerobot_backend import LeRobotBackend
        return LeRobotBackend(args.ckpt, device=args.device, hz=args.hz)
    mod, _, cls = args.backend.partition(":")
    kwargs = dict(a.split("=", 1) for a in args.arg)
    return getattr(importlib.import_module(mod), cls)(**kwargs)


def serve(backend: Backend, host: str, port: int, max_requests: int = 0) -> None:
    spec = backend.spec()
    sock = zmq.Context.instance().socket(zmq.REP)
    sock.bind(f"tcp://{host}:{port}")
    log.info("serving %s on tcp://%s:%d (protocol v%d)", spec.model, host, port, protocol.PROTOCOL_VERSION)
    n = 0
    while max_requests <= 0 or n < max_requests:
        raw = sock.recv()
        n += 1
        try:
            req = protocol.unpack(raw)
            m = req.get("method")
            if m == "ping":
                rep = {"ok": True, "t_server": time.time()}
            elif m == "describe":
                rep = {"ok": True, "spec": spec.to_dict(), "protocol": protocol.PROTOCOL_VERSION}
            elif m == "reset":
                backend.reset()
                rep = {"ok": True}
            elif m == "infer":
                t = time.perf_counter()
                actions = np.asarray(backend.infer(req["obs"], req.get("task")), np.float32)
                if actions.ndim == 1:
                    actions = actions[None]
                rep = {"ok": True, "actions": actions, "infer_ms": (time.perf_counter() - t) * 1e3,
                       "model": spec.model, "request_id": req.get("request_id", 0)}
            else:
                rep = {"ok": False, "error": f"unknown method {m!r}"}
        except Exception as e:  # noqa: BLE001 - report to the agent, keep serving
            log.exception("request failed")
            rep = {"ok": False, "error": f"{type(e).__name__}: {e}"}
        sock.send(protocol.pack(rep))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="g1server", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--backend", default="lerobot", help="lerobot | dummy | package.module:Class")
    p.add_argument("--ckpt", default=None)
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=5555)
    p.add_argument("--device", default="cuda")
    p.add_argument("--hz", type=float, default=None, help="override the training rate reported to the agent")
    p.add_argument("--no-warmup", action="store_true")
    p.add_argument("--arg", action="append", default=[], help="key=value for a custom backend")
    g = p.add_argument_group("dummy backend")
    g.add_argument("--chunk", type=int, default=16, help="0 = do not report it (agent probes)")
    g.add_argument("--family", default="unknown")
    g.add_argument("--latency-ms", type=float, default=0.0)
    g.add_argument("--no-hands", action="store_true")
    g.add_argument("--waist", action="store_true")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    backend = make_backend(args)
    if not args.no_warmup:
        log.info("warming up")
        backend.warm_up()
    serve(backend, args.host, args.port)
    return 0


if __name__ == "__main__":
    sys.exit(main())
