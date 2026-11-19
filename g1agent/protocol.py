"""Wire protocol between the robot agent (client) and the policy server.

Transport: ZeroMQ REQ (agent, on the robot) -> REP (policy server, on the GPU machine), one TCP address
``tcp://<server-ip>:<port>``. Payload: msgpack, with numpy arrays packed as an extension type (no pickle,
so the server never unpickles data from the network).

Requests (all dicts):

    {"method": "ping"}                                  -> {"ok": True, "t_server": float}
    {"method": "describe"}                              -> {"ok": True, "spec": PolicySpec.to_dict()}
    {"method": "reset"}                                 -> {"ok": True}
    {"method": "infer", "obs": [frame, ...],            -> {"ok": True, "actions": (T, D) float32,
     "task": str | None, "request_id": int}                 "infer_ms": float, "model": str}

    frame = {"state": (S,) float32, "images": {name: (H, W, 3) uint8 RGB}, "t": float}

``obs`` holds the last ``spec.n_obs_steps`` frames, oldest first, taken at the policy rate. The server is
stateless across requests: everything it needs is in the request, so it can be restarted at any time.
``actions[i]`` is the joint target for control tick ``t0 + i``, where ``t0`` is the tick at which the newest
frame was taken (``actions[0]`` = "now"). The agent skips the entries that are already in the past when the
reply arrives. ``T`` is the chunk length (1 for a single-step policy).

Errors: {"ok": False, "error": str}. The agent treats them as a missing chunk (it holds the robot).
"""

from __future__ import annotations

from typing import Any

import msgpack
import numpy as np

PROTOCOL_VERSION = 1
_ND = 42  # msgpack extension code for numpy arrays


def _default(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        arr = np.ascontiguousarray(obj)
        header = msgpack.packb([arr.dtype.str, list(arr.shape)])
        return msgpack.ExtType(_ND, len(header).to_bytes(4, "little") + header + arr.tobytes())
    if isinstance(obj, np.generic):
        return obj.item()
    raise TypeError(f"cannot serialise {type(obj).__name__}")


def _ext_hook(code: int, data: bytes) -> Any:
    if code != _ND:
        return msgpack.ExtType(code, data)
    n = int.from_bytes(data[:4], "little")
    dtype, shape = msgpack.unpackb(data[4:4 + n])
    if np.dtype(dtype).hasobject:
        raise ValueError("object arrays are not allowed")
    return np.frombuffer(data[4 + n:], dtype=np.dtype(dtype)).reshape(shape).copy()


def pack(msg: dict) -> bytes:
    return msgpack.packb(msg, default=_default, use_bin_type=True)


def unpack(raw: bytes) -> dict:
    return msgpack.unpackb(raw, ext_hook=_ext_hook, raw=False, strict_map_key=False)
