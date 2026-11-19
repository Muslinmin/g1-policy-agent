"""The control loop that runs on the robot.

start():
  1. ``describe`` the server's model; check that its state/action size matches the robot and that every camera
     it wants is mapped to a robot camera;
  2. if the chunk length is unknown, probe it with one inference on the current observation;
  3. measure the inference round trip (median of a few real inferences, not pings);
  4. choose the execution strategy (g1agent.chunking.select_strategy).

run(): a fixed-rate loop at ``control_hz``. Each tick: observe, collect any finished reply, start a request if
the strategy asks for one (blocking strategies wait for it here), take the strategy's action for this tick,
pass it through Safety, command the robot. No action for this tick -> the robot holds. Every tick is logged.
"""

from __future__ import annotations

import json
import logging
import math
import signal
import statistics
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from .chunking import Chunk, Strategy, select_strategy
from .client import PolicyClient, Reply, Requester
from .profiles import family_defaults
from .robot.base import Robot
from .robot.safety import Safety, SafetyError
from .spec import PolicySpec

log = logging.getLogger(__name__)


@dataclass
class AgentConfig:
    server: str                                   # "tcp://<ip>:<port>" or "<ip>:<port>"
    strategy: str = "auto"
    control_hz: Optional[float] = None            # default: the server's training rate, else 30
    task: Optional[str] = None                    # language instruction (pi0 / pi05 / smolvla / groot)
    duration_s: float = 60.0
    timeout_s: float = 2.0
    camera_map: dict[str, str] = field(default_factory=dict)  # policy image key -> robot camera name
    dry_run: bool = False                         # compute and log actions, never command the robot
    allow_far_start: bool = False
    max_step: float = 0.05                        # rad per tick
    first_jump: float = 0.35                      # rad
    log_dir: Optional[str] = "logs"
    probe_infers: int = 3


class Agent:
    def __init__(self, robot: Robot, cfg: AgentConfig) -> None:
        self.robot, self.cfg = robot, cfg
        self.spec: Optional[PolicySpec] = None
        self.strategy: Optional[Strategy] = None
        self.hz = 30.0
        self.rtt_s = 0.0
        self._stop = False

    # ---- start-up ----------------------------------------------------------------------------------------
    def _frames(self, history) -> list[dict]:
        frames = []
        for obs in history:
            f = obs.to_frame()
            f["images"] = {key: obs.images[cam] for key, cam in self._cam_for_key.items()}
            frames.append(f)
        return frames

    def start(self) -> None:
        cfg = self.cfg
        client = PolicyClient(cfg.server, cfg.timeout_s)
        try:
            ping = statistics.median(client.ping() for _ in range(5))
            spec = client.describe()
            log.info("server %s: model %s, family %s, chunk %s, %s Hz, %d obs frame(s), task %s; ping %.1f ms",
                     cfg.server, spec.model, spec.family, spec.chunk_size, spec.control_hz, spec.n_obs_steps,
                     "required" if spec.needs_task else "no", ping * 1e3)
            prof = family_defaults(spec.family)
            if spec.n_obs_steps < 1:
                spec.n_obs_steps = int(prof.get("n_obs_steps", 1))
            self.robot.connect()
            self._check_compat(spec)
            if spec.needs_task and not cfg.task:
                raise SystemExit(f"{spec.family} is language-conditioned: pass --task \"...\"")
            obs = self.robot.observe()
            frames = self._frames([obs] * spec.n_obs_steps)
            times, shape = [], None
            for _ in range(max(1, cfg.probe_infers)):
                t = time.perf_counter()
                rep = client.infer(frames, cfg.task, 0, timeout_s=max(cfg.timeout_s, 30.0))
                times.append(time.perf_counter() - t)
                shape = np.asarray(rep["actions"]).shape
            if len(shape) == 1:
                shape = (1, shape[0])
            if spec.chunk_size is None:
                log.info("server did not report a chunk length: probed %d", shape[0])
            elif spec.chunk_size != shape[0]:
                log.warning("server reports chunk %d but returned %d actions: using %d", spec.chunk_size,
                            shape[0], shape[0])
            spec.chunk_size = int(shape[0])
            if shape[1] != len(self.robot.joint_names):
                raise SafetyError(f"policy returns {shape[1]}-D actions, robot has {len(self.robot.joint_names)} "
                                  "joints: retrain on this joint layout or change the robot layout")
            self.rtt_s = statistics.median(times[1:] if len(times) > 1 else times)  # first call warms up
            self.hz = float(cfg.control_hz or spec.control_hz or 30.0)
            if spec.control_hz and abs(self.hz - spec.control_hz) > 1e-6:
                log.warning("control rate %.1f Hz differs from the training rate %.1f Hz: chunks will be "
                            "played faster/slower than they were demonstrated", self.hz, spec.control_hz)
            self.spec = spec
            self.strategy = select_strategy(spec, self.rtt_s, self.hz, cfg.strategy)
            log.info("inference round trip %.1f ms (%.2f control periods)", self.rtt_s * 1e3,
                     self.rtt_s * self.hz)
        finally:
            client.close()

    def _check_compat(self, spec: PolicySpec) -> None:
        names = self.robot.joint_names
        if spec.state_dim is not None and spec.state_dim != len(names):
            raise SafetyError(f"policy state is {spec.state_dim}-D, robot gives {len(names)} joints")
        if spec.state_names and spec.state_names != names:
            diff = [(a, b) for a, b in zip(spec.state_names, names) if a != b][:3]
            raise SafetyError(f"policy state order differs from the robot's, e.g. {diff}")
        obs = self.robot.observe()
        self._cam_for_key = {}
        for key in spec.image_keys:
            cam = self.cfg.camera_map.get(key, key)
            if cam not in obs.images:
                raise SafetyError(f"policy wants camera '{key}', robot has {sorted(obs.images)}: map it with "
                                  f"--camera {key}=<robot camera>")
            self._cam_for_key[key] = cam

    # ---- loop --------------------------------------------------------------------------------------------
    def stop(self, *_):
        self._stop = True

    def run(self) -> dict:
        assert self.spec is not None and self.strategy is not None, "call start() first"
        cfg, spec, strat = self.cfg, self.spec, self.strategy
        period = 1.0 / self.hz
        req = Requester(cfg.server, cfg.timeout_s, cfg.task)
        hist: deque = deque(maxlen=spec.n_obs_steps)
        safety = Safety(*self._limits(), max_step=cfg.max_step, first_jump=cfg.first_jump,
                        allow_far_start=cfg.allow_far_start)
        logf = self._open_log()
        stats = {"ticks": 0, "commanded": 0, "starved": 0, "errors": 0, "overruns": 0, "requests": 0,
                 "rtt_ms": [], "infer_ms": []}
        ema = self.rtt_s
        old = signal.signal(signal.SIGINT, self.stop)
        tick, t_start = 0, time.perf_counter()
        nxt = t_start
        try:
            obs = self.robot.observe()
            safety.reset(obs.state)
            while not self._stop and time.perf_counter() - t_start < cfg.duration_s:
                obs = self.robot.observe()
                hist.append(obs)
                while len(hist) < spec.n_obs_steps:
                    hist.appendleft(obs)
                replies = []
                r = req.poll()
                if r is not None:
                    replies.append(r)
                queried = False
                if strat.wants_query(tick, req.pending):
                    req.submit(self._frames(hist), tick)
                    stats["requests"] += 1
                    queried = True
                    if strat.blocking:
                        r = req.poll(wait_s=cfg.timeout_s + 0.5)
                        if r is not None:
                            replies.append(r)
                for r in replies:
                    stats["rtt_ms"].append(r.rtt_s * 1e3)
                    if r.actions is None:
                        stats["errors"] += 1
                        log.warning("request %d failed: %s", r.request_id, r.error)
                        continue
                    stats["infer_ms"].append(r.infer_ms)
                    a = np.asarray(r.actions, np.float32)
                    strat.add(Chunk(a.reshape(len(a), -1) if a.ndim == 2 else a[None], r.t0), tick)
                    ema = 0.8 * ema + 0.2 * r.rtt_s
                    strat.set_latency_ticks(int(math.ceil(ema / period)))
                action = strat.action(tick)
                row = {"tick": tick, "t": round(time.time(), 4), "queried": queried,
                       "replies": [round(r.rtt_s * 1e3, 1) for r in replies]}
                if action is None:
                    self.robot.hold()
                    stats["starved"] += 1
                    row["hold"] = True
                else:
                    target = safety.filter(action, self.robot.joint_names)
                    if not cfg.dry_run:
                        self.robot.command(target)
                    stats["commanded"] += 1
                    row["target"] = [round(float(x), 4) for x in target]
                row["state"] = [round(float(x), 4) for x in obs.state]
                if logf:
                    logf.write(json.dumps(row) + "\n")
                tick += 1
                stats["ticks"] = tick
                nxt += period
                lag = time.perf_counter() - nxt
                if lag > 0:
                    stats["overruns"] += 1
                    if lag > period:
                        nxt = time.perf_counter()   # do not try to catch up a long stall
                else:
                    time.sleep(-lag)
        except SafetyError as e:
            log.error("SAFETY STOP: %s", e)
            stats["safety_stop"] = str(e)
        finally:
            signal.signal(signal.SIGINT, old)
            self.robot.hold()
            req.close()
            if logf:
                logf.close()
        return self._summary(stats, time.perf_counter() - t_start, safety)

    def _limits(self):
        lo = getattr(self.robot, "lo", None)
        hi = getattr(self.robot, "hi", None)
        if lo is None:
            n = len(self.robot.joint_names)
            lo, hi = np.full(n, -np.pi, np.float32), np.full(n, np.pi, np.float32)
        return lo, hi

    def _open_log(self):
        if not self.cfg.log_dir:
            return None
        d = Path(self.cfg.log_dir)
        d.mkdir(parents=True, exist_ok=True)
        p = d / time.strftime("run_%Y%m%d_%H%M%S.jsonl")
        f = p.open("w")
        f.write(json.dumps({"spec": self.spec.to_dict(), "strategy": self.strategy.describe(), "hz": self.hz,
                            "rtt_ms": self.rtt_s * 1e3, "dry_run": self.cfg.dry_run,
                            "joint_names": self.robot.joint_names}) + "\n")
        log.info("logging to %s", p)
        return f

    def _summary(self, s: dict, elapsed: float, safety: Safety) -> dict:
        def pct(v, q):
            return round(float(np.percentile(v, q)), 1) if v else None
        out = {k: v for k, v in s.items() if k not in ("rtt_ms", "infer_ms")}
        out.update(elapsed_s=round(elapsed, 2), achieved_hz=round(s["ticks"] / max(elapsed, 1e-9), 2),
                   rtt_ms_p50=pct(s["rtt_ms"], 50), rtt_ms_p95=pct(s["rtt_ms"], 95),
                   infer_ms_p50=pct(s["infer_ms"], 50), clipped_ticks=safety.clipped,
                   strategy=self.strategy.describe(), model=self.spec.model)
        log.info("summary: %s", out)
        return out
