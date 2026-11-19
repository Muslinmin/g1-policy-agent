"""Unitree G1 upper body over unitree_sdk2 (DDS): arms (+ optional waist) via rt/arm_sdk, Dex3 hands.

The legs stay with Unitree's own locomotion controller: rt/arm_sdk is blended into the robot's command with a
weight (motor_cmd[29].q, 0 = Unitree's controller, 1 = us). The weight is ramped 0 -> 1 when the agent starts
and 1 -> 0 when it stops, exactly as in Unitree's g1_arm7_sdk_dds_example.py. The robot must already be
standing (or on its gantry) in a mode that accepts arm_sdk.

A publisher thread runs at ``pub_hz`` (default 100 Hz) and moves the commanded posture toward the latest
target at no more than ``max_vel`` rad/s per joint, so policy-rate targets (e.g. 30 Hz) reach the motors as a
smooth, rate-limited stream. If no new target arrives for ``watchdog_s``, it keeps the last posture.

STATUS: written against unitree_sdk2_python (2026) and the conventions of a working G1 deployment, but NOT
run on hardware in this repository. Before the first run: run Unitree's arm example on your robot, then
``g1-agent check-robot`` (reads state, moves nothing), then a dry run with the e-stop in hand.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Optional

import numpy as np

from . import g1_layout as L
from .base import Observation, Robot
from .cameras import Camera

log = logging.getLogger(__name__)


class UnitreeG1(Robot):
    def __init__(self, iface: str = "eth0", with_waist: bool = False, with_hands: bool = True,
                 cameras: Optional[list[Camera]] = None, pub_hz: float = 100.0, max_vel: float = 1.0,
                 ramp_s: float = 2.0, watchdog_s: float = 0.5, hand_kp_scale: float = 1.0,
                 domain: int = 0) -> None:
        self.iface, self.domain = iface, domain
        self.with_waist, self.with_hands = with_waist, with_hands
        self.joint_names = L.joint_names(with_waist, with_hands)
        self.lo, self.hi = L.limits_for(self.joint_names)
        self.cameras = cameras or []
        self.dt, self.max_vel, self.ramp_s, self.watchdog_s = 1.0 / pub_hz, max_vel, ramp_s, watchdog_s
        self.hand_kp = np.array(L.HAND_KP, np.float32) * float(hand_kp_scale)
        self.body_idx = L.ARM_MOTOR_INDEX + (L.WAIST_MOTOR_INDEX if with_waist else [])
        self.body_kp = list(L.ARM_KP) + (list(L.WAIST_KP) if with_waist else [])
        self.body_kd = list(L.ARM_KD) + (list(L.WAIST_KD) if with_waist else [])
        self.nb = len(self.body_idx)
        self._low = None
        self._hand_state = {}
        self._lock = threading.Lock()
        self._target: Optional[np.ndarray] = None
        self._cmd: Optional[np.ndarray] = None
        self._t_target = 0.0
        self._weight = 0.0
        self._weight_goal = 0.0
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    # ---- setup -------------------------------------------------------------------------------------------
    def connect(self, read_only: bool = False) -> None:
        """read_only: subscribe to state and cameras only; publish nothing (``g1-agent check-robot``)."""
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber
        from unitree_sdk2py.idl.default import unitree_hg_msg_dds__HandCmd_, unitree_hg_msg_dds__LowCmd_
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import HandCmd_, HandState_, LowCmd_, LowState_
        from unitree_sdk2py.utils.crc import CRC

        ChannelFactoryInitialize(self.domain, self.iface)
        sub = ChannelSubscriber("rt/lowstate", LowState_)
        sub.Init(self._on_low, 10)
        self._subs = [sub]
        if self.with_hands:
            self._new_hand_cmd = unitree_hg_msg_dds__HandCmd_
            self._hand_pub = {}
            for side in L.SIDES:
                s = ChannelSubscriber(f"rt/dex3/{side}/state", HandState_)
                s.Init(lambda m, side=side: self._hand_state.__setitem__(side, m), 10)
                self._subs.append(s)

        t = time.time()
        while self._low is None or (self.with_hands and len(self._hand_state) < 2):
            if time.time() - t > 5.0:
                missing = ["rt/lowstate"] if self._low is None else []
                missing += [f"rt/dex3/{s}/state" for s in L.SIDES if s not in self._hand_state]
                raise RuntimeError(f"no {', '.join(missing)} within 5 s on {self.iface} (DDS domain "
                                   f"{self.domain}); is the robot on and the interface right?")
            time.sleep(0.02)
        for c in self.cameras:
            c.start()
        if read_only:
            log.info("read-only: state and cameras only, nothing published")
            return
        self._crc = CRC()
        self._low_cmd = unitree_hg_msg_dds__LowCmd_()
        self._arm_pub = ChannelPublisher("rt/arm_sdk", LowCmd_)
        self._arm_pub.Init()
        if self.with_hands:
            for side in L.SIDES:
                p = ChannelPublisher(f"rt/dex3/{side}/cmd", HandCmd_)
                p.Init()
                self._hand_pub[side] = p
        q = self._measured()
        with self._lock:
            self._target, self._cmd, self._t_target = q.copy(), q.copy(), time.time()
        self._weight_goal = 1.0
        self._thread = threading.Thread(target=self._publish_loop, name="g1-arm-sdk", daemon=True)
        self._thread.start()
        t = time.time()
        while self._weight < 1.0:
            if time.time() - t > self.ramp_s + 2.0:
                raise RuntimeError("arm_sdk weight did not ramp up")
            time.sleep(0.02)
        log.info("arm_sdk active on %d body joints%s", self.nb, " + 14 hand joints" if self.with_hands else "")

    def _on_low(self, msg) -> None:
        self._low = msg

    # ---- state -------------------------------------------------------------------------------------------
    def _measured(self) -> np.ndarray:
        low = self._low
        q = [low.motor_state[i].q for i in self.body_idx]
        if self.with_hands:
            for side in L.SIDES:
                hs = self._hand_state[side]
                q += [hs.motor_state[j].q for j in range(7)]
        return np.array(q, np.float32)

    def observe(self) -> Observation:
        imgs = {c.name: c.latest() for c in self.cameras}
        return Observation(self._measured(), imgs, time.time())

    def command(self, q_target: np.ndarray) -> None:
        t = np.clip(np.asarray(q_target, np.float32), self.lo, self.hi)
        with self._lock:
            self._target, self._t_target = t, time.time()

    def hold(self) -> None:
        with self._lock:
            if self._cmd is not None:
                self._target = self._cmd.copy()

    # ---- publisher ---------------------------------------------------------------------------------------
    def _publish_loop(self) -> None:
        step = self.max_vel * self.dt
        dw = self.dt / self.ramp_s
        nxt = time.perf_counter()
        while not self._stop.is_set():
            with self._lock:
                if time.time() - self._t_target > self.watchdog_s:
                    self._target = self._cmd.copy()          # stale policy: keep the last posture
                self._cmd = self._cmd + np.clip(self._target - self._cmd, -step, step)
                cmd = self._cmd.copy()
            self._weight = float(np.clip(self._weight + np.sign(self._weight_goal - self._weight) * dw, 0, 1))
            self._write(cmd)
            nxt += self.dt
            time.sleep(max(0.0, nxt - time.perf_counter()))

    def _write(self, cmd: np.ndarray) -> None:
        lc = self._low_cmd
        for k, idx in enumerate(self.body_idx):
            m = lc.motor_cmd[idx]
            m.q, m.dq, m.tau, m.kp, m.kd = float(cmd[k]), 0.0, 0.0, self.body_kp[k], self.body_kd[k]
        lc.motor_cmd[L.ARM_SDK_WEIGHT_INDEX].q = self._weight
        lc.crc = self._crc.Crc(lc)
        self._arm_pub.Write(lc)
        if self.with_hands and self._weight > 0.0:
            for s_i, side in enumerate(L.SIDES):
                hc = self._new_hand_cmd()
                q = cmd[self.nb + 7 * s_i: self.nb + 7 * (s_i + 1)]
                for j in range(7):
                    m = hc.motor_cmd[j]
                    m.mode = (j & 0x0F) | (0x01 << 4) | (1 << 7)   # id | status enable | timeout (no CRC)
                    m.q, m.dq, m.tau = float(q[j]), 0.0, 0.0
                    m.kp, m.kd = float(self.hand_kp[j]), float(L.HAND_KD[j])
                self._hand_pub[side].Write(hc)

    # ---- shutdown ----------------------------------------------------------------------------------------
    def close(self) -> None:
        if self._thread is None:
            return
        self.hold()
        self._weight_goal = 0.0
        t = time.time()
        while self._weight > 0.0 and time.time() - t < self.ramp_s + 2.0:
            time.sleep(0.02)
        self._stop.set()
        self._thread.join(timeout=1.0)
        for c in self.cameras:
            c.stop()
        log.info("arm_sdk released")
