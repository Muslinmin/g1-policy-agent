"""g1-agent: run on the G1's computer.

    g1-agent describe    --server 192.168.123.50:5555
    g1-agent check-robot --iface eth0 --camera head=realsense:<serial>       # reads state, moves nothing
    g1-agent run --server 192.168.123.50:5555 --robot mock                     # no hardware
    g1-agent run --server 192.168.123.50:5555 --robot unitree --iface eth0 \\
                 --camera head=realsense:<serial> --task "open the door" --dry-run
    g1-agent run ... (same, without --dry-run, e-stop in hand)

--strategy: auto (default) | single | receding:K | async:K | ensemble[:k]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

from .agent import Agent, AgentConfig
from .client import PolicyClient


def _robot(args, read_only: bool = False):
    if args.robot == "mock":
        from .robot.mock import MockG1
        cams = tuple(c.partition("=")[0] for c in args.camera) or ("head",)
        return MockG1(with_waist=args.waist, with_hands=not args.no_hands, image_names=cams)
    from .robot.cameras import parse_camera
    from .robot.unitree_g1 import UnitreeG1
    return UnitreeG1(iface=args.iface, with_waist=args.waist, with_hands=not args.no_hands,
                     cameras=[parse_camera(c) for c in args.camera], max_vel=args.max_vel,
                     hand_kp_scale=args.hand_kp_scale)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="g1-agent", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("describe", help="print what the policy server reports")
    d.add_argument("--server", required=True)

    def robot_args(sp):
        sp.add_argument("--robot", choices=["mock", "unitree"], default="unitree")
        sp.add_argument("--iface", default="eth0", help="network interface on the robot's DDS network")
        sp.add_argument("--camera", action="append", default=[],
                        help="name=realsense:<serial>[:WxH@FPS] | name=opencv:<dev> (repeatable)")
        sp.add_argument("--waist", action="store_true", help="also control the 3 waist joints")
        sp.add_argument("--no-hands", action="store_true", help="arms only (no Dex3)")
        sp.add_argument("--max-vel", type=float, default=1.0, help="rad/s cap in the robot driver")
        sp.add_argument("--hand-kp-scale", type=float, default=1.0)

    c = sub.add_parser("check-robot", help="read joint state and cameras once; publishes nothing")
    robot_args(c)

    r = sub.add_parser("run", help="run a policy")
    robot_args(r)
    r.add_argument("--server", required=True, help="policy server <ip>:<port>")
    r.add_argument("--strategy", default="auto")
    r.add_argument("--hz", type=float, default=None, help="control rate (default: the model's training rate)")
    r.add_argument("--task", default=None, help="language instruction for VLA models")
    r.add_argument("--duration", type=float, default=60.0)
    r.add_argument("--timeout", type=float, default=2.0)
    r.add_argument("--map-camera", action="append", default=[], metavar="POLICY_KEY=ROBOT_CAM",
                   help="when the policy's camera names differ from the robot's")
    r.add_argument("--dry-run", action="store_true", help="compute and log actions, never command")
    r.add_argument("--allow-far-start", action="store_true")
    r.add_argument("--max-step", type=float, default=0.05, help="rad per control tick")
    r.add_argument("--log-dir", default="logs")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    if args.cmd == "describe":
        cl = PolicyClient(args.server)
        print(json.dumps(cl.describe().to_dict(), indent=2))
        print(f"ping {cl.ping() * 1e3:.1f} ms")
        cl.close()
        return 0

    if args.cmd == "check-robot":
        robot = _robot(args)
        if args.robot == "unitree":
            robot.connect(read_only=True)
        else:
            robot.connect()
        obs = robot.observe()
        for n, q in zip(robot.joint_names, obs.state):
            print(f"{n:34s} {q:+.3f}")
        for k, im in obs.images.items():
            print(f"camera {k}: {im.shape} {im.dtype}")
        return 0

    robot = _robot(args)
    cfg = AgentConfig(server=args.server, strategy=args.strategy, control_hz=args.hz, task=args.task,
                      duration_s=args.duration, timeout_s=args.timeout,
                      camera_map=dict(m.split("=", 1) for m in args.map_camera), dry_run=args.dry_run,
                      allow_far_start=args.allow_far_start, max_step=args.max_step, log_dir=args.log_dir)
    agent = Agent(robot, cfg)
    try:
        agent.start()
        summary = agent.run()
    finally:
        robot.close()
    print(json.dumps(summary, indent=2))
    return 0 if "safety_stop" not in summary else 2


if __name__ == "__main__":
    sys.exit(main())
