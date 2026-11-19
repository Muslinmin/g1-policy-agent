# Manifest

What is in the repository, what each part is for, how it is run, and how far it has been tested.

## Components

| Component | Runs on | Entry point | Needs |
|---|---|---|---|
| Agent | G1 development computer | `g1-agent` / `python -m g1agent` | numpy, pyzmq, msgpack; unitree_sdk2_python, pyrealsense2 / opencv for the real robot |
| Policy server | GPU machine | `g1-server` / `python -m g1server` | numpy, pyzmq, msgpack; lerobot 0.4.1 + torch for real models |
| Training | GPU machine | `python train/train.py` | lerobot 0.4.1 |

## Files

| Path | What it does | Tested |
|---|---|---|
| `g1agent/protocol.py` | wire format: msgpack + numpy arrays, no pickle; message definitions | unit |
| `g1agent/spec.py` | `PolicySpec`: what the server reports about its model | via end-to-end |
| `g1agent/profiles.py` | chunking defaults per family (act, diffusion, pi0, pi05, smolvla, groot, unknown) | via selection tests |
| `g1agent/client.py` | ZeroMQ client (timeouts, socket reset on a lost reply) and the background requester | end-to-end |
| `g1agent/chunking/base.py` | `Chunk` (tick-indexed actions), `Strategy` interface | unit |
| `g1agent/chunking/strategies.py` | `SingleStep`, `Receding`, `AsyncChunking` (prefetch + cross-fade), `TemporalEnsemble` (ACT) | unit, with simulated latency |
| `g1agent/chunking/select.py` | picks the strategy: override, chunk length, family, latency rules | unit |
| `g1agent/agent.py` | start-up (describe, compatibility checks, probe, latency) and the fixed-rate control loop; per-tick JSONL log | end-to-end (mock robot) |
| `g1agent/cli.py` | `describe`, `check-robot` (read-only), `run` (mock or unitree, dry run) | smoke test (mock) |
| `g1agent/robot/base.py` | `Robot` interface, `Observation` | |
| `g1agent/robot/g1_layout.py` | G1 joint names, DDS motor indices, URDF limits, Dex3 slot order, gains | values checked against Unitree's example, the URDF and a working deployment |
| `g1agent/robot/safety.py` | first-action guard, joint limits, step limit, non-finite check | end-to-end |
| `g1agent/robot/mock.py` | G1-shaped robot without hardware | used by tests |
| `g1agent/robot/unitree_g1.py` | the real G1: rt/arm_sdk (weight ramp), rt/dex3 hands, 100 Hz rate-limited publisher, watchdog, read-only mode | **not run on hardware** |
| `g1agent/robot/cameras.py` | RealSense and OpenCV cameras in background threads | **not run on hardware** |
| `g1server/server.py` | REP server: ping, describe, reset, infer; errors returned, never fatal | end-to-end |
| `g1server/backends/base.py` | `Backend` interface, warm-up | |
| `g1server/backends/lerobot_backend.py` | any LeRobot checkpoint: policy, pre/post-processors, image resize, observation history, per-step un-normalisation, measured chunk length | real ACT checkpoint; random Diffusion Policy |
| `g1server/backends/dummy.py` | model-free sinusoid backend; can hide its chunk length, add latency, pose as any family | used by tests |
| `g1server/backends/custom_template.py` | template for non-LeRobot models | |
| `train/train.py` | training launcher with presets per policy; co-training via `--alpha-sim` | unit (command building) |
| `train/train_weighted.py` | stock `lerobot-train` with a per-frame weighted sim/real sampler | weight unit test; used for real ACT co-training runs |
| `train/make_mapping.py` | episode -> source mapping for co-training | |
| `train/check_dataset.py` | checks a dataset's state/action size, joint order, cameras and fps against the agent | unit |
| `tests/` | 30 tests (`python -m pytest -q`) | |
| `docs/` | architecture, chunking, G1 setup, training | |

## Interfaces that must agree

| Between | Contract | Defined in |
|---|---|---|
| dataset <-> agent | `observation.state` and `action`: the joint order of `g1_layout.py`, absolute targets, dataset fps | `g1_layout.py`, `train/check_dataset.py` |
| agent <-> server | protocol v1 (`ping`, `describe`, `reset`, `infer`), `actions[i]` for tick `t0 + i` | `g1agent/protocol.py` |
| server <-> model | LeRobot checkpoint folder (`config.json`, `model.safetensors`, processors, `train_config.json`) | `lerobot_backend.py` |
| agent <-> robot | arms via rt/arm_sdk (blend weight at motor 29), hands via rt/dex3/*/cmd, state via rt/lowstate and rt/dex3/*/state | `unitree_g1.py` |

## Safety layers, outside to inside

1. First-action guard (refuses a start more than 0.35 rad from the measured posture).
2. Joint limits from the URDF, minus a 0.02 rad margin.
3. Step limit per control tick (0.05 rad).
4. Driver rate limit (1 rad/s by default) at 100 Hz.
5. Driver watchdog: holds the posture if targets stop for 0.5 s.
6. No chunk for the current tick: hold.
7. Stop: hold, then ramp the arm_sdk weight back to Unitree's controller over 2 s.
8. The e-stop. None of the above replaces it.

## Known limits

* Upper body only (arms, hands, optional waist). Legs stay with Unitree's locomotion controller.
* Absolute joint-space actions only. Delta or end-effector actions must be converted in a custom backend.
* Co-training sampler: not for Diffusion Policy.
* Async chunking is open-loop with a cross-fade at chunk seams. It has no in-model inpainting (real-time chunking).
