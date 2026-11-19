# g1-policy-agent

Run imitation-learning policies (ACT, Diffusion Policy, pi0 / pi0.5, SmolVLA, GR00T) on a **Unitree G1**.

* **The agent runs on the robot.** It reads joints and cameras, asks a policy server for actions, decides how to
  execute the action chunks, and commands the arms and Dex3 hands. Unitree's own controller keeps the legs.
* **The model runs on a GPU machine** behind one address, `IP:port`.
* **Training** uses LeRobot, with presets per model and optional sim + real co-training at a fixed sim share.

The agent does not need to know whether a model predicts chunks or how long they are. It asks the server, probes
if the server does not say, measures the round trip, and picks a strategy: temporal ensembling (ACT), receding
horizon (Diffusion Policy), async prefetch (VLAs, slow links) or single step.

```
robot (G1 dev computer)                                GPU machine
  g1-agent run --server 10.0.0.5:5555  ── tcp ──▶  python -m g1server --backend lerobot --ckpt <ckpt> --port 5555
```

## Quick start

```bash
pip install -e ".[dev]"
# terminal 1: a model-free server
python -m g1server --backend dummy --port 5555
# terminal 2: the agent against a simulated G1 (no hardware)
python -m g1agent run --server 127.0.0.1:5555 --robot mock --duration 10
python -m pytest -q          # 30 tests: protocol, chunking, the agent end to end over ZeroMQ
```

On the real robot: follow **docs/G1_SETUP.md** step by step (read-only check, dry run, first run).

## Documents

| | |
|---|---|
| [MANIFEST.md](MANIFEST.md) | every file, what it does, what is tested |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | the G1 side, the agent, the server, the protocol, timing |
| [docs/CHUNKING.md](docs/CHUNKING.md) | how chunks are executed and how the strategy is chosen |
| [docs/G1_SETUP.md](docs/G1_SETUP.md) | install, checks and first run on a G1 |
| [docs/TRAINING.md](docs/TRAINING.md) | dataset format, training presets, co-training, serving |

## Status

* Tested (CPU, no robot): protocol, all four chunking strategies, strategy selection, the agent end to end against
  a dummy server for each model family (including a hidden chunk length, 40-60 ms latency, safety stops,
  dimension mismatches), the training launcher and dataset checker. The LeRobot backend was run on a real ACT
  checkpoint and a Diffusion Policy model (observation history path).
* **Not yet run on a G1:** the Unitree driver (`g1agent/robot/unitree_g1.py`). It follows Unitree's arm_sdk example
  and a working G1 deployment's conventions. Use the staged first run in docs/G1_SETUP.md.
* Not tested here: pi0 / pi0.5 / SmolVLA / GR00T checkpoints through the LeRobot backend (same code path as ACT;
  they also need a `--task`).
