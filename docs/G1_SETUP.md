# G1 setup and first run

**The Unitree driver (`g1agent/robot/unitree_g1.py`) follows Unitree's official arm_sdk example and the
conventions of a working G1 deployment, but it has not been run on a robot from this repository.** Go through the
steps below in order, with someone at the e-stop.

## 1. Install on the robot computer

```bash
# unitree_sdk2_python (DDS bindings) from source
git clone https://github.com/unitreerobotics/unitree_sdk2_python && cd unitree_sdk2_python && pip install -e . && cd ..
git clone <this repo> && cd g1-policy-agent
pip install -e ".[robot]"          # numpy, pyzmq, msgpack, pyrealsense2, opencv
```

The robot computer does **not** need torch or the model.

## 2. Install on the GPU machine

```bash
pip install -e ".[server]"         # + lerobot 0.4.1 (and its torch)
python -m g1server --backend dummy --port 5555          # model-free, for the first runs
```

Open the port in the GPU machine's firewall. The robot reaches it at `<gpu-ip>:5555`.

## 3. Check the link (nothing moves)

```bash
g1-agent describe --server <gpu-ip>:5555                 # model spec + ping
```

## 4. Check the robot (nothing moves)

Run Unitree's own example first, to confirm your SDK, interface and robot mode:
`python unitree_sdk2_python/example/g1/high_level/g1_arm7_sdk_dds_example.py eth0` (it moves the arms).

Then, read-only (publishes nothing):

```bash
g1-agent check-robot --iface eth0 --camera head=realsense:<serial>
```

Every joint should read a plausible angle (rad), and both hands and every camera should report. If a hand
topic is missing, run with `--no-hands`. If your G1 is a 23-DoF version, edit `g1_layout.py` first (no wrist
pitch/yaw).

## 5. Dry run against the real robot

```bash
g1-agent run --server <gpu-ip>:5555 --robot unitree --iface eth0 --camera head=realsense:<serial> \
             --duration 20 --dry-run
```

The driver takes the arms (the arm_sdk weight ramps to 1 over 2 s) and **holds the current posture**. The policy
runs and is logged, but nothing is commanded. Check the summary: `achieved_hz` close to the model's rate,
`starved` about 0 after the first chunk, `errors` 0, `clipped_ticks` 0. On exit the weight ramps back to 0.

## 6. First real run

Start the robot near the posture the demonstrations started from. The first-action guard refuses to start if the
policy's first target is more than 0.35 rad from the measured posture on any joint.

```bash
g1-agent run --server <gpu-ip>:5555 --robot unitree --iface eth0 --camera head=realsense:<serial> \
             --duration 30 --max-vel 0.5 --task "pick up the cup"
```

Ctrl-C stops: hold, then hand back to Unitree's controller. Raise `--max-vel` (rad/s, driver) and `--max-step`
(rad per tick, safety) only after watching a few runs.

## Hand gains

Real Dex3 fingers have static friction: with the default gains they may barely move. `--hand-kp-scale 2` to `3`
helps; above that, watch for fingers pushing hard against objects.

## Things that differ between G1s

| Check | Where to change |
|---|---|
| 23-DoF vs 29-DoF (wrist pitch/yaw, waist) | `g1_layout.py`: `ARM_JOINTS`, `ARM_MOTOR_INDEX`, `LIMITS` |
| hand model (Dex3, Inspire, none) | `--no-hands`, or a new hand writer in `unitree_g1.py` |
| network interface name | `--iface` |
| DDS domain | `UnitreeG1(domain=...)` |
| gains | `ARM_KP/KD`, `HAND_KP/KD` in `g1_layout.py` |
