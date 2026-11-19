#!/usr/bin/env python
"""Train a policy on a LeRobot dataset of G1 demonstrations, with presets per model family.

    python train/train.py --policy act       --repo-id me/g1_task --root data/g1_task --out outputs/act_g1
    python train/train.py --policy diffusion --repo-id me/g1_task --root data/g1_task --out outputs/dp_g1
    python train/train.py --policy pi05      --repo-id me/g1_task --root data/g1_task --out outputs/pi05_g1
    # sim + real co-training at a fixed sim share per batch (not for diffusion):
    python train/train.py --policy act ... --alpha-sim 0.9 --mapping mappings/g1_task_mapping.json
    # anything after `--` goes to lerobot-train unchanged, overriding the preset:
    python train/train.py --policy act ... -- --batch_size=16 --steps=50000
    python train/train.py --policy act ... --print      # show the command, run nothing

The checkpoint to serve is <out>/checkpoints/last/pretrained_model (g1server --backend lerobot --ckpt ...).
Presets are starting points from each method's reference recipe, not tuned for any G1 task (docs/TRAINING.md).
Compare models only at a fixed batch size, step count and GPU.
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

PRESETS: dict[str, list[str]] = {
    # ACT (Zhao et al. 2023): chunk 100, executed with temporal ensembling by the agent.
    "act": ["--policy.type=act", "--policy.chunk_size=100", "--policy.n_action_steps=100",
            "--policy.optimizer_lr=3e-5", "--batch_size=40", "--steps=100000", "--save_freq=20000"],
    # Diffusion Policy (Chi et al. 2023): 2 obs frames, horizon 16, 8 executed.
    "diffusion": ["--policy.type=diffusion", "--policy.n_obs_steps=2", "--policy.horizon=16",
                  "--policy.n_action_steps=8", "--batch_size=64", "--steps=100000", "--save_freq=20000"],
    # pi0.5 (Physical Intelligence), fine-tuned from the LeRobot port of the base model. Needs a task string.
    "pi05": ["--policy.path=lerobot/pi05_base", "--batch_size=32", "--steps=30000", "--save_freq=5000"],
    "pi0": ["--policy.path=lerobot/pi0_base", "--batch_size=32", "--steps=30000", "--save_freq=5000"],
    # SmolVLA (LeRobot), fine-tuned from its base model.
    "smolvla": ["--policy.path=lerobot/smolvla_base", "--batch_size=64", "--steps=20000", "--save_freq=5000"],
    # GR00T N1.x through LeRobot's groot policy (needs the Isaac-GR00T dependencies installed).
    "groot": ["--policy.type=groot", "--batch_size=32", "--steps=20000", "--save_freq=5000"],
}


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    extra: list[str] = []
    if "--" in argv:
        i = argv.index("--")
        argv, extra = argv[:i], argv[i + 1:]
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--policy", required=True, choices=sorted(PRESETS))
    p.add_argument("--repo-id", required=True, help="dataset repo id (name only if it lives in --root)")
    p.add_argument("--root", default=None, help="local dataset folder")
    p.add_argument("--out", required=True, help="output dir")
    p.add_argument("--alpha-sim", type=float, default=None, help="sim share per batch (co-training)")
    p.add_argument("--mapping", default=None, help="episode -> source JSON (train/make_mapping.py)")
    p.add_argument("--exclude-episodes", type=int, nargs="*", default=[])
    p.add_argument("--seed", type=int, default=1000)
    p.add_argument("--device", default="cuda")
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--print", action="store_true", help="print the command and exit")
    a = p.parse_args(argv)

    lr = [f"--dataset.repo_id={a.repo_id}", f"--output_dir={a.out}", f"--job_name={Path(a.out).name}",
          f"--seed={a.seed}", f"--policy.device={a.device}", "--policy.push_to_hub=false",
          f"--wandb.enable={'true' if a.wandb else 'false'}", *PRESETS[a.policy]]
    if a.root:
        lr.append(f"--dataset.root={a.root}")
    lr += extra

    if a.alpha_sim is not None:
        if a.policy == "diffusion":
            raise SystemExit("--alpha-sim is not supported for diffusion (EpisodeAwareSampler); see docs/TRAINING.md")
        if a.mapping is None:
            raise SystemExit("--alpha-sim needs --mapping (which episodes are sim and which are real)")
        own = ["--alpha-sim", str(a.alpha_sim), "--slot-mapping", a.mapping]
        if a.exclude_episodes:
            own += ["--exclude-episodes", *map(str, a.exclude_episodes)]
        cmd = [sys.executable, str(HERE / "train_weighted.py"), *own, *lr]
    else:
        if a.exclude_episodes:
            raise SystemExit("--exclude-episodes needs the weighted sampler: pass --alpha-sim too")
        cmd = [sys.executable, "-m", "lerobot.scripts.lerobot_train", *lr]

    print(" ".join(shlex.quote(c) for c in cmd), flush=True)
    if a.print:
        return 0
    return subprocess.call(cmd, env=os.environ.copy())


if __name__ == "__main__":
    sys.exit(main())
