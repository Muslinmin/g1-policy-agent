# Training

Training uses LeRobot (0.4.1) on a LeRobot-format dataset of G1 demonstrations. The trained checkpoint folder
(`<out>/checkpoints/last/pretrained_model`) is what the server loads. Nothing else needs exporting.

## 1. The dataset has to match the agent

The agent sends `observation.state` and executes `action` in the joint order of `g1agent/robot/g1_layout.py`:
14 arm joints (left then right: shoulder pitch, roll, yaw, elbow, wrist roll, pitch, yaw), then 14 Dex3 joints
(left then right: thumb_0, thumb_1, thumb_2, index_0, index_1, middle_0, middle_1), optionally the 3 waist joints
after the arms. Actions are **absolute joint targets** at the dataset fps. Camera features are
`observation.images.<name>`.

```bash
python train/check_dataset.py --root data/g1_task --camera head
```

It refuses a dataset whose state or action size or joint order differs from the agent's, or that lacks a camera
or the fps.

## 2. Train

```bash
python train/train.py --policy act       --repo-id me/g1_task --root data/g1_task --out outputs/act_g1
python train/train.py --policy diffusion --repo-id me/g1_task --root data/g1_task --out outputs/dp_g1
python train/train.py --policy pi05      --repo-id me/g1_task --root data/g1_task --out outputs/pi05_g1
python train/train.py --policy smolvla   ...
python train/train.py --policy groot     ...      # needs the Isaac-GR00T dependencies
# override anything after `--`:
python train/train.py --policy act ... -- --batch_size=16 --steps=50000
# see the command without running it:
python train/train.py --policy act ... --print
```

| Policy | Preset (starting point) | Language task |
|---|---|---|
| act | chunk 100, lr 3e-5, batch 40, 100k steps | no |
| diffusion | 2 obs frames, horizon 16, 8 executed, batch 64, 100k steps | no |
| pi05 / pi0 | fine-tune `lerobot/pi05_base` / `pi0_base`, batch 32, 30k steps | yes |
| smolvla | fine-tune `lerobot/smolvla_base`, batch 64, 20k steps | yes |
| groot | LeRobot `groot` policy, batch 32, 20k steps | yes |

The presets come from each method's reference recipe and have not been tuned on a G1 task. VLA fine-tuning (pi0, pi0.5, SmolVLA,
GR00T) needs a large GPU: if memory runs out, lower the batch size and check LeRobot's docs for that policy.

## 3. Sim + real co-training (optional)

If the dataset merges real demos with simulated ones (e.g. MimicGen or Isaac Lab data recorded in the same
format), train with a fixed **sim share per batch**, alpha:

```bash
python train/make_mapping.py --root data/g1_merged --ranges real:0-49 sim:50-399 -o mappings/g1_merged.json
python train/train.py --policy act --repo-id me/g1_merged --root data/g1_merged --out outputs/act_a090 \
       --alpha-sim 0.9 --mapping mappings/g1_merged.json
```

alpha is the probability that a training sample is drawn from sim (Maddukuri et al. 2025). It fixes the mix of
every batch regardless of how many sim and real episodes there are, so the number of sim episodes and the mix can
be studied separately. The sampler logs the realised sim fraction at start-up and stops if it is more than 0.01 off.
It is the only change to stock `lerobot-train` (`train/train_weighted.py`). It does not support Diffusion Policy
(its episode-aware sampler would need the same change).

## 4. Serve it

```bash
python -m g1server --backend lerobot --ckpt outputs/act_g1/checkpoints/last/pretrained_model --port 5555
```

The server reads the policy type, chunk length, observation history, cameras and fps from the checkpoint
(the fps from the training dataset's `meta/info.json`, else pass `--hz`).

## Comparing models

Compare checkpoints trained with the same batch size, number of steps and GPU, and evaluate them with the same
agent settings (strategy, control rate) and the same set of start conditions.
