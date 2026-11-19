# Chunking: how the agent turns action chunks into motion

Most modern imitation policies predict a **chunk** of future actions per inference (ACT 100, Diffusion Policy
8 of 16, pi0/pi0.5 50, GR00T 16). How the chunk is executed matters as much as the model: execute too much
open-loop and the robot cannot react; replan too often over a slow link and it stutters.

The agent does not assume the model has chunks. At start-up:

1. `describe` asks the server for `family`, `chunk_size`, `n_action_steps`, `control_hz`.
2. If `chunk_size` is missing, one **probe** inference tells it (the length of the returned array). A length of 1
   means a single-step policy.
3. The **round trip** is measured with real inferences.
4. The **strategy** is chosen: operator override > chunk length > family default > latency checks.

## The four strategies

All of them index chunks by absolute tick: a chunk computed from the observation at tick `t0` has its action `i`
for tick `t0 + i`. A reply that arrives `L` ticks late skips its first `L` actions instead of replaying the past.

| Strategy | Query | What runs | Good for | Cost |
|---|---|---|---|---|
| `single` | every tick, blocking | `actions[0]` | single-step policies | loop runs at most at 1 / round trip |
| `receding:K` | when K actions are used, blocking | first K of each chunk | Diffusion Policy, GR00T on a fast link | one round trip pause per replan |
| `async:K` | `lead + margin` ticks before K run out, in the background | the current chunk while the next is computed; switch is cross-faded | VLAs (pi0, pi0.5, SmolVLA, GR00T), slow links | each chunk is open-loop for K ticks |
| `ensemble[:k]` | every tick (blocking if round trip < 0.8 period, else background) | weighted average of all chunks covering the tick, `w_i = exp(-k i)`, oldest first | ACT | one inference per tick |

## Family defaults (`g1agent/profiles.py`)

| Family | Default | From |
|---|---|---|
| act | ensemble, k = 0.01 | ACT paper (Zhao et al. 2023) |
| diffusion | receding, execute 8 of 16, 2 observation frames | Diffusion Policy (Chi et al. 2023) |
| pi0 / pi05 | async, execute 25 of 50 | openpi defaults; real-time chunking (Black et al. 2025) |
| smolvla | async, execute 25 of 50 | LeRobot async inference |
| groot | async, execute 8 of 16 | Isaac-GR00T N1.x action horizon 16 |
| unknown | receding if the round trip < half a period, else async with execute = half the chunk | |

Latency rules that override the default: blocking ensembling becomes background ensembling when the round trip is
above 0.8 of a period; receding becomes async above 2 periods; async raises `execute` above `lead + margin`.

## Choosing by hand

```bash
g1-agent run ... --strategy async:20      # execute 20 actions per chunk
g1-agent run ... --strategy receding:8
g1-agent run ... --strategy ensemble:0.02
g1-agent run ... --strategy single
```

How to tune with a dry run (`--dry-run` computes and logs, never moves):

* `starved` in the summary > 0 after the first chunk: chunks arrive too late. Raise `execute`, or get a faster
  link or GPU.
* jerky motion at chunk seams (look at `target` in the log): larger cross-fade (async), or ensembling for ACT.
* sluggish reaction to disturbances: lower `execute`.

## What is not here

* **Real-time chunking with inpainting** (Black et al. 2025): freezing the actions already committed while the model
  generates the rest. It needs support inside the model's sampler. The async strategy is its open-loop core with a
  cross-fade at the seam.
* **Speed-up / slow-down** of chunks to hide latency. The agent always plays at the training rate.
