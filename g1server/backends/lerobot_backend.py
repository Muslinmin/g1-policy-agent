"""Serve any LeRobot policy checkpoint (``.../pretrained_model``): act, diffusion, pi0, pi05, smolvla, groot.

The policy type is read from the checkpoint's config.json, the pre/post-processors (normalisation, tokenizer)
from the same folder, so the server needs nothing but the folder. Images are resized to the training size.

Observation history: if the policy takes ``n_obs_steps > 1`` frames (Diffusion Policy: 2), the agent sends that
many consecutive frames and this backend refills the policy's queues from them on every call, so the server
stays stateless (a policy's own ``select_action`` keeps that history inside the process instead).

Extra state features (e.g. an end-effector pose a model was trained with) must come from the robot side in
``frame["extra"][<feature name>]``; anything the checkpoint needs and the request lacks is reported by name.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np

from g1agent.profiles import LEROBOT_TYPE_TO_FAMILY
from g1agent.spec import PolicySpec

from .base import Backend

log = logging.getLogger(__name__)
IMG_PREFIX = "observation.images."


class LeRobotBackend(Backend):
    def __init__(self, ckpt: str, device: str = "cuda", hz: Optional[float] = None) -> None:
        import torch
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.factory import get_policy_class, make_pre_post_processors

        self.torch, self.device = torch, torch.device(device)
        self.ckpt = Path(ckpt)
        cfg = PreTrainedConfig.from_pretrained(str(self.ckpt))
        self.type = cfg.type
        self.policy = get_policy_class(cfg.type).from_pretrained(str(self.ckpt)).to(self.device).eval()
        ov = {"device": device}
        self.pre, self.post = make_pre_post_processors(
            self.policy.config, pretrained_path=str(self.ckpt),
            preprocessor_overrides={"device_processor": ov}, postprocessor_overrides={"device_processor": ov})
        c = self.policy.config
        feats = c.input_features
        self.image_hw = {k[len(IMG_PREFIX):]: list(v.shape[1:]) for k, v in feats.items()
                         if k.startswith(IMG_PREFIX)}
        self.state_dim = feats["observation.state"].shape[0] if "observation.state" in feats else None
        self.extra_keys = [k for k in feats if not k.startswith(IMG_PREFIX) and k != "observation.state"]
        self.action_dim = c.output_features["action"].shape[0]
        self.n_obs = int(getattr(c, "n_obs_steps", 1) or 1)
        self.chunk = int(getattr(c, "chunk_size", None) or getattr(c, "horizon", None) or 1)
        self.n_action_steps = getattr(c, "n_action_steps", None)
        meta = self._dataset_meta()
        self.hz = float(hz or meta.get("fps") or 0) or None
        names = (meta.get("features", {}).get("observation.state", {}) or {}).get("names")
        anames = (meta.get("features", {}).get("action", {}) or {}).get("names")
        self.state_names = list(names) if isinstance(names, list) else []
        self.action_names = list(anames) if isinstance(anames, list) else []
        self.family = LEROBOT_TYPE_TO_FAMILY.get(self.type, "unknown")
        self.needs_task = self.family in ("pi0", "pi05", "smolvla", "groot")
        log.info("loaded %s (%s): chunk %d, executes %s, %d obs frame(s), state %s-D, action %d-D, cameras %s, "
                 "%s Hz", self.ckpt, self.type, self.chunk, self.n_action_steps, self.n_obs, self.state_dim,
                 self.action_dim, sorted(self.image_hw), self.hz)

    def _dataset_meta(self) -> dict:
        try:
            tc = json.loads((self.ckpt / "train_config.json").read_text())
            return json.loads((Path(tc["dataset"]["root"]) / "meta" / "info.json").read_text())
        except (OSError, KeyError, TypeError, json.JSONDecodeError):
            log.warning("training dataset meta/info.json not found next to the checkpoint: pass --hz; joint "
                        "names will not be checked")
            return {}

    def spec(self) -> PolicySpec:
        return PolicySpec(model=str(self.ckpt), family=self.family, control_hz=self.hz, chunk_size=self.chunk,
                          n_action_steps=self.n_action_steps, n_obs_steps=self.n_obs, state_dim=self.state_dim,
                          action_dim=self.action_dim, state_names=self.state_names,
                          action_names=self.action_names, image_keys=sorted(self.image_hw),
                          image_hw=self.image_hw, needs_task=self.needs_task)

    def reset(self) -> None:
        self.policy.reset()

    def warm_up(self) -> None:
        """Also measures the chunk length predict_action_chunk really returns (Diffusion Policy and GR00T
        return the executable n_action_steps, not the full horizon), so ``describe`` reports what ``infer``
        gives."""
        s = self.spec()
        frame = {"state": np.zeros(s.state_dim or 1, np.float32),
                 "images": {k: np.zeros((*s.image_hw[k], 3), np.uint8) for k in s.image_keys}, "t": 0.0,
                 "extra": {k: np.zeros(self.policy.config.input_features[k].shape, np.float32)
                           for k in self.extra_keys}}
        for _ in range(2):
            a = self.infer([frame] * self.n_obs, "warm up" if self.needs_task else None)
        if len(a) != self.chunk:
            log.info("config chunk %d, predict_action_chunk returns %d: reporting %d", self.chunk, len(a), len(a))
            self.chunk = len(a)

    def _obs(self, f: dict) -> dict:
        from PIL import Image
        obs = {}
        if self.state_dim is not None:
            obs["observation.state"] = np.asarray(f["state"], np.float32).copy()
        for k in self.extra_keys:
            if k not in (f.get("extra") or {}):
                raise ValueError(f"checkpoint needs '{k}', send it in frame['extra']['{k}']")
            obs[k] = np.asarray(f["extra"][k], np.float32).copy()
        for name, (h, w) in self.image_hw.items():
            img = f["images"].get(name)
            if img is None:
                raise ValueError(f"no image '{name}' in the request (have {sorted(f['images'])})")
            if img.shape[:2] != (h, w):
                img = np.asarray(Image.fromarray(img).resize((w, h), Image.BILINEAR))
            obs[IMG_PREFIX + name] = np.ascontiguousarray(img)
        return obs

    def infer(self, frames: list[dict], task: Optional[str]) -> np.ndarray:
        from lerobot.policies.utils import populate_queues, prepare_observation_for_inference
        from lerobot.utils.constants import ACTION, OBS_IMAGES

        torch = self.torch
        with torch.inference_mode():
            if self.n_obs > 1:
                if len(frames) < self.n_obs:
                    frames = [frames[0]] * (self.n_obs - len(frames)) + list(frames)
                self.policy.reset()
                for f in frames[-self.n_obs:]:
                    batch = dict(self.pre(prepare_observation_for_inference(self._obs(f), self.device, task)))
                    batch.pop(ACTION, None)      # as the policy's own select_action does
                    if self.policy.config.image_features:
                        batch[OBS_IMAGES] = torch.stack([batch[k] for k in self.policy.config.image_features],
                                                        dim=-4)
                    self.policy._queues = populate_queues(self.policy._queues, batch)
            else:
                batch = self.pre(prepare_observation_for_inference(self._obs(frames[-1]), self.device, task))
            chunk = self.policy.predict_action_chunk(batch)            # (1, T, D), normalised
            chunk = torch.stack([self.post(chunk[:, i]) for i in range(chunk.shape[1])], dim=1)
        return chunk[0].float().cpu().numpy()
