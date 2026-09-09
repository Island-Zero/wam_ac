"""Portable RoboTwin policy implementing the frozen base Adaptive 2/10 router."""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import time
import types
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .kinematics import execution_position_mean_m


HERE = Path(__file__).resolve()
FASTWAM_ROOT = Path(os.environ.get(
    "FASTWAM_ROOT", HERE.parents[4] / "repos/FastWAM"
)).expanduser().resolve()
BASE_PATH = FASTWAM_ROOT / "experiments/robotwin/fastwam_policy/deploy_policy.py"
if not BASE_PATH.is_file():
    raise FileNotFoundError(
        f"Original Fast-WAM policy not found at {BASE_PATH}. Set FASTWAM_ROOT."
    )
SPEC = importlib.util.spec_from_file_location("_base_adaptive_upstream_policy", BASE_PATH)
if SPEC is None or SPEC.loader is None:
    raise ImportError(BASE_PATH)
BASE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = BASE
SPEC.loader.exec_module(BASE)


class AdaptiveRouter:
    """Intercept Fast-WAM action sampling without changing model parameters."""

    def __init__(self, model, denormalize):
        self.model = model
        self.denormalize = denormalize
        self.threshold = float(os.environ.get(
            "WAM_ADAPTIVE_THRESHOLD_M", "0.009702832328772866"))
        self.urdf = Path(os.environ.get("WAM_ADAPTIVE_URDF", "")).expanduser().resolve()
        if not self.urdf.is_file():
            raise FileNotFoundError(f"Set WAM_ADAPTIVE_URDF to the RoboTwin Aloha URDF: {self.urdf}")
        self.original_infer = model.infer_action
        self.original_pre = model.action_expert.pre_dit
        self.original_mot = model.mot.forward_action_with_video_cache
        self.original_post = model.action_expert.post_dit
        self.original_predict = model._predict_action_noise_with_cache
        self.reset_episode()
        self._install()

    def reset_episode(self):
        self.chunk_records: list[dict[str, Any]] = []
        self._reset_capture()

    def _reset_capture(self):
        self.initial_noise = None
        self.context = None
        self.context_mask = None
        self.mot_template = None
        self.velocities = []
        self.capturing = False

    def _capture_pre(self, action_tokens, context, context_mask):
        if self.capturing and self.initial_noise is None:
            self.initial_noise = action_tokens.detach().clone()
            self.context, self.context_mask = context, context_mask

    def _capture_mot(self, kwargs):
        if self.capturing and self.mot_template is None:
            self.mot_template = {k: kwargs[k] for k in
                                 ("video_kv_cache", "attention_mask", "video_seq_len")}

    @torch.no_grad()
    def _action_forward(self, z, timestep):
        if self.context is None or self.mot_template is None:
            raise RuntimeError("Incomplete Official-2 context capture")
        pre = self.original_pre(
            action_tokens=z,
            timestep=timestep.reshape(1).to(device=z.device, dtype=z.dtype),
            context=self.context,
            context_mask=self.context_mask,
        )
        hidden = self.original_mot(
            action_tokens=pre["tokens"], action_freqs=pre["freqs"], action_t_mod=pre["t_mod"],
            action_context_payload={"context": pre["context"], "mask": pre["context_mask"]},
            video_kv_cache=self.mot_template["video_kv_cache"],
            attention_mask=self.mot_template["attention_mask"],
            video_seq_len=self.mot_template["video_seq_len"],
        )
        return self.original_post(hidden, pre)

    def _schedule(self, k, device, dtype):
        timesteps, deltas = self.model.infer_action_scheduler.build_inference_schedule(k, device, dtype)
        if len(timesteps) != k or len(deltas) != k:
            raise AssertionError(f"Official-{k} schedule returned {len(timesteps)} evaluations")
        return timesteps, deltas

    @torch.no_grad()
    def _route(self, official2_cpu):
        if self.initial_noise is None or len(self.velocities) != 2:
            raise RuntimeError(f"Official-2 capture failed: velocities={len(self.velocities)}")
        noise = self.initial_noise
        k1 = self.velocities[0]
        official2 = official2_cpu.to(device=noise.device, dtype=noise.dtype)
        if official2.ndim == 2:
            official2 = official2.unsqueeze(0)
        coarse = noise - k1
        coarse_qpos = self.denormalize(coarse[0].cpu().float())[0]
        two_qpos = self.denormalize(official2[0].cpu().float())[0]
        eta = execution_position_mean_m(coarse_qpos, two_qpos, self.urdf, 24)
        accept = eta <= self.threshold
        selected, actual_nfe = official2, 2
        if not accept:
            timesteps, deltas = self._schedule(10, noise.device, noise.dtype)
            # Reuse the identical sigma=1,z0 velocity. Official-2's second
            # velocity is at sigma=5/6 and is not part of Official-10.
            z = self.model.infer_action_scheduler.step(k1, deltas[0], noise)
            for timestep, delta in zip(timesteps[1:], deltas[1:]):
                velocity = self._action_forward(z, timestep)
                z = self.model.infer_action_scheduler.step(velocity, delta, z)
            selected, actual_nfe = z, 11
        self.chunk_records.append({
            "chunk": len(self.chunk_records),
            "eta_position_mean_m": eta,
            "threshold_m": self.threshold,
            "decision": "official2" if accept else "full10_fallback",
            "actual_action_dit_nfe": actual_nfe,
        })
        print(
            f"[base-adaptive-2/10] chunk={len(self.chunk_records)-1} "
            f"eta_mm={eta*1000:.4f} decision={self.chunk_records[-1]['decision']} nfe={actual_nfe}",
            flush=True,
        )
        return selected[0].detach().cpu().float()

    def _install(self):
        router = self

        def infer_bound(_model, *args, **kwargs):
            if int(kwargs.get("num_inference_steps", 10)) != 10:
                raise RuntimeError("Entry point must request Official-10; router owns the Official-2 probe")
            router._reset_capture()
            kwargs["num_inference_steps"] = 2
            router.capturing = True
            result = router.original_infer(*args, **kwargs)
            router.capturing = False
            result["action"] = router._route(result["action"])
            return result

        def pre_bound(_expert, *args, **kwargs):
            result = router.original_pre(*args, **kwargs)
            router._capture_pre(kwargs["action_tokens"], kwargs["context"], kwargs.get("context_mask"))
            return result

        def mot_bound(_mot, *args, **kwargs):
            router._capture_mot(kwargs)
            return router.original_mot(*args, **kwargs)

        def predict_bound(_model, *args, **kwargs):
            value = router.original_predict(*args, **kwargs)
            if router.capturing:
                router.velocities.append(value.detach().clone())
            return value

        self.model.infer_action = types.MethodType(infer_bound, self.model)
        self.model.action_expert.pre_dit = types.MethodType(pre_bound, self.model.action_expert)
        self.model.mot.forward_action_with_video_cache = types.MethodType(mot_bound, self.model.mot)
        self.model._predict_action_noise_with_cache = types.MethodType(predict_bound, self.model)


class AdaptivePolicy(BASE.WorldActionRobotWinPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.action_horizon != 32 or self.replan_steps != 24:
            raise ValueError(f"Frozen protocol requires H=32,E=24; got H={self.action_horizon},E={self.replan_steps}")
        self.adaptive_router = AdaptiveRouter(self.model, self._denormalize_action)

    def reset(self):
        super().reset()
        if hasattr(self, "adaptive_router"):
            self.adaptive_router.reset_episode()

    def record_episode_result(self, *, task: str, setting: str, seed: int,
                              success: bool, environment_steps: int):
        path = os.environ.get("WAM_ADAPTIVE_EPISODE_RECORDS")
        if not path:
            return
        chunks = list(self.adaptive_router.chunk_records)
        row = {
            "task": task, "setting": setting, "seed": int(seed), "success": bool(success),
            "environment_steps": int(environment_steps), "action_chunks": len(chunks),
            "chunk_actual_nfe": [x["actual_action_dit_nfe"] for x in chunks],
            "chunk_router": chunks,
        }
        destination = Path(path).expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


# Upstream get_model resolves this global class at call time.
BASE.WorldActionRobotWinPolicy = AdaptivePolicy
encode_obs = BASE.encode_obs
eval = BASE.eval
reset_model = BASE.reset_model


def get_model(usr_args):
    return BASE.get_model(usr_args)


def record_episode_result(model, **kwargs):
    model.record_episode_result(**kwargs)
