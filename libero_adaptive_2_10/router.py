"""Exact native sampler wrapper with shared-condition 2/10 routing."""
from __future__ import annotations

import time
import torch
from metrics import action_readouts


class Router:
    def __init__(self, model):
        self.model = model
        self.native = model.infer_action
        self.predict = model._predict_action_noise_with_cache
        self.mot = model.mot.forward_action_with_video_cache
        model.infer_action = self.infer
        model._predict_action_noise_with_cache = self.capture_predict
        model.mot.forward_action_with_video_cache = self.count_mot
        self.forward_count = 0
        self.capture = False
        self.audit_done = False
        self.rows, self.tensors = [], []

    def configure(self, method, execution_horizon=10):
        self.method, self.e = method, execution_horizon
        self.rows, self.tensors = [], []

    def count_mot(self, *args, **kwargs):
        self.forward_count += 1
        return self.mot(*args, **kwargs)

    def capture_predict(self, *args, **kwargs):
        value = self.predict(*args, **kwargs)
        if self.capture and self.first is None:
            self.first = dict(kwargs)
            self.first["latents_action"] = kwargs["latents_action"].detach().clone()
            self.v0 = value.detach().clone()
        return value

    def schedule(self, k):
        z = self.first["latents_action"]
        return self.model.infer_action_scheduler.build_inference_schedule(
            k, z.device, z.dtype, shift_override=self.shift)

    def replay(self, k, reuse=True):
        z = self.first["latents_action"].clone()
        ts, ds = self.schedule(k)
        for i, (t, d) in enumerate(zip(ts, ds)):
            if i == 0 and reuse:
                v = self.v0
            else:
                payload = dict(self.first, latents_action=z, timestep_action=t.reshape(1))
                v = self.model._predict_action_noise_with_cache(**payload)
            z = self.model.infer_action_scheduler.step(v, d, z)
        return z[0].detach().cpu().float()

    @torch.no_grad()
    def infer(self, *args, **kwargs):
        if args:
            raise ValueError("Expected keyword-only native inference")
        mode = self.method["mode"]
        self.first, self.v0 = None, None
        self.shift = kwargs.get("sigma_shift")
        self.forward_count = 0
        torch.cuda.synchronize()
        start = time.perf_counter()
        native_k = self.method.get("nfe", 2) if mode == "fixed" else 2
        self.capture = mode != "fixed"
        try:
            result = self.native(**dict(kwargs, num_inference_steps=native_k))
        finally:
            self.capture = False
        if self.forward_count != native_k:
            raise AssertionError(f"Native {native_k} != counted {self.forward_count}")
        row = {"chunk": len(self.rows), "mode": mode, "actual_nfe": native_k}
        if mode != "fixed":
            z = self.first["latents_action"]
            if tuple(z.shape) != (1, 32, 7):
                raise AssertionError(f"Unexpected action shape {z.shape}")
            coarse = (z - self.v0)[0].detach().cpu().float()
            a2 = result["action"].clone()
            dc, d2 = coarse.numpy(), a2.cpu().numpy()
            measure = lambda a, b: action_readouts(a, b, self.e)
            values = measure(dc, d2)
            row.update(values)
            if mode == "shadow":
                a10 = self.replay(10)
                if self.forward_count != 11:
                    raise AssertionError("Shared A2/A10 must use exactly 11 forwards")
                d10 = a10.numpy()
                row.update({"residual_" + k: v for k, v in measure(d2, d10).items()})
                row["decision"] = "teacher10"
                row["actual_nfe"] = 11
                self.tensors.append({"coarse": dc, "a2": d2, "a10": d10})
                if not self.audit_done:
                    count_before = self.forward_count
                    replica2 = self.replay(2, reuse=False)
                    replica10 = self.replay(10, reuse=False)
                    native10 = self.native(**dict(kwargs, num_inference_steps=10))["action"]
                    audit = {
                        "a2_replay_max_abs": float((replica2-a2).abs().max()),
                        "a10_reuse_max_abs": float((replica10-a10).abs().max()),
                        "a10_native_max_abs": float((native10-a10).abs().max()),
                        "diagnostic_forwards": self.forward_count-count_before,
                        "schedule": {str(k): {"timesteps": self.schedule(k)[0].float().cpu().tolist(),
                                               "deltas": self.schedule(k)[1].float().cpu().tolist()}
                                     for k in (2,4,10)},
                    }
                    if any(audit[k] != 0 for k in ("a2_replay_max_abs", "a10_reuse_max_abs", "a10_native_max_abs")):
                        raise AssertionError(f"Sampler equivalence failed: {audit}")
                    row["sampler_audit"] = audit
                    self.audit_done = True
                result["action"] = a10
            elif mode == "adaptive":
                accepted = values[self.method["metric"]] <= self.method["threshold"]
                row["score"] = values[self.method["metric"]]
                row["decision"] = "accept_2_step" if accepted else "fallback_10_step"
                row["actual_nfe"] = 2 if accepted else 11
                if not accepted:
                    result["action"] = self.replay(10)
                if self.forward_count != row["actual_nfe"]:
                    raise AssertionError("Adaptive actual forward count mismatch")
            else:
                raise ValueError(mode)
        torch.cuda.synchronize()
        row["total_counted_forwards"] = self.forward_count
        row["decision_ms_concurrent"] = (time.perf_counter()-start)*1000
        self.rows.append(row)
        self.first, self.v0 = None, None
        return result
