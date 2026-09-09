#!/usr/bin/env python3
"""Run paired Full-10 and Adaptive 2/10 RoboTwin clean evaluation on one GPU."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
from pathlib import Path


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
METHOD = PROJECT / "code/base_adaptive_2_10"
DEFAULT_MANIFEST = HERE / "manifests/clean_50tasks_20seeds_v1.json"
ANSI = re.compile(r"\x1b\[[0-9;]*m")


def parse_log(path: Path) -> list[dict]:
    rows, success, nfes = [], None, []
    for raw in path.read_text(errors="replace").splitlines():
        line = ANSI.sub("", raw)
        match = re.search(r"\[base-adaptive-2/10\].*\bnfe=(\d+)", line)
        if match:
            nfes.append(int(match.group(1)))
        if line.strip() == "Success!":
            success = True
        elif line.strip() == "Fail!":
            success = False
        seed_match = re.search(r"current seed:\s*(\d+)", line)
        if seed_match and success is not None:
            rows.append({"seed": int(seed_match.group(1)), "success": success,
                         "chunk_actual_nfe": nfes})
            success, nfes = None, []
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=PROJECT / "outputs/paired_clean_50x20_v1")
    parser.add_argument("--method", choices=("both", "full10", "adaptive_2_10"), default="both")
    args = parser.parse_args()

    manifest_path = args.manifest.resolve()
    manifest = json.loads(manifest_path.read_text())
    if len(manifest["tasks"]) != 50 or any(len(v) != 20 for v in manifest["task_seeds"].values()):
        raise RuntimeError("Manifest must contain exactly 50 tasks x 20 unique seeds")
    digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "campaign_metadata.json").write_text(json.dumps({
        "manifest": str(manifest_path), "manifest_sha256": digest, "gpu": args.gpu,
        "setting": "demo_clean", "single_gpu": True,
    }, indent=2) + "\n")

    fastwam = Path(os.environ.get("FASTWAM_ROOT", PROJECT / "repos/FastWAM")).resolve()
    models = Path(os.environ.get("DIFFSYNTH_MODEL_BASE_PATH", PROJECT / "models")).resolve()
    checkpoint = fastwam / "checkpoints/fastwam_release/robotwin_uncond_3cam_384.pt"
    stats = fastwam / "checkpoints/fastwam_release/robotwin_uncond_3cam_384_dataset_stats.json"
    evaluator = fastwam / "third_party/RoboTwin/script/eval_policy.py"
    for required in (checkpoint, stats, evaluator):
        if not required.exists():
            raise FileNotFoundError(required)
    subprocess.run(["bash", str(METHOD / "install_policy.sh")], check=True)

    methods = ("full10", "adaptive_2_10") if args.method == "both" else (args.method,)
    env = os.environ.copy()
    for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        env.pop(key, None)
    env.update({
        "FASTWAM_ROOT": str(fastwam), "DIFFSYNTH_MODEL_BASE_PATH": str(models),
        "DIFFSYNTH_SKIP_DOWNLOAD": "true", "CUDA_VISIBLE_DEVICES": str(args.gpu),
        "WAM_ADAPTIVE_THRESHOLD_M": "0.009702832328772866",
        "WAM_ADAPTIVE_URDF": str(fastwam / "third_party/RoboTwin/assets/embodiments/aloha-agilex/urdf/arx5_description_isaac.urdf"),
        "PYTHONUNBUFFERED": "1",
    })

    for task in manifest["tasks"]:
        expected = list(map(int, manifest["task_seeds"][task]))
        for method in methods:
            task_out = output / method / task
            task_out.mkdir(parents=True, exist_ok=True)
            result_path = task_out / "episodes.json"
            if result_path.exists():
                saved = json.loads(result_path.read_text())
                if [int(x["seed"]) for x in saved] == expected:
                    print(f"SKIP {method}/{task}: 20 paired seeds complete", flush=True)
                    continue
                raise RuntimeError(f"Existing result has unexpected seeds: {result_path}")
            log = task_out / "run.log"
            policy = "fastwam_policy" if method == "full10" else "base_adaptive_2_10"
            config = fastwam / ("experiments/robotwin/fastwam_policy/deploy_policy.yml" if method == "full10"
                                else "third_party/RoboTwin/policy/base_adaptive_2_10/deploy_policy.yml")
            env["WAM_ADAPTIVE_EPISODE_RECORDS"] = str(task_out / "adaptive_episode_records.jsonl")
            cmd = [
                "python", "-u", str(evaluator), "--config", str(config), "--overrides",
                "--task_name", task, "--task_config", "demo_clean",
                "--ckpt_setting", str(checkpoint), "--seed", str(expected[0]),
                "--policy_name", policy, "--instruction_type", "unseen",
                "--eval_num_episodes", "20", "--sim_cfg_path", str(fastwam / "configs/sim_robotwin.yaml"),
                "--sim_task", "robotwin_uncond_3cam_384_1e-4", "--eval_output_dir", str(task_out / "evaluation"),
                "--mixed_precision", "bf16", "--device", "cuda", "--dataset_stats_path", str(stats),
                "--action_horizon", "32", "--replan_steps", "24", "--num_inference_steps", "10",
                "--text_cfg_scale", "1.0", "--negative_prompt", "", "--rand_device", "cpu",
                "--tiled", "False", "--timing_enabled", "True", "--skip_get_obs_within_replan", "True",
                "--eval_video_log", "False",
            ]
            print(f"START {method}/{task} seeds={expected[0]}.. ({len(expected)})", flush=True)
            with log.open("w") as stream:
                rc = subprocess.run(cmd, cwd=fastwam / "third_party/RoboTwin", env=env,
                                    stdout=stream, stderr=subprocess.STDOUT).returncode
            rows = parse_log(log)
            realized = [x["seed"] for x in rows]
            if rc != 0 or realized != expected:
                raise RuntimeError(
                    f"{method}/{task} failed or seed drifted: rc={rc}, expected={expected}, realized={realized}; see {log}"
                )
            for row in rows:
                row.update(task=task, method=method, setting="demo_clean")
            result_path.write_text(json.dumps(rows, indent=2) + "\n")
            print(f"DONE {method}/{task}: success={sum(x['success'] for x in rows)}/20", flush=True)


if __name__ == "__main__":
    main()
