#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FASTWAM_ROOT="${FASTWAM_ROOT:-$(cd "$HERE/../.." && pwd)/repos/FastWAM}"
CHECKPOINT="$FASTWAM_ROOT/checkpoints/fastwam_release/robotwin_uncond_3cam_384.pt"
STATS="$FASTWAM_ROOT/checkpoints/fastwam_release/robotwin_uncond_3cam_384_dataset_stats.json"
TASK=""
EPISODES=2
GPU=0
SEED=42
OUTPUT="$HERE/outputs/run"
SETTING=demo_clean

while [[ $# -gt 0 ]]; do
  case "$1" in
    --task) TASK="$2"; shift 2 ;;
    --episodes) EPISODES="$2"; shift 2 ;;
    --gpu) GPU="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --output) OUTPUT="$2"; shift 2 ;;
    --checkpoint) CHECKPOINT="$2"; shift 2 ;;
    --stats) STATS="$2"; shift 2 ;;
    --setting) SETTING="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

[[ -n "$TASK" ]] || { echo "--task is required" >&2; exit 2; }
test -f "$CHECKPOINT" || { echo "Checkpoint missing: $CHECKPOINT" >&2; exit 2; }
test -f "$STATS" || { echo "Dataset stats missing: $STATS" >&2; exit 2; }
test -f "$FASTWAM_ROOT/third_party/RoboTwin/script/eval_policy.py" || { echo "RoboTwin missing below $FASTWAM_ROOT" >&2; exit 2; }

export FASTWAM_ROOT
export CUDA_VISIBLE_DEVICES="$GPU"
export WAM_ADAPTIVE_THRESHOLD_M="0.009702832328772866"
export WAM_ADAPTIVE_URDF="$FASTWAM_ROOT/third_party/RoboTwin/assets/embodiments/aloha-agilex/urdf/arx5_description_isaac.urdf"
export WAM_ADAPTIVE_EPISODE_RECORDS="$(mkdir -p "$OUTPUT"; cd "$OUTPUT"; pwd)/adaptive_episode_records.jsonl"
export PYTHONUNBUFFERED=1
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy || true

bash "$HERE/install_policy.sh"
cd "$FASTWAM_ROOT/third_party/RoboTwin"
python -u script/eval_policy.py \
  --config policy/base_adaptive_2_10/deploy_policy.yml \
  --overrides \
  --task_name "$TASK" \
  --task_config "$SETTING" \
  --ckpt_setting "$CHECKPOINT" \
  --seed "$SEED" \
  --policy_name base_adaptive_2_10 \
  --instruction_type unseen \
  --eval_num_episodes "$EPISODES" \
  --sim_cfg_path "$FASTWAM_ROOT/configs/sim_robotwin.yaml" \
  --sim_task robotwin_uncond_3cam_384_1e-4 \
  --eval_output_dir "$(cd "$OUTPUT"; pwd)" \
  --mixed_precision bf16 \
  --device cuda \
  --dataset_stats_path "$STATS" \
  --action_horizon 32 \
  --replan_steps 24 \
  --num_inference_steps 10 \
  --text_cfg_scale 1.0 \
  --negative_prompt '' \
  --rand_device cpu \
  --tiled False \
  --timing_enabled True \
  --skip_get_obs_within_replan True
