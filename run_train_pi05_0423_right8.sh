#!/usr/bin/env bash
set -euo pipefail

# One-command launcher for pi0.5 fine-tuning on the scene1 right-arm dataset.
#
# Common overrides:
#   EXP_NAME=my_run BATCH_SIZE=64 NUM_TRAIN_STEPS=10000 ./run_train_pi05_0423_right8.sh
#   RESUME=1 ./run_train_pi05_0423_right8.sh
#   DATASET_DIR=./scene1_right8_chest_rightwrist ./run_train_pi05_0423_right8.sh

CONFIG_NAME="${CONFIG_NAME:-pi05_0423_right8_chest_wrist_low_mem_finetune}"
DATASET_DIR="${DATASET_DIR:-/data/scene1_right8_chest_rightwrist}"
ASSETS_DIR="${ASSETS_DIR:-/data/assets/pi05_scene1_right8_chest_wrist_finetune}"
ASSET_ID="${ASSET_ID:-0423_right8_chest_rightwrist}"
CHECKPOINT_BASE_DIR="${CHECKPOINT_BASE_DIR:-./checkpoints}"
OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-$(pwd)}"
export OPENPI_DATA_HOME

EXP_NAME="${EXP_NAME:-scene1_right8_chest_wrist_low_mem_lr3e6_bs1_3k}"
BATCH_SIZE="${BATCH_SIZE:-1}"
NUM_TRAIN_STEPS="${NUM_TRAIN_STEPS:-3000}"
WARMUP_STEPS="${WARMUP_STEPS:-500}"
PEAK_LR="${PEAK_LR:-3e-6}"
DECAY_STEPS="${DECAY_STEPS:-20000}"
DECAY_LR="${DECAY_LR:-3e-6}"
SAVE_INTERVAL="${SAVE_INTERVAL:-1000}"
LOG_INTERVAL="${LOG_INTERVAL:-50}"
KEEP_PERIOD="${KEEP_PERIOD:-5000}"
NUM_WORKERS="${NUM_WORKERS:-2}"
FSDP_DEVICES="${FSDP_DEVICES:-1}"
WANDB_ENABLED="${WANDB_ENABLED:-0}"
RESUME="${RESUME:-0}"
OVERWRITE="${OVERWRITE:-1}"

if [[ ! -f "${DATASET_DIR}/meta/info.json" ]]; then
  echo "Missing dataset: ${DATASET_DIR}/meta/info.json" >&2
  echo "Expected derived dataset directory: ${DATASET_DIR}" >&2
  exit 1
fi

if [[ ! -f "${ASSETS_DIR}/${ASSET_ID}/norm_stats.json" ]]; then
  echo "Missing norm stats: ${ASSETS_DIR}/${ASSET_ID}/norm_stats.json" >&2
  echo "Run scripts/prepare_0423_right8_dataset.py or copy assets from the source machine." >&2
  exit 1
fi

cmd=(
  uv run scripts/train.py "${CONFIG_NAME}"
  --exp-name "${EXP_NAME}"
  --data.repo-id "${DATASET_DIR}"
  --data.assets.assets-dir "${ASSETS_DIR}"
  --data.assets.asset-id "${ASSET_ID}"
  --checkpoint-base-dir "${CHECKPOINT_BASE_DIR}"
  --batch-size "${BATCH_SIZE}"
  --num-train-steps "${NUM_TRAIN_STEPS}"
  --lr-schedule.warmup-steps "${WARMUP_STEPS}"
  --lr-schedule.peak-lr "${PEAK_LR}"
  --lr-schedule.decay-steps "${DECAY_STEPS}"
  --lr-schedule.decay-lr "${DECAY_LR}"
  --save-interval "${SAVE_INTERVAL}"
  --log-interval "${LOG_INTERVAL}"
  --keep-period "${KEEP_PERIOD}"
  --num-workers "${NUM_WORKERS}"
  --fsdp-devices "${FSDP_DEVICES}"
)

if [[ "${WANDB_ENABLED}" == "0" ]]; then
  cmd+=(--no-wandb-enabled)
fi

if [[ "${RESUME}" == "1" ]]; then
  cmd+=(--resume)
else
  cmd+=(--no-resume)
fi

if [[ "${OVERWRITE}" == "1" ]]; then
  cmd+=(--overwrite)
else
  cmd+=(--no-overwrite)
fi

echo "Running training command:"
printf ' %q' "${cmd[@]}"
echo

"${cmd[@]}"
