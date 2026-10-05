#!/bin/bash
set -euo pipefail

CONFIG="${1:?Usage: launch.sh <config.yaml>}"
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"

eval "$(python3 "${PROJECT_DIR}/config_env.py" --config "${CONFIG}")"
EXP_NAME="${NAME:-qwen35_distill_best_of_n}"
NUM_SHARDS="${NUM_SHARDS:-20}"
LOG_DIR="${PROJECT_DIR}/logs/${EXP_NAME}"
mkdir -p "${LOG_DIR}"

echo "Launching ${NUM_SHARDS} best-of-n shards for ${EXP_NAME}"
echo "Config: ${CONFIG}"
echo "Logs: ${LOG_DIR}"

for SHARD_ID in $(seq 0 $((NUM_SHARDS - 1))); do
    echo "  launch shard ${SHARD_ID}"
    bash "${PROJECT_DIR}/single_job.sh" "${CONFIG}" "${SHARD_ID}" > "${LOG_DIR}/launch_${SHARD_ID}.log" 2>&1 &
    sleep 1
done

echo "Submitted ${NUM_SHARDS} local launcher processes."
echo "Monitor: python ${PROJECT_DIR}/status.py --config ${CONFIG}"
echo "SLURM:   squeue -u ${USER} | grep ${EXP_NAME}"
