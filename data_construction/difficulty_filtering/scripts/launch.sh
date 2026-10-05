#!/bin/bash
# Launch all split jobs in the background.
set -euo pipefail

CONFIG="${1:?Usage: launch.sh <config.yaml>}"

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
SCRIPT_DIR="${PROJECT_DIR}/scripts"

eval "$(python3 "${SCRIPT_DIR}/config_env.py" --config "${CONFIG}")"
EXP_NAME="${NAME:-qwen35_4b_pr}"
NUM_SPLITS="${NUM_SPLITS:-100}"
LOG_DIR="${PROJECT_DIR}/logs/${EXP_NAME}"
LOCK_DIR="${OUTPUT_DIR}/locks"
RESERVED_MARKER_DIR="${LOCK_DIR}/${EXP_NAME}.reserved_markers"
mkdir -p "${LOG_DIR}"
rm -rf "${RESERVED_MARKER_DIR}"
mkdir -p "${RESERVED_MARKER_DIR}"

echo "Launching ${NUM_SPLITS} splits for ${EXP_NAME}"
echo "Config: ${CONFIG}"
echo "Logs: ${LOG_DIR}"
echo "Cleared reserved marker dir: ${RESERVED_MARKER_DIR}"

for INDEX in $(seq 0 $((NUM_SPLITS - 1))); do
    echo "  launch split ${INDEX}"
    bash "${SCRIPT_DIR}/single_job.sh" "${CONFIG}" "${INDEX}" > "${LOG_DIR}/launch_${INDEX}.log" 2>&1 &
    sleep 1
done

echo "Submitted ${NUM_SPLITS} local launcher processes."
echo "Monitor: python ${SCRIPT_DIR}/status.py --config ${CONFIG}"
echo "SLURM:   squeue -u ${USER} | grep ${EXP_NAME}"
