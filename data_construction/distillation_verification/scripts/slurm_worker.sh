#!/bin/bash
set -euo pipefail

CONFIG="${1:?Usage: slurm_worker.sh <config.yaml> <index>}"
INDEX="${2:?Usage: slurm_worker.sh <config.yaml> <index>}"

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
SCRIPT_DIR="${PROJECT_DIR}/scripts"
eval "$(python3 "${SCRIPT_DIR}/config_env.py" --config "${CONFIG}")"

EXP_NAME="${NAME:-qwen35_27b_distillation_verification}"
CONTAINER="${VLLM_CONTAINER:?missing vllm.container}"
OUTPUT="${OUTPUT_DIR:?missing output_dir}"
TP="${VLLM_TENSOR_PARALLEL_SIZE:-2}"
RUNTIME_CACHE_ROOT="${VLLM_RUNTIME_CACHE_ROOT:-${OUTPUT}/verify_runtime_cache}"

LOG_DIR="${PROJECT_DIR}/logs/${EXP_NAME}"
mkdir -p "${LOG_DIR}"
HOST="$(hostname)"
SPLIT_LOG="${LOG_DIR}/split_${INDEX}_${HOST}.log"
JOB_CACHE_ID="${SLURM_JOB_ID:-local}_${INDEX}_${HOST}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
GPU_VISIBLE="${CUDA_VISIBLE_DEVICES:-${SLURM_STEP_GPUS:-${SLURM_JOB_GPUS:-}}}"
if [[ -n "${GPU_VISIBLE}" ]]; then
    export CUDA_VISIBLE_DEVICES="${GPU_VISIBLE}"
    export NVIDIA_VISIBLE_DEVICES="${GPU_VISIBLE}"
    export APPTAINERENV_CUDA_VISIBLE_DEVICES="${GPU_VISIBLE}"
    export APPTAINERENV_NVIDIA_VISIBLE_DEVICES="${GPU_VISIBLE}"
    export SINGULARITYENV_CUDA_VISIBLE_DEVICES="${GPU_VISIBLE}"
    export SINGULARITYENV_NVIDIA_VISIBLE_DEVICES="${GPU_VISIBLE}"
fi

mkdir -p "${RUNTIME_CACHE_ROOT}/hf" "${RUNTIME_CACHE_ROOT}/xdg/${JOB_CACHE_ID}" "${RUNTIME_CACHE_ROOT}/triton/${JOB_CACHE_ID}"

echo "=== ${EXP_NAME} split ${INDEX} on ${HOST} ==="
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-<unset>}"
echo "SLURM_STEP_GPUS=${SLURM_STEP_GPUS:-<unset>}"
echo "tensor_parallel_size=${TP}"
echo "log=${SPLIT_LOG}"

apptainer exec --nv --cleanenv \
    --env HF_HUB_OFFLINE="${HF_HUB_OFFLINE}" \
    --env HF_HOME="${RUNTIME_CACHE_ROOT}/hf" \
    --env HF_DATASETS_CACHE="${RUNTIME_CACHE_ROOT}/hf/datasets" \
    --env XDG_CACHE_HOME="${RUNTIME_CACHE_ROOT}/xdg/${JOB_CACHE_ID}" \
    --env TRITON_CACHE_DIR="${RUNTIME_CACHE_ROOT}/triton/${JOB_CACHE_ID}" \
    --env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}" \
    --env NVIDIA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}" \
    --env CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER}" \
    --bind /share:/share,/mnt:/mnt \
    "${CONTAINER}" \
    python "${SCRIPT_DIR}/run_verify.py" --config "${CONFIG}" --index "${INDEX}" \
    2>&1 | tee "${SPLIT_LOG}"
