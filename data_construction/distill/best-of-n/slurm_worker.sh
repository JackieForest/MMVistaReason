#!/bin/bash
set -euo pipefail

CONFIG="${1:?Usage: slurm_worker.sh <config.yaml> <shard_id>}"
SHARD_ID="${2:?Usage: slurm_worker.sh <config.yaml> <shard_id>}"

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"

eval "$(python3 "${PROJECT_DIR}/config_env.py" --config "${CONFIG}")"

EXP_NAME="${NAME:-qwen35_distill_best_of_n}"
CONTAINER="${VLLM_CONTAINER:?missing vllm.container}"
MODEL="${MODEL:?missing model}"
PYTHONPATH_EXTRA="${VLLM_PYTHONPATH:-}"
MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-8192}"
TP="${VLLM_TENSOR_PARALLEL_SIZE:-1}"
MAX_SEQS="${VLLM_MAX_NUM_SEQS:-16}"
GPU_UTIL="${VLLM_GPU_MEMORY_UTILIZATION:-0.90}"
GPU_MIN_FREE_GB="${VLLM_GPU_MIN_FREE_GB:-20}"
STARTUP_TIMEOUT="${VLLM_STARTUP_TIMEOUT:-600}"
BASE_PORT="${VLLM_BASE_PORT:-9900}"
PORT=$((BASE_PORT + SHARD_ID))
CLEAN_PROCESS="${VLLM_CLEAN_PROCESS:-false}"
ENABLE_EXPERT_PARALLEL="${VLLM_ENABLE_EXPERT_PARALLEL:-false}"
ENFORCE_EAGER="${VLLM_ENFORCE_EAGER:-true}"
ASYNC_SCHEDULING="${VLLM_ASYNC_SCHEDULING:-true}"
ENABLE_CHUNKED_PREFILL="${VLLM_ENABLE_CHUNKED_PREFILL:-true}"

LOG_DIR="${PROJECT_DIR}/logs/${EXP_NAME}"
mkdir -p "${LOG_DIR}"
JOB_CACHE_ID="${SLURM_JOB_ID:-local}_${SHARD_ID}_${HOSTNAME:-unknown}"
SHARED_CACHE_DIR="${OUTPUT_DIR:-/tmp}/${EXP_NAME}_runtime_cache"
LOCAL_CACHE_DIR="${VLLM_RUNTIME_CACHE_ROOT:-${SLURM_TMPDIR:-/tmp}/${EXP_NAME}_runtime_cache}"
mkdir -p "${SHARED_CACHE_DIR}/hf" "${LOCAL_CACHE_DIR}/xdg/${JOB_CACHE_ID}" "${LOCAL_CACHE_DIR}/triton/${JOB_CACHE_ID}"
export HF_HOME="${SHARED_CACHE_DIR}/hf"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE_DIR:-${SHARED_CACHE_DIR}/hf/datasets}"
export XDG_CACHE_HOME="${LOCAL_CACHE_DIR}/xdg/${JOB_CACHE_ID}"
export TRITON_CACHE_DIR="${LOCAL_CACHE_DIR}/triton/${JOB_CACHE_ID}"

HOST="$(hostname)"
API_HOST="127.0.0.1"
SERVER_LOG="${LOG_DIR}/server_${SHARD_ID}_${HOST}.log"
SHARD_LOG="${LOG_DIR}/shard_${SHARD_ID}_${HOST}.log"
GPU_VISIBLE="${CUDA_VISIBLE_DEVICES:-${SLURM_STEP_GPUS:-${SLURM_JOB_GPUS:-}}}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
if [[ -n "${GPU_VISIBLE}" ]]; then
    export CUDA_VISIBLE_DEVICES="${GPU_VISIBLE}"
    export NVIDIA_VISIBLE_DEVICES="${GPU_VISIBLE}"
    export APPTAINERENV_CUDA_VISIBLE_DEVICES="${GPU_VISIBLE}"
    export APPTAINERENV_NVIDIA_VISIBLE_DEVICES="${GPU_VISIBLE}"
    export SINGULARITYENV_CUDA_VISIBLE_DEVICES="${GPU_VISIBLE}"
    export SINGULARITYENV_NVIDIA_VISIBLE_DEVICES="${GPU_VISIBLE}"
fi

echo "=== ${EXP_NAME} shard ${SHARD_ID} on ${HOST} ==="
echo "server log: ${SERVER_LOG}"
echo "shard log:  ${SHARD_LOG}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"

if [[ "${CLEAN_PROCESS}" == "true" ]] && command -v swatch >/dev/null 2>&1; then
    swatch -n "${HOST}" clean_process || true
    sleep 5
fi

cleanup() {
    if [[ -n "${VLLM_PID:-}" ]]; then
        kill -TERM "-${VLLM_PID}" 2>/dev/null || true
        sleep 2
        kill -KILL "-${VLLM_PID}" 2>/dev/null || true
    fi
}
trap cleanup EXIT

: > "${SERVER_LOG}"
{
    echo "=== preflight ${EXP_NAME} shard ${SHARD_ID} on ${HOST} ==="
    nvidia-smi --query-gpu=index,uuid,memory.used,memory.total --format=csv,noheader 2>/dev/null || true
} >> "${SERVER_LOG}"

if [[ -n "${GPU_VISIBLE}" ]] && command -v nvidia-smi >/dev/null 2>&1; then
    MIN_FREE_MIB=$(python3 - <<PY
print(int(float("${GPU_MIN_FREE_GB}") * 1024))
PY
)
    LOW_GPU_INFO=$(nvidia-smi -i "${GPU_VISIBLE}" --query-gpu=index,memory.free,memory.total --format=csv,noheader,nounits 2>/dev/null \
        | awk -F, -v min_free="${MIN_FREE_MIB}" '{
            gsub(/^[ \t]+|[ \t]+$/, "", $1);
            gsub(/^[ \t]+|[ \t]+$/, "", $2);
            gsub(/^[ \t]+|[ \t]+$/, "", $3);
            if ($2 + 0 < min_free) print "gpu=" $1 " free_mib=" $2 " total_mib=" $3;
        }')
    if [[ -n "${LOW_GPU_INFO}" ]]; then
        echo "Selected GPUs do not have enough free memory." | tee -a "${SERVER_LOG}"
        echo "${LOW_GPU_INFO}" | tee -a "${SERVER_LOG}"
        exit 42
    fi
fi

VLLM_ARGS=(
    vllm serve "${MODEL}"
    --host 0.0.0.0
    --port "${PORT}"
    --max-model-len "${MAX_MODEL_LEN}"
    --tensor-parallel-size "${TP}"
    --limit-mm-per-prompt.video 0
    --limit-mm-per-prompt.image 1
    --gpu-memory-utilization "${GPU_UTIL}"
    --max-num-seqs "${MAX_SEQS}"
)
if [[ "${ENABLE_EXPERT_PARALLEL}" == "true" ]]; then
    VLLM_ARGS+=(--enable-expert-parallel)
fi
if [[ "${ASYNC_SCHEDULING}" == "true" ]]; then
    VLLM_ARGS+=(--async-scheduling)
fi
if [[ "${ENABLE_CHUNKED_PREFILL}" == "true" ]]; then
    VLLM_ARGS+=(--enable-chunked-prefill)
fi
if [[ "${ENFORCE_EAGER}" == "true" ]]; then
    VLLM_ARGS+=(--enforce-eager)
fi

setsid apptainer exec --nv --cleanenv \
    --env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}" \
    --env NVIDIA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}" \
    --env CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER}" \
    --env HF_HOME="${HF_HOME}" \
    --env HF_DATASETS_CACHE="${HF_DATASETS_CACHE}" \
    --env XDG_CACHE_HOME="${XDG_CACHE_HOME}" \
    --env TRITON_CACHE_DIR="${TRITON_CACHE_DIR}" \
    --env PYTHONPATH="${PYTHONPATH_EXTRA}" \
    --bind /share:/share,/mnt:/mnt \
    "${CONTAINER}" \
    "${VLLM_ARGS[@]}" \
    >> "${SERVER_LOG}" 2>&1 &

VLLM_PID=$!
WAIT_SEC=0
while ! grep -q "Application startup complete." "${SERVER_LOG}" 2>/dev/null; do
    if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
        echo "vLLM process died during startup. Last server log lines:"
        tail -80 "${SERVER_LOG}" || true
        exit 1
    fi
    if [[ "${WAIT_SEC}" -ge "${STARTUP_TIMEOUT}" ]]; then
        echo "vLLM startup timed out after ${STARTUP_TIMEOUT}s. Last server log lines:"
        tail -80 "${SERVER_LOG}" || true
        exit 1
    fi
    sleep 10
    WAIT_SEC=$((WAIT_SEC + 10))
done
echo "vLLM ready after ${WAIT_SEC}s at ${API_HOST}:${PORT}"

python3 "${PROJECT_DIR}/score_shard.py" \
    --config "${CONFIG}" \
    --shard-id "${SHARD_ID}" \
    --url "${API_HOST}" \
    --port "${PORT}" \
    2>&1 | tee "${SHARD_LOG}"
