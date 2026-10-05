#!/bin/bash
#SBATCH --job-name=q35_4b_sft
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --time=14-00:00:00

set -euo pipefail

CONFIG_NAME="${1:?Usage: run_sft_job_sbatch.sh <config_name_without_yaml>}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WORK_DIR="${SFT_WORK_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
LLAMA_FACTORY="${LLAMA_FACTORY_DIR:?Set LLAMA_FACTORY_DIR to the LlamaFactory checkout}"
CONFIG="${WORK_DIR}/config/${CONFIG_NAME}.yaml"
LOG_DIR="${WORK_DIR}/logs"
export CONFIG

if [ -z "${SLURM_JOB_ID:-}" ]; then
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    mkdir -p "${LOG_DIR}"
    SBATCH_ARGS=(
        --job-name="${SFT_JOB_NAME:-q35_sft}"
        --output="${LOG_DIR}/${CONFIG_NAME}_${TIMESTAMP}_%j.out"
        --error="${LOG_DIR}/${CONFIG_NAME}_${TIMESTAMP}_%j.err"
    )
    [[ -n "${SFT_PARTITION:-}" ]] && SBATCH_ARGS+=(--partition="${SFT_PARTITION}")
    [[ -n "${SFT_QUOTA_TYPE:-}" ]] && SBATCH_ARGS+=(--quotatype="${SFT_QUOTA_TYPE}")
    [[ -n "${SFT_EXCLUDE:-}" ]] && SBATCH_ARGS+=(--exclude="${SFT_EXCLUDE}")
    exec sbatch "${SBATCH_ARGS[@]}" "$0" "$@"
fi

set +u
if [[ -n "${CONDA_SH:-}" ]]; then
    source "${CONDA_SH}"
elif command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
else
    echo "Set CONDA_SH or make conda available on PATH." >&2
    exit 2
fi
conda activate "${SFT_CONDA_ENV:-sft}"
set -u

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="${CUDA_HOME}/bin:${CONDA_PREFIX}/bin:${PATH}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export WANDB_DISABLED="${WANDB_DISABLED:-true}"
SFT_CACHE_ROOT="${SFT_CACHE_ROOT:-${WORK_DIR}/.cache}"
SFT_OUTPUT_ROOT="${SFT_OUTPUT_ROOT:-${WORK_DIR}/output}"
export HF_HOME="${HF_HOME:-${SFT_CACHE_ROOT}/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${SFT_CACHE_ROOT}/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-${SFT_CACHE_ROOT}/torch_extensions}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${SFT_CACHE_ROOT}/xdg}"
export TMPDIR="${TMPDIR:-${SFT_CACHE_ROOT}/tmp}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${SLURM_TMPDIR:-/tmp}/mmvista_reason_triton_${SLURM_JOB_ID:-local}}"
mkdir -p "${HF_HOME}" "${HF_DATASETS_CACHE}" "${TRANSFORMERS_CACHE}" "${HUGGINGFACE_HUB_CACHE}" \
    "${TORCH_EXTENSIONS_DIR}" "${TRITON_CACHE_DIR}" "${XDG_CACHE_HOME}" "${TMPDIR}" \
    "${SFT_OUTPUT_ROOT}/tokenized_data" "${SFT_OUTPUT_ROOT}/models"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
unset QWEN35_DISABLE_FLA

echo "========== NODE/GPU DIAGNOSTICS BEFORE TRAIN =========="
echo "time: $(date '+%F %T')"
echo "hostname: $(hostname)"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID:-NA}"
echo "SLURM_JOB_NODELIST: ${SLURM_JOB_NODELIST:-NA}"
echo "SLURM_GPUS: ${SLURM_GPUS:-NA}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-NA}"
echo "which python: $(which python)"
python - <<'PYDIAG'
import importlib.metadata as md
import torch
try:
    import triton
    print("triton:", triton.__version__)
except Exception as exc:
    print("triton import error:", repr(exc))
print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
for pkg in ["transformers", "llamafactory", "datasets", "pyarrow", "deepspeed", "flash-attn", "liger-kernel"]:
    try:
        print(f"{pkg}:", md.version(pkg))
    except Exception as exc:
        print(f"{pkg}: ERR {exc!r}")
print("torch cuda available:", torch.cuda.is_available())
print("torch device count:", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(f"torch device {i}:", torch.cuda.get_device_name(i))
PYDIAG
if command -v nvidia-smi >/dev/null 2>&1; then
    echo "----- nvidia-smi summary -----"
    nvidia-smi
    echo "----- nvidia-smi query -----"
    nvidia-smi --query-gpu=index,uuid,name,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu,pstate,ecc.errors.uncorrected.volatile.total --format=csv,noheader,nounits || true
    echo "----- nvidia-smi compute apps -----"
    nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader,nounits || true
fi
echo "======================================================="

python - <<'PYGPUCHECK'
import os
import subprocess
import sys
min_free_mib = int(os.environ.get("MIN_TRAIN_GPU_FREE_MIB", "70000"))
visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
if not visible or visible == "NA":
    print("[GPU CHECK] CUDA_VISIBLE_DEVICES is not set; skip physical GPU free-memory check.")
    sys.exit(0)
physical_ids = [x.strip() for x in visible.split(",") if x.strip().isdigit()]
if not physical_ids:
    print(f"[GPU CHECK] CUDA_VISIBLE_DEVICES={visible!r} is not numeric; skip physical GPU free-memory check.")
    sys.exit(0)
rows = subprocess.check_output([
    "nvidia-smi", "--query-gpu=index,memory.free,ecc.errors.uncorrected.volatile.total", "--format=csv,noheader,nounits",
], text=True).strip().splitlines()
stats = {}
for row in rows:
    parts = [p.strip() for p in row.split(",")]
    if len(parts) >= 3:
        stats[parts[0]] = {"free": int(parts[1]), "ecc": int(parts[2])}
bad = []
for gpu_id in physical_ids:
    stat = stats.get(gpu_id)
    if stat is None:
        bad.append(f"gpu {gpu_id}: missing from nvidia-smi query")
        continue
    print(f"[GPU CHECK] physical gpu {gpu_id}: free={stat['free']} MiB, ecc_uncorrected={stat['ecc']}")
    if stat["free"] < min_free_mib:
        bad.append(f"gpu {gpu_id}: free {stat['free']} MiB < {min_free_mib} MiB")
    if stat["ecc"] != 0:
        bad.append(f"gpu {gpu_id}: volatile uncorrected ECC {stat['ecc']} != 0")
if bad:
    print("[GPU CHECK] Refusing to start training on unhealthy/busy allocated GPU(s):")
    for item in bad:
        print(f"[GPU CHECK] - {item}")
    sys.exit(42)
print("[GPU CHECK] allocated GPUs look clean enough for training.")
PYGPUCHECK

python - <<'PYPREFLIGHT'
import json
import os
import sys
from pathlib import Path
import yaml
config_path = Path(os.environ["CONFIG"])
with config_path.open() as f:
    cfg = yaml.safe_load(f)
dataset_info = Path(cfg["dataset_dir"]) / "dataset_info.json"
required_paths = {
    "model_name_or_path": Path(cfg["model_name_or_path"]),
    "deepspeed": Path(cfg["deepspeed"]),
    "dataset_info": dataset_info,
    "output_dir_parent": Path(cfg["output_dir"]).parent,
    "tokenized_path_parent": Path(cfg["tokenized_path"]).parent,
}
resume = cfg.get("resume_from_checkpoint")
if resume:
    required_paths["resume_from_checkpoint"] = Path(resume)
missing = [f"{name}: {path}" for name, path in required_paths.items() if not path.exists()]
if missing:
    print("[PREFLIGHT] Missing required path(s):")
    for item in missing:
        print(f"[PREFLIGHT] - {item}")
    sys.exit(43)
info = json.loads(dataset_info.read_text())
dataset_name = cfg["dataset"]
if dataset_name not in info:
    print(f"[PREFLIGHT] dataset {dataset_name} missing from {dataset_info}")
    sys.exit(45)
data_file = Path(info[dataset_name]["file_name"])
if not data_file.exists():
    print(f"[PREFLIGHT] data file missing: {data_file}")
    sys.exit(45)
print(f"[PREFLIGHT] dataset={dataset_name} file={data_file} size={data_file.stat().st_size}")
print(f"[PREFLIGHT] resume_from_checkpoint={resume}")
try:
    import flash_attn  # noqa: F401
    import causal_conv1d  # noqa: F401
except Exception as exc:
    print(f"[PREFLIGHT] flash-attn/causal-conv1d import failed: {exc!r}")
    sys.exit(44)
try:
    from liger_kernel.transformers import apply_liger_kernel_to_qwen3_5  # noqa: F401
    print("[PREFLIGHT] liger qwen3.5 import OK.")
except Exception as exc:
    print(f"[PREFLIGHT] liger qwen3.5 import failed: {exc!r}")
    sys.exit(47)
print("[PREFLIGHT] config paths and imports look OK.")
PYPREFLIGHT

cd "${LLAMA_FACTORY}"
export PYTHONPATH="${LLAMA_FACTORY}/src${PYTHONPATH:+:${PYTHONPATH}}"

FORCE_TORCHRUN=1 NNODES=1 NPROC_PER_NODE="${SFT_NPROC_PER_NODE:-8}" \
    llamafactory-cli train "${CONFIG}"
