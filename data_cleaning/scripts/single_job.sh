#!/bin/bash
# Submit/retry one split until its stop flag exists.
set -uo pipefail

CONFIG="${1:?Usage: single_job.sh <config.yaml> <index>}"
INDEX="${2:?Usage: single_job.sh <config.yaml> <index>}"

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
SCRIPT_DIR="${PROJECT_DIR}/scripts"

eval "$(python3 "${SCRIPT_DIR}/config_env.py" --config "${CONFIG}")"

PARTITION="${SLURM_PARTITION:-gpu}"
QUOTA="${SLURM_QUOTA_TYPE:-spot}"
RESERVED_MIN_RUNNING="${SLURM_RESERVED_MIN_RUNNING:-0}"
RESERVED_QUOTA="${SLURM_RESERVED_QUOTA_TYPE:-reserved}"
SPOT_QUOTA="${SLURM_SPOT_QUOTA_TYPE:-spot}"
GPUS="${SLURM_GPUS:-1}"
TIME_LIMIT="${SLURM_TIME:-7200}"
EXCLUDE="${SLURM_EXCLUDE:-}"
EXP_NAME="${NAME:-qwen35_122b_data_cleaning}"
OUTPUT="${OUTPUT_DIR:?missing output_dir}"
STOP_FILE="${OUTPUT}/stop_files/rollout_${INDEX}.flag"
JOB_NAME="${INDEX}_${EXP_NAME}"
LOCK_DIR="${OUTPUT}/locks"
SUBMIT_LOCK="${LOCK_DIR}/${EXP_NAME}.submit.lock"
RESERVED_MARKER_DIR="${LOCK_DIR}/${EXP_NAME}.reserved_markers"
RESERVED_MARKER="${RESERVED_MARKER_DIR}/${INDEX}.marker"

mkdir -p "${LOCK_DIR}" "${RESERVED_MARKER_DIR}"

count_reserved_active_jobs() {
    local exp_name="$1"
    local job_ids
    job_ids=$(squeue -u "${USER}" -h 2>/dev/null | awk '($7 == "R" || $7 == "PD") && $4 == "reserved" {print $1}')
    if [[ -z "${job_ids}" ]]; then
        echo 0
        return
    fi
    echo "${job_ids}" \
        | xargs -r -n 50 scontrol show job 2>/dev/null \
        | awk -v exp_name="${exp_name}" '
            /^JobId=/ {
                if (job_name ~ ("^[0-9]+_" exp_name "$") && job_state ~ /^(RUNNING|PENDING)$/) {
                    c++
                }
                job_name = job_state = ""
            }
            {
                for (i = 1; i <= NF; i++) {
                    if ($i ~ /^JobName=/) {
                        job_name = substr($i, 9)
                    } else if ($i ~ /^JobState=/) {
                        job_state = substr($i, 10)
                    }
                }
            }
            END {
                if (job_name ~ ("^[0-9]+_" exp_name "$") && job_state ~ /^(RUNNING|PENDING)$/) {
                    c++
                }
                print c + 0
            }
        '
}

pick_quota() {
    local exp_name="$1"
    local reserved_min_running="$2"
    local default_quota="$3"
    local reserved_quota="$4"
    local spot_quota="$5"
    if [[ "${reserved_min_running}" =~ ^[0-9]+$ ]] && (( reserved_min_running > 0 )); then
        local reserved_count
        reserved_count="$(count_reserved_active_jobs "${exp_name}")"
        local pending_marker_count
        pending_marker_count="$(find "${RESERVED_MARKER_DIR}" -type f -name '*.marker' 2>/dev/null | wc -l | tr -d ' ')"
        if [[ "${pending_marker_count}" =~ ^[0-9]+$ ]]; then
            reserved_count=$((reserved_count + pending_marker_count))
        fi
        if [[ "${reserved_count}" =~ ^[0-9]+$ ]] && (( reserved_count < reserved_min_running )); then
            echo "${reserved_quota}"
            return
        fi
        echo "${spot_quota}"
        return
    fi
    echo "${default_quota}"
}

cleanup_old_jobs() {
    local old_jobs
    old_jobs=$(squeue -u "${USER}" -n "${JOB_NAME}" -o "%i" --noheader 2>/dev/null)
    if [[ -n "${old_jobs}" ]]; then
        echo "[${EXP_NAME}] cancelling old jobs for split ${INDEX}: ${old_jobs}"
        echo "${old_jobs}" | while read -r jid; do
            [[ -n "${jid}" ]] && scancel "${jid}" 2>/dev/null || true
        done
        sleep 5
    fi
}

while true; do
    if [[ -f "${STOP_FILE}" ]]; then
        echo "[${EXP_NAME}] split ${INDEX} stop flag exists; exiting."
        cleanup_old_jobs
        exit 0
    fi

    cleanup_old_jobs

    QUOTA_TO_USE="$(
        {
            flock -x 9
            chosen_quota="$(pick_quota "${EXP_NAME}" "${RESERVED_MIN_RUNNING}" "${QUOTA}" "${RESERVED_QUOTA}" "${SPOT_QUOTA}")"
            if [[ "${chosen_quota}" == "${RESERVED_QUOTA}" ]]; then
                : > "${RESERVED_MARKER}"
            else
                rm -f "${RESERVED_MARKER}"
            fi
            echo "${chosen_quota}"
        } 9>"${SUBMIT_LOCK}"
    )"

    SRUN=(srun -p "${PARTITION}" --gres="gpu:${GPUS}" --quotatype="${QUOTA_TO_USE}" --job-name="${JOB_NAME}" --time="${TIME_LIMIT}")
    if [[ -n "${EXCLUDE}" ]]; then
        SRUN+=(-x "${EXCLUDE}")
    fi

    echo "[${EXP_NAME}] submitting split ${INDEX}: partition=${PARTITION} quota=${QUOTA_TO_USE} gpus=${GPUS}"
    "${SRUN[@]}" bash "${SCRIPT_DIR}/slurm_worker.sh" "${CONFIG}" "${INDEX}"
    rc=$?
    if [[ "${QUOTA_TO_USE}" == "${RESERVED_QUOTA}" ]]; then
        rm -f "${RESERVED_MARKER}"
    fi
    echo "[${EXP_NAME}] split ${INDEX} attempt exited rc=${rc}; retrying after 20s unless complete."
    sleep 20
done
