#!/usr/bin/env bash
# Run inside an allocated 8-GPU training job. Qwen3.5-27B judge is expected at
# QWEN_JUDGE_URL or at the endpoint recorded in paths.judge_url_file.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CONFIG="${1:-${SCRIPT_DIR}/config.yaml}"

eval "$(python3 - "${CONFIG}" <<'PY'
import shlex
import sys
import yaml

c = yaml.safe_load(open(sys.argv[1], encoding="utf-8"))
values = {
    "EXPERIMENT_NAME": c["experiment_name"],
    "VERL_DIR": c["paths"]["verl_dir"],
    "MODEL_PATH": c["paths"]["model"],
    "TRAIN_FILE": c["paths"]["train_file"],
    "VAL_FILE": c["paths"]["val_file"],
    "RESUME_FROM": c["paths"]["resume_from"],
    "OUTPUT_DIR": c["paths"]["output_dir"],
    "JUDGE_URL_FILE": c["paths"]["judge_url_file"],
    "REWARD_FUNCTION": c["paths"]["reward_function"],
    "MAX_PROMPT_LENGTH": c["data"]["max_prompt_length"],
    "MAX_RESPONSE_LENGTH": c["data"]["max_response_length"],
    "TRAIN_BATCH_SIZE": c["data"]["train_batch_size"],
    "ACTOR_LR": c["algorithm"]["actor_lr"],
    "ACTOR_KL": c["algorithm"]["actor_kl_loss_coef"],
    "PPO_MINI_BATCH_SIZE": c["algorithm"]["ppo_mini_batch_size"],
    "PPO_MAX_TOKEN_LEN": c["algorithm"]["ppo_max_token_len_per_gpu"],
    "SP_SIZE": c["algorithm"]["sequence_parallel_size"],
    "WARMUP_STEPS": c["algorithm"]["warmup_steps"],
    "WEIGHT_DECAY": c["algorithm"]["weight_decay"],
    "ROLLOUT_N": c["rollout"]["n"],
    "ROLLOUT_TP": c["rollout"]["tensor_parallel_size"],
    "ROLLOUT_GPU_UTIL": c["rollout"]["gpu_memory_utilization"],
    "ROLLOUT_MODEL_LEN": c["rollout"]["max_model_len"],
    "ROLLOUT_BATCHED_TOKENS": c["rollout"]["max_num_batched_tokens"],
    "ROLLOUT_MAX_SEQS": c["rollout"]["max_num_seqs"],
    "TEMPERATURE": c["rollout"]["temperature"],
    "TOP_P": c["rollout"]["top_p"],
    "GPUS_PER_NODE": c["trainer"]["gpus_per_node"],
    "NNODES": c["trainer"]["nodes"],
    "TRAIN_STEPS": c["trainer"]["total_training_steps"],
    "TOTAL_EPOCHS": c["trainer"]["total_epochs"],
    "SAVE_FREQ": c["trainer"]["save_freq"],
    "TEST_FREQ": c["trainer"]["test_freq"],
    "PROJECT_NAME": c["trainer"]["project_name"],
    "JUDGE_MODEL": c["judge"]["model"],
    "JUDGE_TOKENIZER": c["judge"]["tokenizer"],
    "JUDGE_TIMEOUT": c["judge"]["timeout"],
    "JUDGE_MAX_TOKENS": c["judge"]["max_tokens"],
    "JUDGE_TEMPERATURE": c["judge"]["temperature"],
    "JUDGE_TOP_P": c["judge"]["top_p"],
    "JUDGE_RETRIES": c["judge"]["retries"],
    "HARD_INVALID_PENALTY": c["reward"]["hard_invalid_penalty"],
}
for key, value in values.items():
    print(f"{key}={shlex.quote(str(value))}")
PY
)"

if [[ -z "${QWEN_JUDGE_URL:-}" ]]; then
    [[ -s "${JUDGE_URL_FILE}" ]] || { echo "Missing judge URL: ${JUDGE_URL_FILE}" >&2; exit 2; }
    QWEN_JUDGE_URL="$(<"${JUDGE_URL_FILE}")"
fi
export QWEN_JUDGE_URL QWEN_JUDGE_MODEL="${JUDGE_MODEL}"
export QWEN_JUDGE_TOKENIZER_PATH="${JUDGE_TOKENIZER}"
export QWEN_JUDGE_TIMEOUT="${JUDGE_TIMEOUT}" QWEN_JUDGE_MAX_TOKENS="${JUDGE_MAX_TOKENS}"
export QWEN_JUDGE_TEMPERATURE="${JUDGE_TEMPERATURE}" QWEN_JUDGE_TOP_P="${JUDGE_TOP_P}"
export QWEN_JUDGE_RETRIES="${JUDGE_RETRIES}"

mkdir -p "${OUTPUT_DIR}"
cd "${VERL_DIR}"

python -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${VAL_FILE}" \
    data.prompt_key=prompt data.image_key=images data.truncation=left \
    data.max_prompt_length="${MAX_PROMPT_LENGTH}" \
    data.max_response_length="${MAX_RESPONSE_LENGTH}" \
    data.train_batch_size="${TRAIN_BATCH_SIZE}" \
    data.filter_overlong_prompts=True data.filter_overlong_prompts_workers=8 \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.policy_loss.loss_mode=gspo \
    actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean \
    actor_rollout_ref.actor.clip_ratio_low=3e-4 \
    actor_rollout_ref.actor.clip_ratio_high=4e-4 \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef="${ACTOR_KL}" \
    actor_rollout_ref.actor.optim.lr="${ACTOR_LR}" \
    actor_rollout_ref.actor.optim.lr_warmup_steps="${WARMUP_STEPS}" \
    actor_rollout_ref.actor.optim.lr_scheduler_type=constant \
    actor_rollout_ref.actor.optim.weight_decay="${WEIGHT_DECAY}" \
    actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE}" \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu="${PPO_MAX_TOKEN_LEN}" \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size="${SP_SIZE}" \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n="${ROLLOUT_N}" \
    actor_rollout_ref.rollout.tensor_model_parallel_size="${ROLLOUT_TP}" \
    actor_rollout_ref.rollout.gpu_memory_utilization="${ROLLOUT_GPU_UTIL}" \
    actor_rollout_ref.rollout.max_model_len="${ROLLOUT_MODEL_LEN}" \
    actor_rollout_ref.rollout.max_num_batched_tokens="${ROLLOUT_BATCHED_TOKENS}" \
    actor_rollout_ref.rollout.max_num_seqs="${ROLLOUT_MAX_SEQS}" \
    actor_rollout_ref.rollout.temperature="${TEMPERATURE}" \
    actor_rollout_ref.rollout.top_p="${TOP_P}" \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    reward.reward_manager.name=dapo \
    reward.custom_reward_function.path="${REWARD_FUNCTION}" \
    reward.custom_reward_function.name=compute_score \
    +reward.custom_reward_function.reward_kwargs.hard_invalid_penalty="${HARD_INVALID_PENALTY}" \
    trainer.project_name="${PROJECT_NAME}" \
    trainer.experiment_name="${EXPERIMENT_NAME}" \
    trainer.n_gpus_per_node="${GPUS_PER_NODE}" trainer.nnodes="${NNODES}" \
    trainer.val_before_train=False trainer.test_freq="${TEST_FREQ}" trainer.save_freq="${SAVE_FREQ}" \
    trainer.total_training_steps="${TRAIN_STEPS}" trainer.total_epochs="${TOTAL_EPOCHS}" \
    trainer.default_local_dir="${OUTPUT_DIR}" trainer.rollout_data_dir="${OUTPUT_DIR}/rollout_data" \
    trainer.resume_mode=resume_path trainer.resume_from_path="${RESUME_FROM}"
