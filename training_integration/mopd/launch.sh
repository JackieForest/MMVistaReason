#!/usr/bin/env bash
# Launch MOPD inside one allocation containing four student GPUs and four
# teacher GPUs. The input parquet must contain the configured teacher_key.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CONFIG="${1:-${SCRIPT_DIR}/config_forward_kl.yaml}"

eval "$(python3 - "${CONFIG}" <<'PY'
import shlex
import sys
import yaml

c = yaml.safe_load(open(sys.argv[1], encoding="utf-8"))
teachers = c["teachers"]
if len(teachers) != 2:
    raise ValueError("This compact launcher expects exactly two teachers.")

values = {
    "EXPERIMENT_NAME": c["experiment_name"],
    "VERL_DIR": c["paths"]["verl_dir"],
    "STUDENT_MODEL": c["paths"]["student_model"],
    "TRAIN_FILE": c["paths"]["train_file"],
    "VAL_FILE": c["paths"]["val_file"],
    "OUTPUT_DIR": c["paths"]["output_dir"],
    "REWARD_FUNCTION": c["paths"]["reward_function"],
    "TEACHER_KEY": c["data"]["teacher_key"],
    "MAX_PROMPT_LENGTH": c["data"]["max_prompt_length"],
    "MAX_RESPONSE_LENGTH": c["data"]["max_response_length"],
    "TRAIN_BATCH_SIZE": c["data"]["global_batch_size"],
    "STUDENT_GPUS": c["student"]["gpus"],
    "PPO_MINI_BATCH_SIZE": c["student"]["ppo_mini_batch_size"],
    "PPO_MAX_TOKEN_LEN": c["student"]["ppo_max_token_len_per_gpu"],
    "SP_SIZE": c["student"]["sequence_parallel_size"],
    "TEACHER_A_NAME": teachers[0]["name"],
    "TEACHER_A_KEY": teachers[0]["routing_key"],
    "TEACHER_A_MODEL": teachers[0]["model"],
    "TEACHER_A_TP": teachers[0]["tensor_parallel_size"],
    "TEACHER_A_REPLICAS": teachers[0]["num_replicas"],
    "TEACHER_B_NAME": teachers[1]["name"],
    "TEACHER_B_KEY": teachers[1]["routing_key"],
    "TEACHER_B_MODEL": teachers[1]["model"],
    "TEACHER_B_TP": teachers[1]["tensor_parallel_size"],
    "TEACHER_B_REPLICAS": teachers[1]["num_replicas"],
    "TEACHER_GPUS": sum(t["gpus"] for t in teachers),
    "ROLLOUT_N": c["rollout"]["n"],
    "ROLLOUT_TP": c["rollout"]["tensor_parallel_size"],
    "ROLLOUT_GPU_UTIL": c["rollout"]["gpu_memory_utilization"],
    "MAX_MODEL_LEN": c["rollout"]["max_model_len"],
    "ROLLOUT_BATCHED_TOKENS": c["rollout"]["max_num_batched_tokens"],
    "ROLLOUT_MAX_SEQS": c["rollout"]["max_num_seqs"],
    "TEMPERATURE": c["rollout"]["temperature"],
    "TOP_P": c["rollout"]["top_p"],
    "TEACHER_GPU_UTIL": c["teacher_inference"]["gpu_memory_utilization"],
    "TEACHER_MODEL_LEN": c["teacher_inference"]["max_model_len"],
    "TEACHER_BATCHED_TOKENS": c["teacher_inference"]["max_num_batched_tokens"],
    "TEACHER_MAX_SEQS": c["teacher_inference"]["max_num_seqs"],
    "LOSS_MODE": c["distillation"]["loss_mode"],
    "TOP_K": c["distillation"]["top_k"],
    "CHUNK_SIZE": c["distillation"]["chunk_size"],
    "ACTOR_LR": c["optimizer"]["learning_rate"],
    "WEIGHT_DECAY": c["optimizer"]["weight_decay"],
    "WARMUP_STEPS": c["optimizer"]["warmup_steps"],
    "TRAIN_STEPS": c["trainer"]["training_steps"],
    "TOTAL_EPOCHS": c["trainer"]["epochs"],
    "SAVE_FREQ": c["trainer"]["save_frequency"],
    "TEST_FREQ": c["trainer"]["test_frequency"],
    "PROJECT_NAME": c["trainer"]["project_name"],
}
for key, value in values.items():
    print(f"{key}={shlex.quote(str(value))}")
PY
)"

mkdir -p "${OUTPUT_DIR}"
cd "${VERL_DIR}"

python -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo algorithm.use_kl_in_reward=False \
    data.train_files="${TRAIN_FILE}" data.val_files="${VAL_FILE}" \
    data.prompt_key=prompt data.image_key=images data.truncation=left \
    data.max_prompt_length="${MAX_PROMPT_LENGTH}" \
    data.max_response_length="${MAX_RESPONSE_LENGTH}" \
    data.train_batch_size="${TRAIN_BATCH_SIZE}" \
    data.filter_overlong_prompts=True data.shuffle=False \
    actor_rollout_ref.model.path="${STUDENT_MODEL}" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.policy_loss.loss_mode=vanilla \
    actor_rollout_ref.actor.loss_agg_mode=token-mean \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.optim.lr="${ACTOR_LR}" \
    actor_rollout_ref.actor.optim.lr_warmup_steps="${WARMUP_STEPS}" \
    actor_rollout_ref.actor.optim.lr_scheduler_type=constant \
    actor_rollout_ref.actor.optim.weight_decay="${WEIGHT_DECAY}" \
    actor_rollout_ref.actor.optim.total_training_steps="${TRAIN_STEPS}" \
    actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE}" \
    actor_rollout_ref.actor.ppo_epochs=1 actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu="${PPO_MAX_TOKEN_LEN}" \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size="${SP_SIZE}" \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.name=vllm actor_rollout_ref.rollout.n="${ROLLOUT_N}" \
    actor_rollout_ref.rollout.tensor_model_parallel_size="${ROLLOUT_TP}" \
    actor_rollout_ref.rollout.gpu_memory_utilization="${ROLLOUT_GPU_UTIL}" \
    actor_rollout_ref.rollout.max_model_len="${MAX_MODEL_LEN}" \
    actor_rollout_ref.rollout.max_num_batched_tokens="${ROLLOUT_BATCHED_TOKENS}" \
    actor_rollout_ref.rollout.max_num_seqs="${ROLLOUT_MAX_SEQS}" \
    actor_rollout_ref.rollout.temperature="${TEMPERATURE}" \
    actor_rollout_ref.rollout.top_p="${TOP_P}" \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    reward.custom_reward_function.path="${REWARD_FUNCTION}" \
    reward.custom_reward_function.name=compute_score \
    distillation.enabled=True distillation.teacher_key="${TEACHER_KEY}" \
    distillation.n_gpus_per_node="${TEACHER_GPUS}" distillation.nnodes=1 \
    +distillation.teacher_models."${TEACHER_A_NAME}".key="${TEACHER_A_KEY}" \
    +distillation.teacher_models."${TEACHER_A_NAME}".model_path="${TEACHER_A_MODEL}" \
    +distillation.teacher_models."${TEACHER_A_NAME}".num_replicas="${TEACHER_A_REPLICAS}" \
    +distillation.teacher_models."${TEACHER_A_NAME}".inference.name=vllm \
    +distillation.teacher_models."${TEACHER_A_NAME}".inference.tensor_model_parallel_size="${TEACHER_A_TP}" \
    +distillation.teacher_models."${TEACHER_A_NAME}".inference.gpu_memory_utilization="${TEACHER_GPU_UTIL}" \
    +distillation.teacher_models."${TEACHER_A_NAME}".inference.max_model_len="${TEACHER_MODEL_LEN}" \
    +distillation.teacher_models."${TEACHER_A_NAME}".inference.max_num_batched_tokens="${TEACHER_BATCHED_TOKENS}" \
    +distillation.teacher_models."${TEACHER_A_NAME}".inference.max_num_seqs="${TEACHER_MAX_SEQS}" \
    +distillation.teacher_models."${TEACHER_B_NAME}".key="${TEACHER_B_KEY}" \
    +distillation.teacher_models."${TEACHER_B_NAME}".model_path="${TEACHER_B_MODEL}" \
    +distillation.teacher_models."${TEACHER_B_NAME}".num_replicas="${TEACHER_B_REPLICAS}" \
    +distillation.teacher_models."${TEACHER_B_NAME}".inference.name=vllm \
    +distillation.teacher_models."${TEACHER_B_NAME}".inference.tensor_model_parallel_size="${TEACHER_B_TP}" \
    +distillation.teacher_models."${TEACHER_B_NAME}".inference.gpu_memory_utilization="${TEACHER_GPU_UTIL}" \
    +distillation.teacher_models."${TEACHER_B_NAME}".inference.max_model_len="${TEACHER_MODEL_LEN}" \
    +distillation.teacher_models."${TEACHER_B_NAME}".inference.max_num_batched_tokens="${TEACHER_BATCHED_TOKENS}" \
    +distillation.teacher_models."${TEACHER_B_NAME}".inference.max_num_seqs="${TEACHER_MAX_SEQS}" \
    distillation.distillation_loss.loss_mode="${LOSS_MODE}" \
    distillation.distillation_loss.topk="${TOP_K}" \
    distillation.distillation_loss.use_task_rewards=False \
    distillation.distillation_loss.use_policy_gradient=False \
    +distillation.distillation_loss.use_chunked_topk=True \
    +distillation.distillation_loss.chunked_topk_chunk_size="${CHUNK_SIZE}" \
    distillation.distillation_loss.loss_max_clamp=10.0 \
    distillation.distillation_loss.log_prob_min_clamp=-10.0 \
    trainer.project_name="${PROJECT_NAME}" trainer.experiment_name="${EXPERIMENT_NAME}" \
    trainer.n_gpus_per_node="${STUDENT_GPUS}" trainer.nnodes=1 \
    trainer.val_before_train=False trainer.test_freq="${TEST_FREQ}" trainer.save_freq="${SAVE_FREQ}" \
    trainer.total_training_steps="${TRAIN_STEPS}" trainer.total_epochs="${TOTAL_EPOCHS}" \
    trainer.default_local_dir="${OUTPUT_DIR}"
