#!/bin/bash
set -x

# Warning: Export VLLM_ATTENTION_BACKEND on every machine before starting Ray cluster.
# vLLM without XFORMERS will results in CUDA errors.
export VLLM_ATTENTION_BACKEND=XFORMERS

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL_PATH="$2"
            shift 2
            ;;
        *)
            break
            ;;
    esac
done

N_SAMPLES=16
EPISODE=10000
TBS=128
RBS=128
KL=0.001
TEMP=1.0
MAX_LEN=8192
EL=0.0
SAVE_MODEL_NAME=verl_grpo-qwen_32b_zero-data_sort-tbs_${TBS}-rbs_${RBS}-sample_$N_SAMPLES-kl_${KL}_decay-ep_${EPISODE}-maxlen_${MAX_LEN}-plr_5e7-EL_${EL}-temp_${TEMP}-reward_zero


# Set default model path if not provided
if [ -z "$MODEL_PATH" ]; then
    MODEL_PATH="Qwen/"
fi

$ROOT_FOLDER=/share/project/zhipengchen/RLHF/deepscaler-main

# Train over 4 nodes, 8 A100-80GB GPUs per node.
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=YOUR_TRAIN_DATA.parquet \
    data.val_files=YOUR_TEST_DATA.parquet \
    data.train_batch_size=${RBS} \
    data.val_batch_size=512 \
    data.max_prompt_length=1024 \
    data.max_response_length=${MAX_LEN} \
    data.shuffle=False \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=5e-7 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=${TBS} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.ppo_epochs=1 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=20480 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=${KL} \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.grad_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.temperature=${TEMP} \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.n=${N_SAMPLES} \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    +actor_rollout_ref.actor.use_dynamic_kl_loss=True \
    algorithm.kl_ctrl.kl_coef=${KL} \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='czp_rlhf' \
    trainer.experiment_name=${SAVE_MODEL_NAME} \
    +trainer.val_before_train=True \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=4 \
    trainer.save_freq=20 \
    trainer.test_freq=5 \
    trainer.default_hdfs_dir=null \
    actor_rollout_ref.actor.entropy_coeff=${EL} \
    reward_model.reward_manager=STILLZero \
    trainer.resume_from_path=False \
    trainer.resume_mode=auto \
    trainer.total_epochs=${EPISODE} "${@:1}"