#!/bin/bash

NNODES=$1
NODE_RANK=$2
MASTER_ADDR=$3
MASTER_PORT=$4
JOB_NAME=$5
SCRIPT_PATH=$6
MODEL_PATH=$7
DATA_PATH=$8
SAVE_DIR=$9
per_device_train_batch_size=${10}
gradient_accumulation_steps=${11}
lr=${12}
model_max_length=${13}
prompt_version=${14}
JOB_ID=${15}
EPOCHS=${16}
SAVE_STRATEGY=${17}
SAVE_STEPS=${18}
deepspeed_config=${19}
prompt_template=${20}

export OMP_NUM_THREADS=24
export WANDB_MODE=offline
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# required: based on the cluster configuration
# export CUDA_DEVICE_MAX_CONNECTIONS=1
# export NCCL_SOCKET_IFNAME=
# export NCCL_IB_DISABLE=0
# export NCCL_IB_CUDA_SUPPORT=1
# export NCCL_IB_GID_INDEX=0
# export NCCL_IB_HCA=

# optional: environment variables for debugging
# export NCCL_DEBUG=DEBUG  # https://stackoverflow.com/questions/61075390/pytorch-nccl-error-unhandled-system-error-nccl-version-2-4-8
# export NCCL_DEBUG_SUBSYS=GRAPH # https://pytorch.org/docs/stable/distributed.html
# export TORCH_LOGS=+all
# export TORCH_DISTRIBUTED_DEBUG=DETAIL
# export TORCH_CPP_LOG_LEVEL=INFO

echo ${per_device_train_batch_size} ${gradient_accumulation_steps} ${lr} ${model_max_length} ${prompt_version}

torchrun --nproc_per_node=8 \
    --nnodes=${NNODES} \
    --node_rank=${NODE_RANK} \
    --master_addr=${MASTER_ADDR} \
    --master_port=${MASTER_PORT} \
    --rdzv_backend static \
    --rdzv_id $JOB_ID \
    ${SCRIPT_PATH} \
    --model_name_or_path ${MODEL_PATH} \
    --data_path ${DATA_PATH} \
    --bf16 True \
    --output_dir ${SAVE_DIR}/${JOB_NAME} \
    --num_train_epochs ${EPOCHS} \
    --per_device_train_batch_size ${per_device_train_batch_size} \
    --gradient_accumulation_steps ${gradient_accumulation_steps} \
    --evaluation_strategy "no" \
    --save_strategy ${SAVE_STRATEGY} \
    --save_steps ${SAVE_STEPS} \
    --learning_rate ${lr} \
    --lr_scheduler_type "cosine" \
    --weight_decay 0.1 \
    --warmup_ratio 0.1 \
    --logging_steps 1 \
    --model_max_length ${model_max_length} \
    --tf32 True \
    --deepspeed ${deepspeed_config} \
    --gradient_checkpointing True \
    --report_to none \
    --prompt_version ${prompt_version} \
    --save_only_model True \
    --save_total_limit 4 \
    --prompt_template ${prompt_template}
