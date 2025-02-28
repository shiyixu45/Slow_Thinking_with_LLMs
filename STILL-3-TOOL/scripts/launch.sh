#!/bin/bash

# how to run: cd STILL-3-TOOL && bash scripts/launch.sh
PWD=`pwd`

# required: MODEL_PATH, DATA_PATH, deepspeed_config, prompt_template and nodelist :)
# MODEL_PATH: deepseek-ai/DeepSeek-R1-Distill-Qwen-32B
# DATA_PATH: *.jsonl (with each line formatted as {"input": ..., "output": ...})
# deepspeed_config: "STILL-3-TOOL/config/ds_z3_config.json" or "STILL-3-TOOL/config/ds_z3_config_offload.json"
# prompt_template: "STILL-3-TOOL/config/prompt_template.json"
# nodelist: list of nodes to run the job
EPOCHS=17
SAVE_STRATEGY="epoch"
SAVE_STEPS=10
SCRIPT=${PWD}/scripts/run_sft_multi.sh
SCRIPT_PATH=${PWD}/sft_longcot.py
MODEL_PATH=
DATA_PATH=
deepspeed_config=
prompt_template=
JOB_ID=$RANDOM
per_device_train_batch_size=1
gradient_accumulation_steps=3
lr=1e-5
model_max_length=20000
prompt_version=r1_code
DATA_PATH_BASE=$(basename ${DATA_PATH} .jsonl)
JOB_NAME=STILL_$(basename ${MODEL_PATH})_${DATA_PATH_BASE}_EPOCHS${EPOCHS}_PDTBS${per_device_train_batch_size}_ACS${gradient_accumulation_steps}_LR${lr}_MAXLEN${model_max_length}_PROMPT${prompt_version}
SAVE_DIR=${PWD}/output
mkdir -p ${SAVE_DIR}
mkdir -p ${PWD}/log
# use four nodes to ensure tbs 96
# optional: use two nodes, set gradient_accumulation_steps 6 to ensure tbs 96, set deepspeed_config "STILL-3-TOOL/config/ds_z3_config_offload.json" to offload parameters and optimizer states to CPU memory
nodelist=(\
    # "worker-0" \
    # "worker-1" \
    # "worker-2" \
    # "worker-3" \
)
MASTER_PORT=8004

NNODES="${#nodelist[@]}"
LOCAL_HOST=`hostname`
echo "'"$LOCAL_HOST"'" $NNODES $MASTER_PORT

for ((i=0;i<${NNODES};i=i+1))
do
    echo "${nodelist[i]} => " "cd ${PWD} && bash ${SCRIPT} ${NNODES} $i ${nodelist[0]} ${MASTER_PORT} ${JOB_NAME} ${SCRIPT_PATH} ${MODEL_PATH} ${DATA_PATH} ${SAVE_DIR} ${per_device_train_batch_size} ${gradient_accumulation_steps} ${lr} ${model_max_length} ${prompt_version} ${JOB_ID} ${EPOCHS} ${SAVE_STRATEGY} ${SAVE_STEPS} ${deepspeed_config} ${prompt_template}" "&> ${PWD}/log/${JOB_NAME}_node${i}.log &"
    ssh -o ServerAliveInterval=60 "${nodelist[i]}" "cd ${PWD} && bash ${SCRIPT} ${NNODES} $i ${nodelist[0]} ${MASTER_PORT} ${JOB_NAME} ${SCRIPT_PATH} ${MODEL_PATH} ${DATA_PATH} ${SAVE_DIR} ${per_device_train_batch_size} ${gradient_accumulation_steps} ${lr} ${model_max_length} ${prompt_version} ${JOB_ID} ${EPOCHS} ${SAVE_STRATEGY} ${SAVE_STEPS} ${deepspeed_config} ${prompt_template}" &> ${PWD}/log/${JOB_NAME}_node${i}.log &
done

trap 'cleanup' SIGTERM SIGINT

# cleanup function: clean up all remote processes started by ssh when SIGTERM/SIGINT signal is caught
cleanup() {
  echo "Received SIGTERM/SIGINT at $(date +%Y-%m-%d-%H:%M:%S), cleaning up remote processes..."
  for ((i=0;i<${NNODES};i=i+1))
  do
  ssh "${nodelist[i]}" "pids=\$(pgrep -f ${SCRIPT}); if [ -n \"\$pids\" ]; then kill -9 \$pids; fi"
  ssh "${nodelist[i]}" "pids=\$(pgrep -f ${SCRIPT_PATH}); if [ -n \"\$pids\" ]; then kill -9 \$pids; fi"
  done
exit -15
}

wait
