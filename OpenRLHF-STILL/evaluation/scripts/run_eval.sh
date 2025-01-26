# source /share/project/zhipengchen/zbc/rl_env/bin/activate
export VLLM_WORKER_MULTIPROC_METHOD=spawn

model_list=(\
    "RUC-AIBOX/STILL-3-1.5B-preview" \
)
L=32768
num_model=${#model_list[@]}


for ((i=0;i<$num_model;i++)) do
{
    MODEL_PATH=${model_list[$i]}
    SRC_PATH=${MODEL_PATH}

    python run_eval_mp.py \
        --data_name AIME24 \
        --target_path ${SRC_PATH} \
        --model_name_or_path ./outputs \
        --max_tokens ${L} \
        --system_prompt none \
        --paralle_size 1

    python run_eval_mp.py \
        --data_name MATH_OAI \
        --target_path ${SRC_PATH} \
        --model_name_or_path ./outputs \
        --max_tokens ${L} \
        --system_prompt none \
        --paralle_size 1

    python run_eval_mp.py \
        --data_name LiveAOPSbench \
        --target_path ${SRC_PATH} \
        --model_name_or_path ./outputs \
        --max_tokens ${L} \
        --system_prompt none \
        --decode greedy \
        --paralle_size 1

    python run_eval_mp.py \
        --data_name OMNI \
        --target_path ${SRC_PATH} \
        --model_name_or_path ./outputs \
        --max_tokens ${L} \
        --system_prompt none \
        --decode greedy \
        --paralle_size 1
    
}
done
