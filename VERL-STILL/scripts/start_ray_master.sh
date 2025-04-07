# Set XFormers backend to avoid CUDA errors
export VLLM_ATTENTION_BACKEND=XFORMERS

# Start Ray head node
ray start --head

NODE_IP=$(hostname -I | awk '{print $1}')

echo $NODE_IP