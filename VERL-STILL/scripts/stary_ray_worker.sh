# Set XFormers backend to avoid CUDA errors
export VLLM_ATTENTION_BACKEND=XFORMERS

# Connect to head node (replace with your head node's address)

HEAD_NODE_IP=YOUR_HEAD_NODE_IP

ray start --address=$HEAD_NODE_IP:6379