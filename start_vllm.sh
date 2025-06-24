#!/usr/bin/env bash
set -euo pipefail

# Number of servers to launch
NUM_SERVERS=3
# Base port for servers
BASE_PORT=8000
# Path to your model
MODEL_PATH="/data/home/qyjh/hf_models/Llama-3.1-8B-Instruct"

# Fetch GPU indices sorted by free memory (highest first), take the top N
mapfile -t GPU_LIST < <(
    nvidia-smi --query-gpu=index,memory.free \
        --format=csv,noheader,nounits | \
    sort -t, -k2 -nr | \
    head -n "${NUM_SERVERS}" | \
    awk -F, '{print $1}'
)

if [ "${#GPU_LIST[@]}" -lt "${NUM_SERVERS}" ]; then
    echo "Error: Found only ${#GPU_LIST[@]} GPU(s), but need ${NUM_SERVERS}."
    exit 1
fi

# Launch vllm servers
for i in "${!GPU_LIST[@]}"; do
    GPU_ID=${GPU_LIST[$i]}
    PORT=$((BASE_PORT + i))
    echo "Starting server on GPU ${GPU_ID}, port ${PORT}"
    CUDA_VISIBLE_DEVICES="${GPU_ID}" python -m vllm.entrypoints.openai.api_server \
        --model "${MODEL_PATH}" \
        --port "${PORT}" \
        --dtype bfloat16 &
done

wait