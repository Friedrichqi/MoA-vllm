import json

#!/usr/bin/env python3
"""
Launch vLLM API servers on GPUs sorted by free memory.
"""
import os
import sys
import argparse
import subprocess


def get_gpus():
    """
    Query NVIDIA GPUs and return the GPU indices sorted by free memory (descending).
    """
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
            encoding="utf-8"
        )
    except subprocess.CalledProcessError as e:
        print(f"Error querying GPUs: {e}", file=sys.stderr)
        sys.exit(1)

    gpu_stats = []
    for line in output.strip().splitlines():
        idx_str, mem_str = line.split(',')
        idx = int(idx_str.strip())
        free_mem = int(mem_str.strip())
        gpu_stats.append((idx, free_mem))

    gpu_stats.sort(key=lambda x: x[1], reverse=True)
    selected = [idx for idx, _ in gpu_stats]
    return selected


def launch_servers(gpu_list, model2port, dtype, max_model_length):
    """
    Launch a vLLM server on each GPU in gpu_list.
    """
    processes = []
    for i, (model, port) in enumerate(model2port.items()):
        print(f"Starting server on GPU {gpu_list[i]}, port: {port}, model: {model}")
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_list[i])

        cmd = [
            sys.executable, "-m", "vllm.entrypoints.openai.api_server",
            "--model", model,
            "--port", str(port),
            "--dtype", dtype,
            "--max-model-len", str(max_model_length),
        ]
        p = subprocess.Popen(cmd, env=env)
        processes.append(p)

    for p in processes:
        p.wait()


def main():
    parser = argparse.ArgumentParser(
        description="Launch vLLM API servers on GPUs sorted by free memory."
    )
    parser.add_argument(
        "--base-port", "-p",
        type=int,
        default=8000,
        help="Base port number (default: 8000)"
    )
    parser.add_argument(
        "--dtype", "-d",
        type=str,
        default="bfloat16",
        help="Data type for model (default: bfloat16)"
    )
    parser.add_argument(
        "--max_model_length", "-ml",
        type=int,
        default=16384,
        help="max input and output tokens for model (default: 16384)"
    )
    args =  parser.parse_args() 

    idx = 0
    model2port = {}
    with open("configs.json", "r") as f:
        configs = json.load(f)
    
    for _, layer in configs["proposer"].items():
        for _, layer_proposer in layer.items():
            if layer_proposer['model'] not in model2port:
                model2port[layer_proposer['model']] = args.base_port + idx
                idx += 1
    for _, layer in configs["aggregator"].items():
        for _, layer_aggregator in layer.items():
            if layer_aggregator['model'] not in model2port:
                model2port[layer_aggregator['model']] = args.base_port + idx
                idx += 1

    # Save model2port LUT into a JSON file
    with open("model2port.json", "w") as f:
        json.dump(model2port, f, indent=4)


    # Get all available GPUs and sort them by free memory
    gpu_list = get_gpus()

    # Launch the servers
    launch_servers(gpu_list, model2port, args.dtype, args.max_model_length)


if __name__ == "__main__":
    main()
