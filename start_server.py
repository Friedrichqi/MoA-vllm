import json

#!/usr/bin/env python3
"""
Launch vLLM API servers on GPUs sorted by free memory.
"""
import os
import sys
import argparse
import subprocess


def parse_args():
    parser = argparse.ArgumentParser(
        description="Launch vLLM API servers on top-N GPUs sorted by free memory."
    )
    parser.add_argument(
        "--num-servers", "-n",
        type=int,
        default=1,
        help="Number of servers to launch (default: 1)"
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
    return parser.parse_args()


def get_gpus():
    """
    Query NVIDIA GPUs and return the top-N GPU indices sorted by free memory (descending).
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


def launch_servers(gpu_list, base_port, model_path, dtype):
    """
    Launch a vLLM server on each GPU in gpu_list.
    """
    processes = []
    for i, gpu_id in enumerate(gpu_list):
        port = base_port + i
        print(f"Starting server on GPU {gpu_id}, port {port}")
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

        cmd = [
            sys.executable, "-m", "vllm.entrypoints.openai.api_server",
            "--model", model_path,
            "--port", str(port),
            "--dtype", dtype
        ]
        p = subprocess.Popen(cmd, env=env)
        processes.append(p)

    # Wait for all servers to exit
    for p in processes:
        p.wait()


def main():
    args = parse_args()

    id = 0
    ip_address = {}
    with open("configs.json", "r") as f:
        configs = json.load(f)
    
    for _, layer in configs["proposer"].items():
        for _, layer_proposer in layer.items():
            if layer_proposer['model'] not in ip_address:
                ip_address[layer_proposer['model']] = f"http://localhost:{args.base_port + id}"
                id += 1
    for _, layer in configs["aggregator"].items():
        for _, layer_aggregator in layer.items():
            if layer_aggregator['model'] not in ip_address:
                ip_address[layer_aggregator['model']] = f"http://localhost:{args.base_port + id}"
                id += 1
    # # Print the IP addresses for each model
    # print("IP addresses for each model:")
    # for model, ip in ip_address.items():
    #     print(f"{model}: {ip}")


    # Get all available GPUs and sort them by free memory
    gpu_list = get_gpus()


    # Launch the servers
    launch_servers(gpu_list, args.base_port, args.model_path, args.dtype)


if __name__ == "__main__":
    main()
