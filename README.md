# Mixture-of-Agents with vLLM

This repository provides an implementation of the Mixture-of-Agents (MoA) architecture, leveraging `vLLM` for high-performance inference. The system uses a multi-stage process with "proposer" and "aggregator" agents to generate refined responses to user prompts.

## Key Files

-   **MoA-vllm/configs.json**: The core configuration file for the MoA architecture. Here you define the layers of proposer and aggregator agents, specifying the model and prompt for each.

-   **MoA-vllm/start_server.py**: A utility script to automatically launch vLLM OpenAI-compatible API servers for all unique models defined in configs.json.

-   **MoA-vllm/moa.py**: The main script to execute the MoA pipeline. It orchestrates the calls to the proposer and aggregator agents to produce a final answer.

-   **MoA-vllm/model2port.json**: An auto-generated file that maps the models from configs.json to the ports they are served on.


## Usage

Follow these steps to run the MoA pipeline.

### Step 1: Configure the Architecture

Modify configs.json to define your desired MoA setup. You can specify multiple layers for both the `proposer` and `aggregator` stages, and configure different models, prompts, and parameters for each agent.

### Step 2: Launch the Model Servers

Run the start_server.py script to serve all the models required by your configuration. This script will read configs.json, identify the unique models, and launch a vLLM server for each one on a separate GPU.

```bash

python MoA-vllm/start_server.py

```

The script accepts the following optional arguments:

-   `--base-port`: The starting port for the servers. Defaults to `8000`.

-   `--dtype`: The data type for the model. Defaults to `bfloat16`.

-   `--max_model_length`: The maximum model length for vLLM. Defaults to `16384`.
### Step 3: Run the MoA Pipeline

Once the servers are running, you can execute the MoA pipeline using moa.py.

```bash

python MoA-vllm/moa.py --prompt "Your question here"

```
The script takes the following arguments:

-   `--prompt`: (Required) The user prompt you want to ask the MoA.

-   `--show`: (Optional) A flag to print the intermediate outputs from each layer of proposer agents.