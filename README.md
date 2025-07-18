# Mixture‑of‑Agents (MoA) with vLLM  
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue)](https://www.python.org/)  
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)  

High‑performance “Mixture‑of‑Agents” architecture [MoA](https://github.com/togethercomputer/MoA) built on [vLLM](https://github.com/vllm-project/vllm) for fast multi‑agent inference. The system uses a multi-stage process with "proposer" and "aggregator" agents to generate refined responses to user prompts.

## Installation

```bash
git clone https://github.com/Friedrichqi/MoA-vllm.git
cd MoA-vllm
conda create -n moa_env python=3.12 -y
conda activate moa_env
pip install vllm
```

## Key Files

-   **MoA-vllm/configs.json**: The core configuration file for the MoA architecture. Here you define the layers of proposer and aggregator agents, specifying the model and prompt for each.

Example:
```json
{
    "proposer": {
        "layer1": {
            "1":{
                "model": "Qwen/Qwen3-8B",
                "prompt": "You are a helpful assistant. Basing on [responses], answer the question: [input_prompts]",
                "temperature": 0.7,
                "top_p": 0.9,
                "max_tokens": 2048
            },
            "2":{
                "model": "Qwen/Qwen3-8B",
                "prompt": "You are a helpful assistant. Basing on [responses], answer the question: [input_prompts]",
                "temperature": 0.7,
                "top_p": 0.9,
                "max_tokens": 2048
            },
            "3":{
                "model": "Qwen/Qwen3-8B",
                "prompt": "You are a helpful assistant. Basing on [responses], answer the question: [input_prompts]",
                "temperature": 0.7,
                "top_p": 0.9,
                "max_tokens": 2048
            }
        },
        "layer2": {
            "1":{
                "model": "Qwen/Qwen3-8B",
                "prompt": "You are a helpful assistant. Basing on [responses], answer the question: [input_prompts]",
                "temperature": 0.7,
                "top_p": 0.9,
                "max_tokens": 2048
            },
            "2":{
                "model": "Qwen/Qwen3-8B",
                "prompt": "You are a helpful assistant. Basing on [responses], answer the question: [input_prompts]",
                "temperature": 0.7,
                "top_p": 0.9,
                "max_tokens": 2048
            },
            "3":{
                "model": "Qwen/Qwen3-8B",
                "prompt": "You are a helpful assistant. Basing on [responses], answer the question: [input_prompts]",
                "temperature": 0.7,
                "top_p": 0.9,
                "max_tokens": 2048
            }
        } 
    },
    "aggregator": {
        "layer1":{
            "1":{
                "model": "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
                "prompt": "You have been provided with a set of responses from various open-source models to the latest user query. Your task is to synthesize these responses into a single, high-quality response. It is crucial to critically evaluate the information provided in these responses, recognizing that some of it may be biased or incorrect. Your response should not simply replicate the given answers but should offer a refined, accurate, and comprehensive reply to the instruction. Ensure your response is well-structured, coherent, and adheres to the highest standards of accuracy and reliability. \n Responses from models:\n\n[responses]\n\n Synthesized answer for the question, [input_prompts], should be:\n",
                "temperature": 0.7,
                "top_p": 0.9,
                "max_tokens": 2048
            }
        }
    }
}
```

-   **MoA-vllm/start_server.py**: A utility script to automatically launch vLLM OpenAI-compatible API servers for all unique models defined in configs.json. (Serving each of the agents separately in the moa system with vLLM)

-   **MoA-vllm/moa_api.py**: To launch the moa system using OpenAI-compatible API servers. (Connecting all individual agents served in start_server.py and providing direct communication to the moa system)

-   **MoA-vllm/moa_chat.py**: The playable script to chat with the moa system with single prompt. It orchestrates the calls to the proposer and aggregator agents to produce a final answer.

-   **MoA-vllm/model2port.json**: An auto-generated file that maps the models from configs.json to the ports they are served on.


## Usage

Follow these steps to run the MoA pipeline.

### Step 1: Configure the Architecture

Modify configs.json to define your desired MoA setup. You can specify multiple layers for both the `proposer` and `aggregator` stages, and configure different models, prompts, and parameters for each agent.

### Step 2: Launch the Model Servers

Run the start_server.py script to serve all the models required by your configuration. This script will read configs.json, identify the unique models, and launch a vLLM server for each one on a separate GPU.

```bash

python start_server.py

```

The script accepts the following optional arguments:

-   `--base-port`: The starting port for the servers. Defaults to `8000`.

-   `--dtype`: The data type for the model. Defaults to `bfloat16`.

-   `--max_model_length`: The maximum model length for vLLM. Defaults to model's default max output length if not specified.
### Step 3: Run the MoA Pipeline

Once the servers are running, you can execute the MoA pipeline using moa_chat.py.

```bash
python MoA-vllm/moa_chat.py \
  --prompt "Hello world" \
  --show
```
The script takes the following arguments:

-   `--prompt`: (Required) The user prompt you want to ask the MoA.

-   `--show`: (Optional) A flag to print the intermediate outputs from each layer of proposer agents.

### Step 4 (optional): Serve the MoA through api for evaluation or curl interaction 
```bash

python moa_api.py

```
- It has a default port of 9000 and then you are curl using

```bash
curl -X POST http://localhost:9000/v1/completions \
     -H "Content-Type: application/json" \
     -d '{ \
           "model": "moa", \
           "prompt": "Tell me a joke." \
         }'
```

## Evaluate with EleutherAI/lm-evaluation-harness

### Step1: Follow the [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) tutorial for installment.

### Step2: Follow the instruction given above to launch the whole MoA system through moa_api.py

### Step3: Run the evaluation pipeline using the following cmd.

```bash
lm_eval --model  local-completions --model_args  "model=moa,base_url=http://0.0.0.0:9000/v1/completions,tokenizer=gpt2,tokenized_requests=false,num_concurrent=4,timeout=86400,max_new_tokens=1024"   --tasks  gsm8k  --output_path output/moa_gsm8k.json --wandb_args project=lm-eval-harness-integration
```
- --wandb_args requires special installment and is totally optional to provide you with finer process of the evaluation.

## Known Bugs:
- TimeoutError: raise asyncio.TimeoutError from exc_val. It has something to do with the api server doesn't respond in some time. Decrease num_concurrent to 1 or decrease max_tokens in config.json to contrain output length might help.

- CUDA out of memory: If you are sure you can serve your model under the precision type set in advance in start_server.py, then it should be vLLM preserving too much memory for KV Cache. Manually set max model length to a smaller value when starting moa agents through

```bash
python start_server.py --max_model_length 2048
```
might help.