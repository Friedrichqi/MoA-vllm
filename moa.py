#!/usr/bin/env python3
"""
moa_vllm_orchestrator.py - Run a Mixture-of-Agents (MoA) pipeline on top of vLLM
==========================================================================
This script assumes that **each agent (proposer or aggregator) is already being
served by a vLLM OpenAI-compatible server**.  The orchestrator then:

1. Sends the user prompt to every *proposer* in parallel.
2. Builds an *aggregate-and-synthesise* prompt that contains all proposer
   outputs.
3. Sends that prompt to one or more *aggregator* agents (allowing multiple
   layers if desired).
4. Returns the final aggregated answer.

The code is intentionally light-weight: it uses only the official *openai* and
*httpx* libraries, so you do **not** need the heavyweight OpenAI proxy SDK.

--------------------------------------------------------------------------
Prerequisites
--------------------------------------------------------------------------
1. **Spin up vLLM servers** - one per model.  For example (run each in its own
   shell or under Docker):

   ```bash
   # Proposer 1 - Qwen 1.5-72B Chat on port 8001
   python -m vllm.entrypoints.openai.api_server \
          --model Qwen/Qwen1.5-72B-Chat \
          --port 8001 --dtype bfloat16 &

   # Proposer 2 - WizardLM-2-8x22B on port 8002
   python -m vllm.entrypoints.openai.api_server \
          --model WizardLM/WizardLM-2-8x22B \
          --port 8002 --dtype bfloat16 &

   # Aggregator - Qwen 1.5-110B Chat on port 8010
   python -m vllm.entrypoints.openai.api_server \
          --model Qwen/Qwen1.5-110B-Chat \
          --port 8010 --dtype bfloat16 &
   ```

2. **Install dependencies**

   ```bash
   pip install "openai>=1.25.0" httpx>=0.27.0 tqdm>=4.66.0
   ```

--------------------------------------------------------------------------
Usage (single-layer MoA)
--------------------------------------------------------------------------
```bash
python moa_vllm_orchestrator.py \
       --prompt "Explain general relativity like I'm five." \
       --proposers qwen72b@http://localhost:8001 wizard@http://localhost:8002 \
       --aggregators qwen110b@http://localhost:8010
```

The script prints the final aggregated answer and (optionally) every
intermediate response.

--------------------------------------------------------------------------
Extending to multi-layer MoA
--------------------------------------------------------------------------
If you specify *more than one* aggregator layer, the output of the previous
layer is concatenated and fed into the next.  Example:

```bash
python moa.py \
       --prompt "Design a weekend itinerary for Berlin" \
       --proposers qwen72b@http://localhost:8001 wizard@http://localhost:8002 \
       --aggregators qwen110b@http://localhost:8010 llama3@http://localhost:8011
```

----------------------------------------------------------------------
Source code starts here
----------------------------------------------------------------------
"""
import argparse
import asyncio
import json
import textwrap
from string import Template
from typing import List, Tuple, Dict
import openai

import httpx
from tqdm import tqdm


async def complete_prompt(
    base_url: str,
    model: str,
    prompt: str,
    temperature: float = 0.7,
    max_tokens: int = 1024,
    top_p: float = 0.9,
) -> str:
    payload = {
        "model": model,
        "prompt": prompt,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
        "top_p": top_p,
    }
    headers = {
        "Content-Type": "application/json",
        # vLLM ignores the token but still expects the header to be present
        "Authorization": "Bearer EMPTY",
    }
    async with httpx.AsyncClient(timeout=600) as client:
        resp = await client.post(f"{base_url}/v1/completions", headers=headers, json=payload)
        resp.raise_for_status()
        # legacy completions API
        return resp.json()["choices"][0]["text"].strip()

# ---------------------------------------------------------------------
# Prompt template used for aggregation - identical to the MoA paper
# ---------------------------------------------------------------------
AGG_TEMPLATE = textwrap.dedent(
    """
    You have been provided with a set of responses from various open-source models to the latest user query. Your task is to synthesize these responses into a single, high-quality response. It is crucial to critically evaluate the information provided in these responses, recognizing that some of it may be biased or incorrect. Your response should not simply replicate the given answers but should offer a refined, accurate, and comprehensive reply to the instruction. Ensure your response is well-structured, coherent, and adheres to the highest standards of accuracy and reliability.

    Responses from models:
    {model_responses}
    """
)

# ---------------------------------------------------------------------
# Orchestrator core
# ---------------------------------------------------------------------
async def run_moa(
    input_prompt: str,
    proposer: Dict[str, Dict[str, Dict]],
    aggregator: Dict[str, Dict[str, Dict]],
    show_intermediates: bool = False,
):
    """Run a 2-stage MoA system (proposers ➜ aggregators) and return the final answer."""

    async def async_query(model_name, prompt, url, temperature=0.7, top_p=0.9, max_tokens=1024):
        return model_name, await complete_prompt(base_url=url, model=model_name, prompt=prompt, temperature=temperature, max_tokens=max_tokens, top_p=top_p)

    with open("model2port.json", "r") as f:
        model2port = json.load(f)

    # ---------- Stage 1: proposers -----------------------------------
    print("\n▸ Collecting proposer responses …")
    # Stores the concatenated output from all proposers in the current layer to be used as context for the next layer.
    last_layer_output = ""
    layer_proposer_output = []
    for layer_name, layer_proposers in proposer.items():
        
        proposer_tasks = []
        for _, proposer_cfg in layer_proposers.items():
            model_name = proposer_cfg["model"]
            url = f"http://localhost:{model2port[model_name]}"
            temperature = proposer_cfg.get("temperature", 0.7)
            top_p = proposer_cfg.get("top_p", 0.9)
            max_tokens = proposer_cfg.get("max_tokens", 1024)
            prompt = proposer_cfg["prompt"].replace("[input_prompts]", input_prompt)
            prompt = prompt.replace("[responses]", last_layer_output)

            proposer_tasks.append(async_query(model_name, prompt, url, temperature, top_p, max_tokens))

        proposer_results = await asyncio.gather(*proposer_tasks)
        layer_proposer_output.append([layer_name, proposer_results])
        last_layer_output = ""
        for _, text in proposer_results:
            last_layer_output.join(text)

    # Pretty print intermediate outputs if desired
    if show_intermediates:
        print("Proposer intermediate results:")
        for layer_name, proposer_results in layer_proposer_output:
            print(f"{layer_name} Results:\n")
            for name, text in proposer_results:
                print(f"\n--- {name} says ---\n{text}\n")


    # ---------- Stage 2: aggregators --------------------------------
    print("\n▸ Synthesising with aggregator(s) …")
    layer_aggregator_output = []
    for layer_name, layer_aggregator in aggregator.items():
        
        aggregator_tasks = []
        for _, aggregator_cfg in layer_aggregator.items():
            model_name = aggregator_cfg["model"]
            url = f"http://localhost:{model2port[model_name]}"
            temperature = aggregator_cfg.get("temperature", 0.7)
            top_p = aggregator_cfg.get("top_p", 0.9)
            max_tokens = aggregator_cfg.get("max_tokens", 1024)
            prompt = aggregator_cfg["prompt"].replace("[input_prompts]", input_prompt)
            prompt = prompt.replace("[responses]", last_layer_output)

            aggregator_tasks.append(async_query(model_name, prompt, url, temperature, top_p, max_tokens))

        aggregator_results = await asyncio.gather(*aggregator_tasks)
        layer_aggregator_output.append([layer_name, aggregator_results])
        last_layer_output = ""
        for _, text in aggregator_results:
            last_layer_output.join(text)

    # Pretty print intermediate outputs if desired
    if show_intermediates:
        print("Aggregator intermediate results:")
        for layer_name, aggregator_results in layer_aggregator_output:
            print(f"{layer_name} Results:\n")
            for name, text in aggregator_results:
                print(f"\n--- {name} says ---\n{text}\n")

    return last_layer_output


def main():
    parser = argparse.ArgumentParser(description="Run Mixture-of-Agents inference on vLLM servers.")
    parser.add_argument("--prompt", required=True, help="User prompt to ask.")
    parser.add_argument("--show", action="store_true", help="Print intermediate agent outputs.")
    args = parser.parse_args()

    with open("configs.json", "r") as f:
        configs = json.load(f)

    final_answer = asyncio.run(
        run_moa(args.prompt, configs["proposer"], configs["aggregator"], show_intermediates=args.show)
    )

    print("\n================ FINAL ANSWER ================\n")
    print(final_answer)


if __name__ == "__main__":
    main()
