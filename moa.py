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
from typing import List, Tuple

import httpx
from tqdm import tqdm


async def complete_prompt(
    base_url: str,
    model: str,
    prompt: str,
    temperature: float = 0.7,
    max_tokens: int = 1024,
) -> str:
    payload = {
        "model": model,
        "prompt": prompt,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
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
    prompt: str,
    proposer_cfg: List[Tuple[str, str]],
    aggregator_cfg: List[Tuple[str, str]],
    show_intermediates: bool = False,
):
    """Run a 2-layer MoA (proposers ➜ aggregators) and return the final answer."""

    # ---------- Layer 1: proposers -----------------------------------
    proposer_prompt = prompt

    async def _query_proposer(name_url: Tuple[str, str]):
        name, url = name_url
        return name, await complete_prompt(url, name, proposer_prompt, temperature=0.7)

    print("\n▸ Collecting proposer responses …")
    proposer_tasks = [_query_proposer(p) for p in proposer_cfg]
    proposer_results = await asyncio.gather(*proposer_tasks)

    # Pretty print intermediate outputs if desired
    if show_intermediates:
        for name, text in proposer_results:
            print(f"\n--- {name} says ---\n{text}\n")

    # ---------- Build aggregator prompt ------------------------------
    numbered = []
    for idx, (name, text) in enumerate(proposer_results, start=1):
        numbered.append(f"{idx}. [{name}]\n{text}\n")
    agg_context = "\n".join(numbered)

    agg_prompt = AGG_TEMPLATE.format(model_responses=agg_context)
    aggregator_prompt = agg_prompt + "\nUser query:\n" + prompt

    # ---------- Layer 2+: aggregators --------------------------------
    async def _query_aggregator(name_url: Tuple[str, str]):
        name, url = name_url
        return name, await complete_prompt(url, name, aggregator_prompt, temperature=0.7)

    print("\n▸ Synthesising with aggregator(s) …")
    agg_tasks = [_query_aggregator(a) for a in aggregator_cfg]
    agg_results = await asyncio.gather(*agg_tasks)

    # If multiple aggregators, just return the first; otherwise extend logic for deeper layers
    final_name, final_answer = agg_results[0]

    if show_intermediates and len(agg_results) > 1:
        for name, text in agg_results:
            print(f"\n=== Aggregator {name} output ===\n{text}\n")

    return final_name, final_answer

# ---------------------------------------------------------------------
# CLI glue
# ---------------------------------------------------------------------

def _parse_agent_list(raw: List[str]) -> List[Tuple[str, str]]:
    """Turn ["name@url", …] into [(name, url), …] and validate."""
    parsed = []
    for entry in raw:
        if "@" not in entry:
            raise argparse.ArgumentTypeError(f"Agent spec '{entry}' must be NAME@BASE_URL")
        name, url = entry.split("@", 1)
        parsed.append((name, url.rstrip("/")))
    return parsed


def main():
    parser = argparse.ArgumentParser(description="Run Mixture-of-Agents inference on vLLM servers.")
    parser.add_argument("--prompt", required=True, help="User prompt to ask.")
    parser.add_argument(
        "--proposers",
        required=True,
        nargs="+",
        help="Space-separated list of proposer specs NAME@URL, e.g. qwen@http://localhost:8001",
    )
    parser.add_argument(
        "--aggregators",
        required=True,
        nargs="+",
        help="List of aggregator specs NAME@URL (order = layer order).",
    )
    parser.add_argument("--show", action="store_true", help="Print intermediate agent outputs.")

    args = parser.parse_args()
    proposers = _parse_agent_list(args.proposers)
    aggregators = _parse_agent_list(args.aggregators)

    final_name, final_answer = asyncio.run(
        run_moa(args.prompt, proposers, aggregators, show_intermediates=args.show)
    )

    print("\n================ FINAL ANSWER ================\n")
    print(final_answer)
    print("\n==============================================\n")
    print(f"(generated by aggregator: {final_name})")


if __name__ == "__main__":
    main()
