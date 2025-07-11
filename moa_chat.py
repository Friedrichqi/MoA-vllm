import argparse
import asyncio
import json
from typing import Dict
import httpx

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
        "repetition_penalty": 1.2,
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer EMPTY",
    }
    async with httpx.AsyncClient(timeout=1200) as client:
        resp = await client.post(f"{base_url}/v1/completions", headers=headers, json=payload)
        resp.raise_for_status()
        return resp.json()["choices"][0]["text"].strip()

async def async_query(model_name, prompt, url, temperature=0.7, top_p=0.9, max_tokens=1024):
    return model_name, await complete_prompt(url, model_name, prompt, temperature, max_tokens, top_p)

async def process_stage(stage_name, stage_config, input_prompt, model2port, initial_context, show_intermediates):
    """Helper function to run a single stage (proposer or aggregator)."""
    # print(f"\n▸ Running {stage_name} stage...")

    last_layer_output = initial_context
    stage_output_history = []

    for layer_name, layer_configs in stage_config.items():
        tasks = []
        for _, config in layer_configs.items():
            model_name = config["model"]
            url = f"http://localhost:{model2port[model_name]}"
            temperature = config.get("temperature", 0.7)
            top_p = config.get("top_p", 0.9)
            max_tokens = config.get("max_tokens", 1024)
            
            prompt = config["prompt"].replace("[input_prompts]", input_prompt)
            prompt = prompt.replace("[responses]", last_layer_output)

            tasks.append(async_query(model_name, prompt, url, temperature, top_p, max_tokens))

        layer_results = await asyncio.gather(*tasks)
        stage_output_history.append((layer_name, layer_results))
        
        # Concatenate results from the current layer to be used as context for the next layer.
        texts = [text for _, text in layer_results]
        last_layer_output = "".join(texts)

    if show_intermediates:
        print(f"\n--- {stage_name.capitalize()} Intermediate Results ---")
        for layer_name, results in stage_output_history:
            print(f"\n--- {layer_name} ---")
            for name, text in results:
                print(f"\n--- {name} says ---\n{text}")
        print("-" * (len(stage_name) + 28))

    return last_layer_output

async def run_moa(
    input_prompt: str,
    show_intermediates: bool,
    proposer_config: Dict[str, Dict[str, Dict]],
    aggregator_config: Dict[str, Dict[str, Dict]],
):
    """
    Runs the Mixture-of-Agents (MoA) pipeline.
    """

    with open("model2port.json", "r") as f:
        model2port = json.load(f)
    
    # Stage 1: Proposers
    proposer_output = await process_stage(
        stage_name="proposer",
        stage_config=proposer_config,
        input_prompt=input_prompt,
        model2port=model2port,
        initial_context="",
        show_intermediates=show_intermediates
    )

    # Stage 2: Aggregators
    aggregator_output = await process_stage(
        stage_name="aggregator",
        stage_config=aggregator_config,
        input_prompt=input_prompt,
        model2port=model2port,
        initial_context=proposer_output,
        show_intermediates=show_intermediates
    )

    return aggregator_output

def main():
    parser = argparse.ArgumentParser(description="Run Mixture-of-Agents inference on vLLM servers.")
    parser.add_argument("--prompt", required=True, help="User prompt to ask.")
    parser.add_argument("--show", action="store_true", help="Print intermediate agent outputs.")
    args = parser.parse_args()

    with open("configs.json", "r") as f:
        configs = json.load(f)

    final_answer = asyncio.run(
        run_moa(input_prompt=args.prompt, show_intermediates=args.show, proposer_config=configs["proposer"], aggregator_config=configs["aggregator"])
    )

    print("\n================ FINAL ANSWER ================\n")
    print(final_answer)


if __name__ == "__main__":
    main()