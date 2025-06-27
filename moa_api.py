import uvicorn, asyncio, os, json
from fastapi import FastAPI
from pydantic import BaseModel
from moa import run_moa

app = FastAPI()

class CompletionRequest(BaseModel):
    prompt: str
    model: str | None = None

@app.post("/v1/completions")
async def completions(req: CompletionRequest):
    tasks = [
        run_moa(
            input_prompt=p,
            show_intermediates=False,
            proposer_config=json.load(open("configs.json"))["proposer"],
            aggregator_config=json.load(open("configs.json"))["aggregator"],
        )
        for p in req.prompt
    ]
    texts = await asyncio.gather(*tasks)

    choices = [
        {
            "index": i,
            "text": text,
            "finish_reason": "stop",
            "logprobs": None,
        }
        for i, text in enumerate(texts)
    ]

    return {
        "object": "text_completion",
        "model": "moa",
        "choices": choices,
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": sum(len(t.split()) for t in texts),
        },
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 9010)))