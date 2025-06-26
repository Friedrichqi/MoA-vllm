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
    text = await run_moa(
        input_prompt=req.prompt,
        show_intermediates=False,
        proposer_config=json.load(open("configs.json"))["proposer"],
        aggregator_config=json.load(open("configs.json"))["aggregator"],
    )
    return {
        "object": "text_completion",
        "choices": [{"text": text}],
        "usage": {"prompt_tokens": 0, "completion_tokens": len(text.split())},
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 9000)))