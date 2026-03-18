# /// script
# requires-python = ">= 3.13, < 3.14"
# dependencies = [
#     "openai",
#     "dotenv",
#     "fastapi",
#     "uvicorn",
#     "httpx",
# ]
# ///

"""
通过 fastapi 构建 sse 流式接口。

启动后，通过 http://localhost:8000/stream 访问 sse 流式接口。
"""

import os
import httpx
from openai import AsyncOpenAI
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import uvicorn

load_dotenv()

client = AsyncOpenAI(
    base_url=os.getenv("DASHSCOPE_BASE_URL"),
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    http_client=httpx.AsyncClient(
        proxy=os.getenv("HTTP_PROXY"),
    ),
)

app = FastAPI()


async def llm_stream(query: str):
    """调用 LLM 并以 SSE 格式逐 token 流式输出。"""
    stream = await client.chat.completions.create(
        model=os.getenv("DASHSCOPE_MAIN_MODEL"),
        messages=[{"role": "user", "content": query}],
        stream=True,
    )

    async for chunk in stream:
        content = chunk.choices[0].delta.content
        if content:
            yield f"data: {content}\n\n"

    yield "data: [DONE]\n\n"


@app.get("/stream")
async def stream_endpoint(q: str = "你好，请介绍一下你自己。"):
    return StreamingResponse(
        llm_stream(q),
        media_type="text/event-stream",
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
