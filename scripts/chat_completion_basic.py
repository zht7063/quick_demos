# /// script
# requires-python = ">= 3.13, < 3.14"
# dependencies = [
#     "openai",
#     "dotenv",
# ]
# ///

"""
简单的 llm-client 使用方法实例。
"""

import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    base_url=os.getenv("DASHSCOPE_BASE_URL"),
    api_key=os.getenv("DASHSCOPE_API_KEY"),
)

completion = client.chat.completions.create(
    model=os.getenv("DASHSCOPE_MAIN_MODEL"),
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"},
    ],
)

print(f"> completion: {completion}")
print(
    f"> completion.choices[0].message.content: {completion.choices[0].message.content}"
)
