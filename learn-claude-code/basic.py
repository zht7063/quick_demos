# /// script
# requires-python = ">= 3.13, < 3.14"
# dependencies = [
#   "openai",
#   "dotenv",
# ]
# ///

import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    base_url=os.getenv("DASHSCOPE_BASE_URL"),
    api_key=os.getenv("DASHSCOPE_API_KEY"),
)

response = client.chat.completions.create(
    model=os.getenv("DASHSCOPE_MAIN_MODEL"),
    messages=[
        {
            "role": "system",
            "content": "You are a helpful assistant."
        },
        {
            "role": "user",
            "content": "What is the capital of France?"
        }
    ]
)
print(response)
