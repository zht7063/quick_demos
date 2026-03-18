# /// script
# requires-python = ">= 3.13, < 3.14"
# dependencies = [
#     "openai",
#     "dotenv",
# ]
# ///

"""
基于基本 openai-client 实现工具调用功能。
"""

import json
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

client = OpenAI(
    base_url=os.getenv("DASHSCOPE_BASE_URL"),
    api_key=os.getenv("DASHSCOPE_API_KEY"),
)


tools = [
    {
        "type": "function",
        "function": {
            "name": "get_capital",
            "description": "Get the capital of a country",
            "parameters": {
                "type": "object",
                "properties": {
                    "country": {
                        "type": "string",
                        "description": "The country to get the capital of",
                    },
                },
            },
            "required": ["country"],
        },
    }
]


def get_response(messages):
    completion = client.chat.completions.create(
        model="deepseek-v3.2", messages=messages, tools=tools
    )
    return completion


uesr_question = "What is the capital of France?"
messages = [{"role": "user", "content": uesr_question}]


def get_capital(country):
    return "Paris"


# 工具名称 -> 函数的映射表
tool_handlers = {
    "get_capital": get_capital,
}

response = get_response(messages)

# 模型驱动的工具调用循环：只要模型返回 tool_calls，就执行并继续对话
while response.choices[0].finish_reason == "tool_calls":
    assistant_message = response.choices[0].message

    # 将模型的 assistant 消息（含 tool_calls）追加到历史
    messages.append(assistant_message)

    # 依次执行每个工具调用
    for tool_call in assistant_message.tool_calls:
        func_name = tool_call.function.name
        func_args = json.loads(tool_call.function.arguments)

        # 调用对应的本地函数
        result = tool_handlers[func_name](**func_args)
        print(f"> 调用工具 [{func_name}]，参数: {func_args}，结果: {result}")

        # 将工具结果以 tool 角色追加到历史
        messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": str(result),
            }
        )

    # 携带工具结果再次请求模型
    response = get_response(messages)

# finish_reason == "stop"：模型已生成最终回答
final_answer = response.choices[0].message.content
print(f"> 模型最终回答: {final_answer}")
