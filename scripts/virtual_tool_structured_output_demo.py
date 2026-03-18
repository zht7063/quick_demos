# /// script
# requires-python = ">= 3.13, < 3.14"
# dependencies = [
#     "openai",
#     "dotenv",
# ]
# ///

"""
使用虚拟工具格式化 llm 输出。
"""

import os
import json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

tools = [
    {
        "type": "function",
        "function": {
            "name": "report_analysis",
            "description": "Report the analysis result of the given text.",
            "parameters": {
                "type": "object",
                "properties": {
                    "sentiment": {
                        "type": "string",
                        "enum": ["positive", "negative"],
                        "description": "Overall sentiment of the text.",
                    }
                },
                "required": ["sentiment"],
            },
        },
    }
]

client = OpenAI(
    base_url=os.getenv("DASHSCOPE_BASE_URL"),
    api_key=os.getenv("DASHSCOPE_API_KEY"),
)

SYSTEM_PROMPT = (
    "你是一名专业的情感分析助手。"
    "你的任务是判断用户提供的文本所表达的整体情感倾向。\n\n"
    "规则：\n"
    "- 若文本整体传达正面、乐观、满意或喜悦等情绪，判定为 positive。\n"
    "- 若文本整体传达负面、悲观、不满或失望等情绪，判定为 negative。\n"
    "- 仅根据文本内容判断，不做额外推断。\n"
    "- 必须调用 report_analysis 工具返回结果，不得以纯文本形式回复。"
)

# 测试用例（可逐一替换以验证不同情感）
TEXT_TO_ANALYZE = "今天发布了新版本，上线后用户反馈很好！"  # 预期: positive
# TEXT_TO_ANALYZE = "系统频繁崩溃，用户投诉不断，问题迟迟得不到解决。"  # 预期: negative
# TEXT_TO_ANALYZE = "新功能终于上线了，团队付出了很多努力，大家都很开心。"  # 预期: positive
# TEXT_TO_ANALYZE = "这次合作完全失败，损失惨重，非常令人失望。"          # 预期: negative

completion = client.chat.completions.create(
    model=os.getenv("DASHSCOPE_MAIN_MODEL"),
    messages=[
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": f"分析以下文本：\n\n{TEXT_TO_ANALYZE}",
        },
    ],
    tools=tools,
    tool_choice={"type": "function", "function": {"name": "report_analysis"}},
    # 如果使用推理模型，则工具选择部分可以省略，模型会自动选择工具。
    # tool_choice = "auto"
)

print(completion)
print()
print(completion.choices[0].message.tool_calls[0].function.arguments)
print()
print(json.loads(completion.choices[0].message.tool_calls[0].function.arguments))
