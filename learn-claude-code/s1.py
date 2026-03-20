# /// script
# requires-python = ">= 3.13, < 3.14"
# dependencies = [
#   "openai",
#   "loguru",
#   "python-dotenv",
# ]
# ///

import json
import os
import subprocess
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

base_url = os.getenv("DASHSCOPE_BASE_URL")
api_key = os.getenv("DASHSCOPE_API_KEY")
model = os.getenv("DASHSCOPE_MAIN_MODEL")

system_prompt = f"You are a coding agent at {os.getcwd()}. Use bash to solve tasks. Act, don't explain."

tools = [
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Run a shell command.",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
            },
        },
    }
]

client = OpenAI(
    base_url=base_url,
    api_key=api_key,
)

# 防止模型反复要工具导致无限循环；可按需调大
MAX_AGENT_TURNS = 20


def run_bash(command: str) -> str:
    dangerous = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"]
    if any(d in command for d in dangerous):
        return "Error: Dangerous command blocked"
    try:
        r = subprocess.run(
            command,
            shell=True,
            cwd=os.getcwd(),
            capture_output=True,
            text=True,
            timeout=120,
        )
        out = (r.stdout + r.stderr).strip()
        return out[:50000] if out else "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: Timeout (120s)"


def agent_loop(messages):
    for _ in range(MAX_AGENT_TURNS):
        response = client.chat.completions.create(
            model=model, messages=messages, tools=tools, max_tokens=8000
        )

        messages.append(response.choices[0].message)

        if not response.choices[0].message.tool_calls:
            return

        for tool_call in response.choices[0].message.tool_calls:
            # tool_call: ChatCompletionMessageFunctionToolCall[]
            function_name = tool_call.function.name
            try:
                args = json.loads(tool_call.function.arguments)
            except json.JSONDecodeError as e:
                result = f"Error: Invalid JSON in tool arguments: {str(e)[:100]}"
                messages.append(
                    {"role": "tool", "tool_call_id": tool_call.id, "content": result}
                )
                continue

            if function_name == "bash":
                result = run_bash(args.get("command", ""))
            else:
                result = f"Error: unknown tool {function_name!r}"

            messages.append(
                {"role": "tool", "tool_call_id": tool_call.id, "content": result}
            )
    
    # 达到最大轮次限制，追加一条 assistant 消息说明
    messages.append({
        "role": "assistant",
        "content": f"(已达到最大轮次限制 {MAX_AGENT_TURNS}，停止执行)"
    })
    

def last_assistant_text(messages) -> str:
    """从消息列表中安全提取最后一条 assistant 消息的文本内容"""
    for msg in reversed(messages):
        role = msg.get("role") if isinstance(msg, dict) else getattr(msg, "role", None)
        if role == "assistant":
            content = msg.get("content") if isinstance(msg, dict) else getattr(msg, "content", None)
            if content:
                return content
    return "(无文本回复)"


if __name__ == "__main__":
    messages = [{"role": "system", "content": system_prompt}]

    while True:
        try:
            query = input("\033[36ms01 >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            break

        if query.strip().lower() in ("q", "exit", ""):
            break

        messages.append({"role": "user", "content": query})
        agent_loop(messages)
        response_content = last_assistant_text(messages)

        print(response_content)
