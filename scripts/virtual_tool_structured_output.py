"""
Virtual Tool Pattern Demo
=========================
演示如何用"虚拟工具 (Virtual Tool)"来规范 LLM 的结构化返回格式。

运行前需要设置环境变量：
    $env:OPENAI_API_KEY = "your-key"
    $env:OPENAI_BASE_URL = "https://api.openai.com/v1"  # 可选，替换为代理地址

运行方式：
    python demo_virtual_tool.py
"""

import json
import os

from openai import OpenAI

client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
    base_url=os.environ.get("OPENAI_BASE_URL"),
)

MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

# ─────────────────────────────────────────────────────────────────────────────
# 示例任务：让 LLM 分析一段文本的情绪，并返回结构化结果
# ─────────────────────────────────────────────────────────────────────────────
TEXT_TO_ANALYZE = """
今天发布了新版本，上线后用户反馈很好，但是有两个已知 Bug 需要尽快修复。
下周一必须完成紧急修复并重新部署，另外文档更新可以排到下周三。
"""


# ─────────────────────────────────────────────────────────────────────────────
# 方法 A：传统自由文本方式（脆弱，难以可靠解析）
# ─────────────────────────────────────────────────────────────────────────────
def approach_a_free_text():
    print("=" * 60)
    print("【方法 A】自由文本方式（传统做法，脆弱）")
    print("=" * 60)

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "你是一个任务分析助手。"
                    "请用 JSON 格式回复，包含 sentiment（positive/negative/neutral）"
                    "和 urgent_tasks（紧急任务列表）两个字段。"
                ),
            },
            {"role": "user", "content": f"分析以下文本：\n\n{TEXT_TO_ANALYZE}"},
        ],
    )

    raw = response.choices[0].message.content
    print(f"\nLLM 原始返回（字符串）：\n{raw}\n")

    # 问题所在：LLM 可能在 JSON 外包裹 markdown 代码块，导致 json.loads 直接失败
    try:
        # 尝试暴力剥离 markdown 代码块
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = "\n".join(cleaned.split("\n")[1:])  # 去掉第一行 ```json
        if cleaned.endswith("```"):
            cleaned = "\n".join(cleaned.split("\n")[:-1])  # 去掉最后一行 ```
        result = json.loads(cleaned)
        print(f"解析结果（需要手动清洗才能成功）：{result}")
    except json.JSONDecodeError as e:
        print(f"解析失败（这很常见！）: {e}")

    print("\n痛点：")
    print("  - LLM 可能输出 ```json ... ``` 包裹的 markdown")
    print("  - LLM 可能在 JSON 前后加解释性文字")
    print("  - 字段名可能拼错、类型可能不一致")
    print("  - 你需要写大量防御性代码来清洗和容错")


# ─────────────────────────────────────────────────────────────────────────────
# 方法 B：虚拟工具方式（Nanobot 的做法，健壮且精确）
# ─────────────────────────────────────────────────────────────────────────────

# 第一步：定义"虚拟工具" —— 这个工具在服务端根本不存在，只是一个 JSON Schema 约束
ANALYSIS_TOOL = [
    {
        "type": "function",
        "function": {
            "name": "report_analysis",  # 虚拟工具名称，随意起名
            "description": "Report the analysis result of the given text.",
            "parameters": {
                "type": "object",
                "properties": {
                    "sentiment": {
                        "type": "string",
                        "enum": ["positive", "negative", "neutral", "mixed"],
                        "description": "Overall sentiment of the text.",
                    },
                    "urgent_tasks": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of urgent tasks mentioned in the text.",
                    },
                    "deadline": {
                        "type": "string",
                        "description": "The earliest deadline mentioned, in YYYY-MM-DD format if possible.",
                    },
                },
                "required": [
                    "sentiment",
                    "urgent_tasks",
                ],  # 强制 LLM 必须提供这两个字段
            },
        },
    }
]


def approach_b_virtual_tool():
    print("\n" + "=" * 60)
    print("【方法 B】虚拟工具方式（Nanobot 的做法，健壮）")
    print("=" * 60)

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": "你是一个任务分析助手。请调用工具汇报你的分析结果。",
            },
            {"role": "user", "content": f"分析以下文本：\n\n{TEXT_TO_ANALYZE}"},
        ],
        tools=ANALYSIS_TOOL,
        tool_choice="required",  # 关键：强制 LLM 必须调用工具，不能输出自由文本
    )

    message = response.choices[0].message
    print(f"\nfinish_reason: {response.choices[0].finish_reason}")  # 应为 "tool_calls"
    print(f"tool_calls 数量: {len(message.tool_calls)}")

    # 第二步：直接读取 arguments —— 始终是合法的 JSON，无需任何清洗
    tool_call = message.tool_calls[0]
    print(f"\n工具名称: {tool_call.function.name}")
    print(f"原始 arguments（字符串）:\n{tool_call.function.arguments}")

    args: dict = json.loads(tool_call.function.arguments)  # 这里永远不会失败

    print(f"\n解析后的结构化数据:")
    print(f"  情绪:     {args['sentiment']}")
    print(f"  紧急任务: {args['urgent_tasks']}")
    print(f"  最早截止: {args.get('deadline', '未提及')}")

    print("\n优势：")
    print("  - arguments 始终是合法 JSON，json.loads 永远成功")
    print("  - 字段名、类型、enum 值均由 schema 强制约束")
    print("  - required 字段保证不缺失")
    print("  - 无需写任何正则/清洗逻辑")
    return args


# ─────────────────────────────────────────────────────────────────────────────
# 方法 C：进阶 —— tool_choice 指定工具名（比 "required" 更严格）
# ─────────────────────────────────────────────────────────────────────────────
def approach_c_force_specific_tool():
    print("\n" + "=" * 60)
    print("【方法 C】进阶：tool_choice 指定具体工具名（最严格）")
    print("=" * 60)

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": "你是一个任务分析助手。请调用工具汇报你的分析结果。",
            },
            {"role": "user", "content": f"分析以下文本：\n\n{TEXT_TO_ANALYZE}"},
        ],
        tools=ANALYSIS_TOOL,
        # 指定必须调用的工具名，比 "required" 更严格
        # 即使定义了多个工具，也只会调用这一个
        tool_choice={"type": "function", "function": {"name": "report_analysis"}},
    )

    args = json.loads(response.choices[0].message.tool_calls[0].function.arguments)
    print(f"\n强制指定工具名后的结果: {json.dumps(args, ensure_ascii=False, indent=2)}")

    print("\n适用场景：")
    print("  - 你定义了多个工具，但在特定流程中只想要某一个的输出")
    print("  - Nanobot heartbeat 中就是传入单个工具，LLM 必然调用它")


# ─────────────────────────────────────────────────────────────────────────────
# 核心机制解析
# ─────────────────────────────────────────────────────────────────────────────
def explain_mechanism():
    print("\n" + "=" * 60)
    print("【核心机制】为什么虚拟工具能规范输出格式？")
    print("=" * 60)
    print("""
  LLM 在 Tool Calling 协议下的行为：

  ┌─────────────────────────────────────────────────────────┐
  │  正常对话模式                                           │
  │  User → LLM → 自由文本（任何格式）                      │
  └─────────────────────────────────────────────────────────┘

  ┌─────────────────────────────────────────────────────────┐
  │  Tool Calling 模式（虚拟工具）                          │
  │                                                         │
  │  User + tool_definitions → LLM                         │
  │                              ↓                          │
  │                    LLM 内部意图：                        │
  │                    "我需要调用 report_analysis 工具"     │
  │                              ↓                          │
  │                    根据 JSON Schema 生成合规参数          │
  │                              ↓                          │
  │                    返回 tool_calls[].function.arguments  │
  │                    （始终是合法 JSON）                   │
  └─────────────────────────────────────────────────────────┘

  关键约束参数：
  ┌──────────────────────┬─────────────────────────────────┐
  │ tool_choice          │ 作用                            │
  ├──────────────────────┼─────────────────────────────────┤
  │ "auto"（默认）        │ LLM 自行决定是否调用工具         │
  │ "none"               │ 禁止工具调用，只输出文本          │
  │ "required"           │ 强制调用（任意一个工具）          │
  │ {"function": {name}} │ 强制调用指定工具（最严格）        │
  └──────────────────────┴─────────────────────────────────┘

  Nanobot heartbeat 中的应用：
    - 工具名: "heartbeat"
    - 强制字段: action（枚举 skip/run）
    - 可选字段: tasks（任务描述）
    - 没有传 tool_choice，但只定义了一个工具，LLM 会自动调用它
    - 比解析自由文本或 HEARTBEAT_OK 令牌更可靠
""")


# ─────────────────────────────────────────────────────────────────────────────
# 主程序
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if not os.environ.get("OPENAI_API_KEY"):
        print("错误：请先设置 OPENAI_API_KEY 环境变量")
        print("  PowerShell: $env:OPENAI_API_KEY = 'your-key'")
        raise SystemExit(1)

    explain_mechanism()  # 先看原理图
    approach_a_free_text()  # 对比：传统自由文本
    approach_b_virtual_tool()  # 核心：虚拟工具
    approach_c_force_specific_tool()  # 进阶：指定工具名

    print("\n" + "=" * 60)
    print("Demo 完成！")
    print("=" * 60)
