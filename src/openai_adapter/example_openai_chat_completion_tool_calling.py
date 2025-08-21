from openai import OpenAI
import json

client = OpenAI(base_url="http://127.0.0.1:8001", api_key="dummy")

tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get current weather by city",
        "parameters": {
            "type": "object",
            "properties": { "city": { "type": "string" } },
            "required": ["city"]
        }
    }
}]

messages = [{"role": "user", "content": "What's the weather in Los Angeles?"}]

# 1) Ask model (with tools)
res = client.chat.completions.create(
    model="gemma-1b",
    messages=messages,
    tools=tools,
    tool_choice="auto",
)

choice = res.choices[0]
tool_calls = getattr(choice.message, "tool_calls", None)

if tool_calls:
    # 2) Execute each tool
    for tc in tool_calls:
        name = tc.function.name
        args = json.loads(tc.function.arguments)

        # Your actual tool execution goes here:
        if name == "get_weather":
            tool_result = {"temp_f": 77, "condition": "Sunny"}  # ← stub

        # 3) Append assistant tool_call msg + the tool result message
        messages.append({"role": "assistant", "content": None, "tool_calls": [tc.model_dump()]})
        messages.append({"role": "tool", "tool_call_id": tc.id, "content": json.dumps(tool_result)})

    # 4) Ask model again with tool results
    res2 = client.chat.completions.create(model="gemma-1b", messages=messages)
    print(res2.choices[0].message.content)
else:
    print(choice.message.content)
