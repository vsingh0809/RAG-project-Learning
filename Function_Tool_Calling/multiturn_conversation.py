import json
import os
from groq import Groq
import random
import datetime
from zoneinfo import ZoneInfo
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

tools=[
    {
        "type":"function",
        "function":{
            "name":"get_weather",
            "description":"when user ask for any city weather",
            "parameters":{
                "type":"object",
                "properties":{
                    "city":{
                        "type":"string"
                    }
                },
                "required":["city"]
            }
        }
    },
    {
        "type":"function",
        "function":{
            "name":"calculate",
            "description":"Perform arithmetic on two numbers",
            "parameters":{
                "type":"object",
                "properties":{
                    "a":{"type":"number"},
                    "b":{"type":"number"},
                    "operation":{
                        "type":"string",
                        "enum":["add","substract","multiply","divide"]
                    }
                },
                "required":["operation","a","b"]
            }
        }
    }
]

def calculate(operation, a, b):
    ops = {"add": a+b, "subtract": a-b, "multiply": a*b, "divide": a/b}
    return ops[operation]

def get_weather(city):
    data = {"Mumbai": "32°C Humid", "Delhi": "38°C Sunny", "Nagpur": "36°C Hot"}
    return data.get(city, f"{city}: 30°C Clear")

Tool_Registry={
    "calculate":calculate,
    "get_weather":get_weather
}

conversation=[
    {
        "role":"system",
        "content":"You are a helpful assistant with access to tools. Use them when needed. Remember previous context in the conversation."
    }
]

def chat(user_input:str)->str:
    """
    Send a message. Handle tool calls if needed. Return final response.
    Conversation history is maintained across calls.
    """
    conversation.append({"role":"user","content":user_input})
    print(f"\nUser: {user_input}")

    # Keep looping until LLM gives a final text response (no more tool calls)

    while True:
        response=client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=conversation,
            tools=tools,
            tool_choice='auto'
        )


        message=response.choices[0].message
        conversation.append(message)

        if not message.tool_calls:
            # LLM gave a final text response — we're done
            print(f"Assistant: {message.content}")
            return message.content
        
        # LLM wants to call tools — process them
        print(f"  [calling {len(message.tool_calls)} tool(s)]")

        for tool_call in message.tool_calls:
            name = tool_call.function.name
            args=json.loads(tool_call.function.arguments)
            print(f"  → {name}({args})")
            result = str(Tool_Registry[name](**args))
            print(f"  ← {result}")

            conversation.append(
                {
                "role":"tool",
                "tool_call_id":tool_call.id,
                "content":result
                }
            )

# ── MULTI-TURN TEST ──────────────────────────────────────
print("=" * 60)
print("MULTI-TURN CONVERSATION WITH TOOL MEMORY")
print("=" * 60)

# Turn 1: simple tool call
chat("What is 250 multiplied by 18?")

# Turn 2: follow-up referencing the result
chat("Now divide that result by 5")     # LLM remembers 4500 from Turn 1

# Turn 3: different tool
chat("What's the weather in Mumbai?")

# Turn 4: question combining memory + tool
chat("If the temperature in Mumbai goes up by the result from turn 1 divided by 1000, what would it be?")
        
# Turn 5: pure memory — no tool needed
chat("What calculations have we done so far in this conversation?")

print("\n" + "=" * 60)
print(f"Total messages in conversation history: {len(conversation)}")
print("=" * 60)


# ── INSPECT THE FULL HISTORY ─────────────────────────────
print("\nFull conversation history structure:")
for i, msg in enumerate(conversation):
    role = msg.get("role") if isinstance(msg, dict) else msg.role
    if hasattr(msg, 'content'):
        content = str(msg.content)[:60] if msg.content else "[tool call]"
    else:
        content = str(msg.get("content",""))[:60]
    print(f"  [{i}] {role}: {content}...")
