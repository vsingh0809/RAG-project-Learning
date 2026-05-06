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
            "name":"calculate",
            "description":"when user ask to calculate operation",
            "parameters":
            {
                "type":"object",
                "properties":
                {
                    "operation":{
                        "type":"string",
                        "enum": ["add", "subtract", "multiply", "divide"],
                    },
                    "a":{
                        "type":"number",
                    },
                    "b":{
                         "type": "number",
                    }
                },
                "required":["operation","a","b"]
            }
        }
    },
    {
         "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "Get the current date and time. Use when user asks about time or date.",
            "parameters": {
                "type": "object",
                "properties": {
                    "timezone": {
                        "type": "string",
                        "description": "Timezone name e.g. 'Asia/Kolkata', 'UTC'",
                    }
                },
             "required": ["timezone"]
            }
        }
    },
    {
        "type":"function",
        "function":{
            "name":"get_weather",
            "description":"Get current weather for a city. Use when user asks about weather.",
            "parameters":{
                "type":"object",
                "properties":{
                    "city":{
                        "type":"string",
                        "description": "City name e.g. 'Mumbai', 'Delhi'"
                    }
                },
                "required":["city"]
            }
        }
    },
    {
         "type":"function",
         "function":{
              "name":"convert_length",
              "description":"user when ask to convert the length",
              "parameters":{
                   "type":"object",
                   "properties":{
                        "value":{
                             "type":"number"
                        },
                        "unit_from":{
                             "type":"string"
                        },
                        "unit_to":{
                             "type":"string"
                        }

                   }
              }
         }
         
    }
]

def convert_length(value, unit_from, unit_to):
    # Base unit is meters (m)
    factors = {
        'mm': 0.001,
        'cm': 0.01,
        'm': 1.0,
        'km': 1000.0,
        'inch': 0.0254,
        'ft': 0.3048
    }
    
    # Convert input to base unit (meters), then to target unit
    meters = value * factors[unit_from]
    result = meters / factors[unit_to]
    return result

def calculate(operation:str,a:float,b:float)->str:
    if operation == "add":      return a + b
    if operation == "subtract": return a - b
    if operation == "multiply": return a * b
    if operation == "divide":
        if b == 0:
            raise ValueError("Cannot divide by zero")
        return a / b

def get_weather(city:str):
    # In real app: call OpenWeatherMap API
    # For practice: return fake data so you focus on the flow
    fake_data = {
        "Mumbai": {"temp": 32, "condition": "Humid", "humidity": 85},
        "Delhi": {"temp": 38, "condition": "Sunny", "humidity": 30},
        "Nagpur": {"temp": 36, "condition": "Hot", "humidity": 40},
    }
    data = fake_data.get(city, {"temp": random.randint(20,40), "condition": "Clear", "humidity": 50})
    return f"Weather in {city}: {data['temp']}°C, {data['condition']}, Humidity {data['humidity']}%"

def get_current_time(timezone: str) -> str:
    """Get current time in the specified timezone."""
    try:
        tz = ZoneInfo(timezone)
        now = datetime.datetime.now(tz)
        return f"Current time in {timezone}: {now.strftime('%Y-%m-%d %H:%M:%S')}"
    except Exception as e:
        return f"Error with timezone '{timezone}': {e}"


Tool_Registry={
    "calculate":calculate,
    "get_current_time":get_current_time,
    "get_weather":get_weather,
    "convert_length":convert_length
}

# ── KEY DIFFERENCE: handle MULTIPLE tool calls ───────────

def process_too_calls(tool_calls:list)->list[dict]:
     """
    Run ALL tool calls the LLM requested.
    Returns list of tool result messages.
    """
     results=[]

     for tool_call in tool_calls:
          name=tool_call.function.name
          args=json.loads(tool_call.function.arguments)
          print(f"  → Calling: {name}({args})")
          result = Tool_Registry[name](**args)
          print(f"  → Result: {result}")
        
        # Each tool result needs its own message with matching tool_call_id
          results.append({
             "role":"tool",
             "tool_call_id":tool_call.id,
             "content":str(result)
        })
     return results


def ask_parallel(question:str):
    """Send a question, let LLM pick the right tool, get answer."""
    print(f"\n User: {question}")
    print("-" * 50)

    response=client.chat.completions.create(
        model="llama-3.3-70b-versatile",
    messages=[{"role": "user", "content": question}],
    tools=tools,
    tool_choice='auto',
   # parallel_tool_calls=False
    )

    message=response.choices[0].message
    
    num_calls=len(message.tool_calls) if message.tool_calls else 0
    print(f"  → LLM requested {num_calls} tool call(s) simultaneously")

    if not message.tool_calls:
                # LLM answered directly without a tool
                print(f"  → No tool needed")
                print(f"Answer: {message.content}")
                return


     # Send result back, get final answer
    result=process_too_calls(message.tool_calls)
    print(f"  → Tool result: {result}")

    final=client.chat.completions.create(
         model="llama-3.3-70b-versatile",
         messages=[
            {"role": "user", "content": question},
            message,
            *result
        ],
        tools=tools
    )

    print(f"Answer: {final.choices[0].message.content}")


ask_parallel("what is 5 plus 3 and How is the weather in Mumbai today and what is the 1km in meter?")
#ask_parallel("what is the 1km in meter")
ask_parallel("What time is it in India right now?")
ask_parallel("How is the weather in Mumbai today?")
#ask_parallel("What is the capital of India?")
